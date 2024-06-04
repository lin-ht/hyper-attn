import random

import torch
from attention.hyper_attn.utils import exact_attention_xformers as exact_attention


class KroneckerAttention(torch.nn.Module):
    def __init__(
        self,
        attn_calculator,
    ):
        """
        Kronecker attention module.
        Input parameters:
            - attn_calculator: the calculator to compute attention result and log-sum-exp
            which should be called with input of (query, key, value, scale=None, causal=False, return_lse=False)
        """
        super().__init__()
        self.attn_calculator = attn_calculator

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        n_query_groups: int,
        n_key_groups: int,
        scale=None,
        causal=False,
        return_lse=False,
    ):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        dim = key.shape[-1]
        scale = dim ** (-0.5) if scale is None else scale

        # Without causal masking
        if not causal:
            return self.forward_no_causal_mask(
                query, key, value, n_query_groups, n_key_groups, scale, return_lse
            )
        # With causal masking
        else:
            raise NotImplementedError("Causal masking is not implemented yet.")

    def forward_no_causal_mask(
        self, query, key, value, n_query_groups, n_key_groups, scale, return_lse=False
    ):
        """
        Return the approximated attention result A*V and log-sum-exp lse.
        by approximating the attention matrix A through kronecker product:
        A_{pm x qn} = softmax(S_{m x n} ⊗ B_{p x q})
                    [s_00*B, s_01*B, ..., s_0n*B]
        = softmax(  [s_10*B, s_11*B, ..., s_1n*B] )
                    [  ...,    ...,  ...,   ... ]
                    [s_m0*B, s_m1*B, ..., s_mn*B]
        where m = n_query_groups, n = n_key_groups, p = n_query/m, q = n_key/n.
        """

        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        if n_query_groups == 1 and n_key_groups == 1:
            return self.attn_calculator(
                query, key, value, scale, causal=False, return_lse=return_lse
            )

        n_gp = [n_query // n_query_groups, n_key // n_key_groups]
        # TODO: Ensure n_query / n_key is a multiple of n_query_groups, n_key_groups

        # 1. Estimate the grouped matrix: B * [V_0, V_1, ..., V_n]
        c_gp = [n_query_groups // 2, n_key_groups // 2]
        q_gp = query[:, :, c_gp[0] * n_gp[0] : c_gp[0] * n_gp[0] + n_gp[0], :]
        k_gp = key[:, :, c_gp[1] * n_gp[1] : c_gp[1] * n_gp[1] + n_gp[1], :]

        # split value tensor into n_key_groups along the 2nd last dimension
        v_gps = value.chunk(n_key_groups, dim=-2)
        # concatenate the v_gps along the last dimension
        v_gp = torch.cat(v_gps, dim=-1)

        # append 0 to q_gp and k_gp to match the shape of v_gp
        zero_dim = torch.zeros(
            q_gp.shape[:-1] + (v_gp.shape[-1] - q_gp.shape[-1],),
            device=q_gp.device,
            dtype=q_gp.dtype,
        )
        q_gp_ = torch.cat([q_gp, zero_dim], dim=-1)
        zero_dim = torch.zeros(
            k_gp.shape[:-1] + (v_gp.shape[-1] - k_gp.shape[-1],),
            device=q_gp.device,
            dtype=q_gp.dtype,
        )
        k_gp_ = torch.cat([k_gp, zero_dim], dim=-1)

        # attn_gps shape = [batch_size, head_size, n_gp[0], dim * n_key_groups]
        # lsp_gp shape = [batch_size, head_size, n_gp[0], 1]
        attn_rst = self.attn_calculator(
            q_gp_, k_gp_, v_gp, scale, causal=False, return_lse=return_lse
        )

        if not return_lse:
            attn_gp = attn_rst
        else:
            attn_gp, lse_gp = attn_rst

        # 2. Estimate the scaling factor matrix: S = C_{m x 1} * S'_{m x n}
        # where C_{m x 1} is the scaling factors for one picked key group.
        k_picked = random.randint(0, n_gp[1] - 1)
        k_sample = key[:, :, k_picked :: n_gp[1], :]

        q_ = query.reshape([-1, n_query, dim])
        # k_ = einops.rearrange(k_sample, "b h l d -> (b h) d l")
        k_ = k_sample.reshape([q_.shape[0], -1, dim]).transpose(-1, -2)
        w_ = torch.bmm(q_, k_).reshape(batch_size, head_size, n_query, -1)
        w_gps = torch.cat(w_.chunk(n_query_groups, dim=-2), dim=-1).transpose(-1, -2)
        w_gps = w_gps.reshape(batch_size, head_size, n_query_groups, n_key_groups, -1)
        norm_gps = w_gps.norm(p=2, dim=-1)
        # s_gps shape = [batch_size, head_size, n_query_groups, n_key_groups]
        s_gps = norm_gps / norm_gps[..., c_gp[0], c_gp[1]].unsqueeze_(-1).unsqueeze_(-1)
        print(f"s_gps_approx[0,0] = {s_gps[0,0]}")

        # 3. Approximate each entry Softmax(s_ij⊗B)V_k of the kronecker product
        # Approximate softmax(sB)V from known softmax(B)V
        # Taylor series expansion: exp(x) = 1 + x + x^2/2! + x^3/3! + ...
        # SUM_j exp(b_j)*v_j = SUM_j (1 + b_j + b_j^2/2! + b_j^3/3! + ...) * v_j
        # SUM_j exp(s*b_j)*v_j = SUM_j (1 + s*b_j + s^2*b_j^2/2 + s^3*b_j^3/6 + ...) * v_j
        # = (1 - s) SUM_j{v_j} +  SUM_j{s*(1 + b_j + s*b_j^2/2 + s^2*b_j^3/6 + ...) * v_j}
        # (if b_j is small and s is around 1.0, the higher order terms are negligible)
        # ~ (1 - s) SUM_j{v_j} +  SUM_j{s*(1 + b_j + s*b_j^2/2) * v_j}
        # = (1 - s) SUM_j{v_j} +  s * SUM_j{ (1 + b_j + b_j^2/2) * v_j} + s(s-1) * SUM_j{b_j^2/2 * v_j}
        # ~ (1 - s) SUM_j{v_j} + s * SUM_j{exp(b_j) * v_j} + s(s-1)/2 * SUM_j{b_j^2 * v_j}
        # ~ (1 - s) SUM_j{v_j} + s * SUM_j{exp(b_j) * v_j}
        # = (1 - s) SUM_j{v_j} + s * exp(lse) * (attn)
        # = SUM_j{v_j} + s * (SUM_j{exp(b_j) * v_j} - SUM_j{v_j})
        # The last one means the approximation result is a linear interpolation.
        # SUM_j{exp(s*b_j)}
        # ~ (1 - s) SUM_j{1} + s * SUM_j{exp(b_j)} + s(s-1)/2 * SUM_j{b_j^2}
        # ~ (1 - s) SUM_j{1} + s * SUM_j{exp(b_j)}
        # = (1 - s) SUM_j{1} + s * (exp(lse))
        # Therefore, we can approximate the softmax(sB)V by
        # {(1 - s) SUM_j{v_j} + s * exp(lse) * (attn)} / {(1 - s) SUM_j{1} + s * (exp(lse))}

        # 3.1 Compute approximation part 1: (1 - s) SUM_j{v_j}
        # Make v_gps shape = [batch_size, head_size, n_key_groups, n_gp[1], dim]
        v_gps = value.reshape(batch_size, head_size, n_key_groups, -1, dim)
        # Make v_gps shape = [batch_size, head_size, 1, n_key_groups, dim]
        v_gps = v_gps.sum(dim=-2, keepdim=False).unsqueeze_(2)

        # s_gps shape = [batch_size, head_size, n_query_groups, n_key_groups, 1]
        one_minus_s_gps = torch.clamp(1 - s_gps.unsqueeze_(-1), min=0.0)
        # num_part1 shape = [batch_size, head_size, n_query_groups, n_key_groups, dim]
        num_part1 = one_minus_s_gps * v_gps
        den_part1 = one_minus_s_gps * n_gp[1]
        num_part1.unsqueeze_(-2)
        den_part1.unsqueeze_(-2)

        # 3.2 Compute approximation part 2: s * exp(lse) * (attn)

        # lsp_gp shape = [batch_size, head_size, n_gp[0], 1]
        lse_gp_exp = lse_gp.exp()
        # attn_gps shape = [batch_size, head_size, n_gp[0], n_key_groups, dim]
        attn_gps = torch.stack(
            (lse_gp_exp * attn_gp).chunk(n_key_groups, dim=-1), dim=-2
        )
        # attn_gps shape = [batch_size, head_size, 1, n_key_groups, n_gp[0], dim]
        attn_gps = attn_gps.transpose_(-2, -3).unsqueeze_(2)
        # Make s_gps shape = [batch_size, head_size, n_query_groups, n_key_groups, 1, 1]
        s_gps.unsqueeze_(-1)
        # Make lsp_gp shape = [batch_size, head_size, 1, 1, n_gp[0], 1]
        lse_gp_exp.unsqueeze_(2).unsqueeze_(2)
        # num_part2 shape = [batch_size, head_size, n_query_groups, n_key_groups, n_gp[0], dim]
        num_part2 = s_gps * attn_gps
        den_part2 = s_gps * lse_gp_exp

        # 3.3 include more terms for higher order approximation if needed
        # ...

        # num_arr shape = [batch_size, head_size, n_query_groups, n_key_groups, n_gp[0], dim]
        num_arr = num_part1 + num_part2
        den_arr = den_part1 + den_part2

        # 4. merge all the entries into the final attn result
        num_agg = num_arr.sum(dim=-3)
        den_agg = den_arr.sum(dim=-3)
        attn = (num_agg / den_agg).reshape(batch_size, head_size, n_query, dim)

        if not return_lse:
            return attn
        else:
            lse = den_agg.log().reshape(batch_size, head_size, n_query, 1)
            return attn, lse


def attn_calculator_func(query, key, value, scale, causal=False, return_lse=False):
    attn, lse = exact_attention(query, key, value, scale, causal)
    if return_lse:
        return attn, lse
    else:
        return attn


def compute_error_ratio(
    attn_approx, lse_approx, exact_attn, exact_lse, ord="fro", log_prefix=""
):
    diff_attn = attn_approx - exact_attn
    diff_lse = lse_approx - exact_lse

    diff_attn_norm = torch.norm(diff_attn, p=ord, dim=-1)
    exact_attn_norm = torch.norm(exact_attn, p=ord, dim=-1)
    spectral_error_ratio = diff_attn_norm / exact_attn_norm

    lse_error_ratio = torch.abs(diff_lse) / torch.abs(exact_lse)

    attn_std_mean = torch.std_mean(spectral_error_ratio.reshape(-1), dim=-1)
    print(
        f"{log_prefix} Attn mean relative err: "
        f"{attn_std_mean[1].item()*100:4.02f}%, "
        f"relative err std: {attn_std_mean[0].item()*100:4.02f}% | "
    )

    lse_std_mean = torch.std_mean(lse_error_ratio.reshape(-1), dim=-1)
    print(
        f"{log_prefix} Lse  mean relative err: "
        f"{lse_std_mean[1].item()*100:4.02f}%, "
        f"relative err std: {lse_std_mean[0].item()*100:4.02f}% | "
    )

    return attn_std_mean, lse_std_mean


def load_random_qkv(
    n_query_groups=3,
    n_query_group_size=5,
    n_key_groups=2,
    n_key_group_size=4,
    std_scale=0.3,
):
    batch_size = 2
    head_size = 4
    dim = 16
    n_query = n_query_group_size * n_query_groups
    n_key = n_key_group_size * n_key_groups

    n_gp = [n_query // n_query_groups, n_key // n_key_groups]
    c_gp = [n_query_groups // 2, n_key_groups // 2]

    q_gp = torch.randn(
        batch_size, head_size, n_gp[0], dim, dtype=torch.bfloat16, device="cuda"
    )
    k_gp = torch.randn(
        batch_size, head_size, n_gp[1], dim, dtype=torch.bfloat16, device="cuda"
    )
    s_q = (
        torch.randn(
            batch_size,
            head_size,
            n_query_groups,
            1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * std_scale
        + 1.0
    )
    s_k = (
        torch.randn(
            batch_size,
            head_size,
            n_key_groups,
            1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * std_scale
        + 1.0
    )
    s_q[..., c_gp[0], :] = 1.0
    s_k[..., c_gp[1], :] = 1.0
    s_gps = s_q * s_k.transpose(-1, -2)
    print(f"s_gps[0,0] = {s_gps[0,0]}")

    query = (q_gp.unsqueeze_(2) * s_q.unsqueeze_(-1)).reshape(
        batch_size, head_size, n_query, dim
    )
    print(f"query[0,0] = {query[0,0]}")
    key = (k_gp.unsqueeze_(2) * s_k.unsqueeze_(-1)).reshape(
        batch_size, head_size, n_key, dim
    )
    print(f"key[0,0] = {key[0,0]}")
    value = torch.randn(
        batch_size, head_size, n_key, dim, dtype=torch.bfloat16, device="cuda"
    )

    return query, key, value, n_query_groups, n_key_groups


def load_real_qkv(path_prefix, n_query_groups, n_key_groups):
    query = torch.permute(torch.load(path_prefix + "q.pt"), (0, 2, 1, 3))
    key = torch.permute(torch.load(path_prefix + "k.pt"), (0, 2, 1, 3))
    value = torch.permute(torch.load(path_prefix + "v.pt"), (0, 2, 1, 3))
    return query, key, value, n_query_groups, n_key_groups


def create_uniform_kronecker_qkv(path_prefix, n_query_groups, n_key_groups):
    query = torch.permute(torch.load(path_prefix + "q.pt"), (0, 2, 1, 3))
    key = torch.permute(torch.load(path_prefix + "k.pt"), (0, 2, 1, 3))
    value = torch.permute(torch.load(path_prefix + "v.pt"), (0, 2, 1, 3))

    b, h, n_query, dim = query.shape
    n_key = key.shape[2]

    n_gp = [n_query // n_query_groups, n_key // n_key_groups]
    c_gp = [n_query_groups // 2, n_key_groups // 2]
    q_gp = query[..., c_gp[0] * n_gp[0] : c_gp[0] * n_gp[0] + n_gp[0], :]
    k_gp = key[..., c_gp[1] * n_gp[1] : c_gp[1] * n_gp[1] + n_gp[1], :]

    query = torch.stack([q_gp] * n_query_groups, dim=2).reshape(b, h, -1, dim)
    key = torch.stack([k_gp] * n_key_groups, dim=2).reshape(b, h, -1, dim)

    return query, key, value, n_query_groups, n_key_groups


def test_kronecker_attn(query, key, value, n_query_groups, n_key_groups, threshold=0.1):
    batch_size, head_size, n_query, dim = query.shape
    attn_module = KroneckerAttention(attn_calculator_func)

    scale = dim ** (-0.5)
    attn_approx, lse_approx = attn_module(
        query, key, value, n_query_groups, n_key_groups, scale, return_lse=True
    )

    torch.set_printoptions(sci_mode=False)

    print(f"attn_approx[0,0] = {attn_approx[0,0]}")
    print(f"lse_approx[0,0] = {lse_approx[0,0]}")
    attn_exact, lse_exact = attn_calculator_func(query, key, value, scale, False, True)
    print(f"attn_exact[0,0] = {attn_exact[0,0]}")
    print(f"lse_exact[0,0] = {lse_exact[0,0]}")

    torch.set_printoptions(profile="default")

    relative_attn_err_std_mean, relative_lse_err_std_mean = compute_error_ratio(
        attn_approx, lse_approx, attn_exact, lse_exact, ord="fro"
    )

    assert torch.all(relative_attn_err_std_mean[0] < threshold)
    assert torch.all(relative_attn_err_std_mean[1] < threshold)
    assert torch.all(relative_lse_err_std_mean[0] < threshold)
    assert torch.all(relative_lse_err_std_mean[1] < threshold)


QKV_LIST = [
    "tests/data/set_2/layer37_self_attn_it307_20240521162722916847_",
    "tests/data/set_3/layer04_self_attn_it999_20240521162129306018_",
]


import math

import numpy as np


def chebyshev_approximation_3(x, coefficients):
    """Approximate a function using a 3-degree Chebyshev polynomial."""
    T = [np.ones_like(x), x, 4 * x**3 - 3 * x]  # T[0] = 1, T[1] = x, T[2] = 4x^3 - 3x
    return sum(coeff * T_i for coeff, T_i in zip(coefficients, T))


def chebyshev_approximation_5(x, coefficients):
    """Approximate a function using a 5-degree Chebyshev polynomial."""
    T = [
        np.ones_like(x),
        x,
        4 * x**3 - 3 * x,
        0,
        16 * x**5 - 20 * x**3 + 5 * x,
    ]  # T[0] = 1, T[1] = x, T[2] = 4x^3 - 3x, T[3] = 0, T[4] = 16x^5 - 20x^3 + 5x
    return sum(coeff * T_i for coeff, T_i in zip(coefficients, T))


def chebyshev_approximation(x, coefficients):
    """Approximate a function using a Chebyshev polynomial."""
    T = [np.ones_like(x), x]  # T[0] = 1, T[1] = x
    for i in range(2, len(coefficients)):
        T.append(2 * x * T[-1] - T[-2])  # Recurrence relation
    return sum(coeff * T_i for coeff, T_i in zip(coefficients, T))


def taylor_expansion(x, degree):
    """Compute the Taylor series expansion of exp(x) up to a given degree."""
    return sum((x**n) / math.factorial(n) for n in range(degree + 1))


def test_chebyshev_approximation():
    """Test the Chebyshev approximation."""
    # Coefficients for the Chebyshev approximation
    # coefficients_3 = [1, 1, 0.5]  # Adjust these coefficients as needed
    coefficients_5 = [1, 1, 0.5, 1 / 6, 1 / 24]  # Adjust these coefficients as needed

    # Test the approximation
    x = np.linspace(-1, 1, 30)
    y_true = np.exp(x)
    y_approx = chebyshev_approximation_5(x, coefficients_5)
    y_approx_rec = chebyshev_approximation(x, coefficients_5)
    y_approx_taylor = taylor_expansion(x, 5)

    print("True values:", y_true)
    print("Approximation_taylor:", y_approx_taylor)
    print("Approximation:", y_approx)
    print("Approximation_rec:", y_approx_rec)


if __name__ == "__main__":
    # test_chebyshev_approximation()

    qkv_id = 1
    # load_random_qkv()
    data = load_real_qkv(QKV_LIST[qkv_id], 6, 6)
    # data = create_uniform_kronecker_qkv(QKV_LIST[qkv_id], 6, 6)
    test_kronecker_attn(*data, threshold=0.5)
    print("All tests passed!")
