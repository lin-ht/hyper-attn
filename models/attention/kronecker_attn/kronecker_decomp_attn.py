import math

import torch
from attention.hyper_attn import utils
from attention.hyper_attn.utils import exact_attention_xformers as exact_attention


def nd_to_1d_index(indices, shape):
    # Convert indices to tensor
    # indices = torch.tensor(indices)
    # Compute strides
    count = torch.prod(torch.tensor(shape[:-1], device=indices.device)).item()
    indices_1d = torch.arange(count, device=indices.device).unsqueeze_(dim=-1) * shape[
        -1
    ] + indices.reshape(count, -1)

    return indices_1d.reshape(-1)


class KroneckerDecompAttention(torch.nn.Module):
    def __init__(
        self,
        attn_calculator,
        sampling_ratio=1 / 30,
    ):
        """
        Kronecker attention module.
        Input parameters:
            - attn_calculator: the calculator to compute attention result and log-sum-exp
            which should be called with input of (query, key, value, scale=None, causal=False, return_lse=False)
        """
        super().__init__()
        self.attn_calculator = attn_calculator
        self.sampling_ratio = sampling_ratio

    @staticmethod
    def estimateKroneckerDecomps(
        query,
        key,
        value,
        n_query_groups,
        n_key_groups,
        config={
            "mode": "median",
            "significant_channels": 3,
        },
    ):
        batch_size, head_size, n_query, dim = query.shape
        # n_key = key.shape[2]
        # n_gp = [n_query // n_query_groups, n_key // n_key_groups]

        query_gps = query.reshape(batch_size, head_size, n_query_groups, -1, dim)
        key_gps = key.reshape(batch_size, head_size, n_key_groups, -1, dim)

        if config["mode"] == "median":
            sig_chns = config["significant_channels"]
            sig_weights = torch.tensor(
                [10 ** (i - 1) for i in range(sig_chns, 0, -1)],
                dtype=query.dtype,
                device=query.device,
            )
            sig_weights = sig_weights.reshape(sig_chns, 1)

            def get_median_group(input_gps, sig_chns, sig_weights):
                b, h, g, n, dim = input_gps.shape
                gp_mean = input_gps.mean(dim=2, keepdim=True)
                gp_topk = torch.topk(gp_mean, sig_chns, dim=-1).indices
                gp_topk_indices = gp_topk.expand(*input_gps.shape[:-1], -1)
                gp_topk_indices_1d = nd_to_1d_index(gp_topk_indices, input_gps.shape)

                input_sig = input_gps.view(-1, 1)[gp_topk_indices_1d].reshape(
                    *input_gps.shape[:-1], sig_chns
                )
                # input_val shape: [b, h, n, g]
                # Note: torch.median is not implemented for bfloat16
                input_val = (
                    torch.matmul(input_sig, sig_weights)
                    .squeeze_(-1)
                    .transpose_(-2, -1)
                    .to(torch.float32)
                )

                gp_median_indices = torch.median(
                    input_val, dim=-1, keepdim=True
                ).indices

                # input_gps shape: [b, h, n, g, dim]
                input_gps_t = input_gps.transpose(-3, -2)
                gp_median_indices_1d = nd_to_1d_index(
                    gp_median_indices, input_gps_t.shape[:-1]
                )
                input_rep = input_gps_t.reshape(-1, dim)[gp_median_indices_1d]
                return input_rep.reshape(b, h, 1, -1, dim)

            query_rep = get_median_group(query_gps, sig_chns, sig_weights)
            key_rep = get_median_group(key_gps, sig_chns, sig_weights)
        else:
            query_rep = query_gps.mean(dim=2, keepdim=True)
            key_rep = key_gps.mean(dim=2, keepdim=True)

        query_residual = query_gps - query_rep
        key_residual = key_gps - key_rep
        # query_rep shape: [batch_size, head_size, 1, n_query_gp, dim]
        # query_residual shape: [batch_size, head_size, n_query_groups, n_query_gp, dim]
        return query_rep, query_residual, key_rep, key_residual

    @staticmethod
    def computeCorrMatrix(query, key):
        n_query, dim = query.shape[-2:]
        n_key = key.shape[-2]

        q_ = query.reshape([-1, n_query, dim])
        # k_ = einops.rearrange(k_sample, "b h l d -> (b h) d l")
        k_ = key.reshape([-1, n_key, dim]).transpose(-1, -2)
        corr_matrix = torch.bmm(q_, k_)

        return corr_matrix.reshape(*query.shape[:-2], n_query, -1)

    @staticmethod
    def matrixSoftmax(matrix, scale=None):
        if scale is not None:
            matrix = matrix * scale
        mat = torch.nn.functional.softmax(matrix, dim=-1)
        den = torch.logsumexp(matrix, dim=-1, keepdim=True)
        return mat, den

    @staticmethod
    def normGuidedSampling(val, ratio, max_sample=None):
        n = val.shape[-2]
        ratio = min(max(0.0, ratio), 1.0)
        m = max(1, int(n * ratio))
        if max_sample is not None:
            m = min(m, max_sample)  # clamp the sample size

        # Random sample according to the norm of the input tensor
        sample_prob = val.norm(p=2, dim=-1).reshape(-1, n) + torch.finfo(val.dtype).eps
        sample_set = torch.multinomial(sample_prob, m, replacement=False)
        return sample_set.reshape(val.shape[:-2] + (m,))

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

        # Groudtruth for debug:
        # attn_t, lse_t = self.attn_calculator(
        #     query,
        #     key,
        #     value,
        #     scale,
        #     causal=False,
        #     return_lse=return_lse,
        # )

        q_rep, q_res, k_rep, k_res = KroneckerDecompAttention.estimateKroneckerDecomps(
            query, key, value, n_query_groups, n_key_groups
        )
        # w_rep shape = [batch_size, head_size, 1, n_query_gp, n_key_gp]
        w_rep = KroneckerDecompAttention.computeCorrMatrix(q_rep, k_rep)
        # lse_rep shape = [batch_size, head_size, 1, n_query_gp, 1]
        attn_rep, lse_rep = KroneckerDecompAttention.matrixSoftmax(w_rep, scale)

        # split value tensor into n_key_groups along the 2nd last dimension
        v_gps = value.chunk(n_key_groups, dim=-2)
        # concatenate the v_gps along the last dimension
        # v_gps_cat shape = [batch_size, head_size, n_key_gp, n_key_groups*dim]
        v_gps_cat = torch.cat(v_gps, dim=-1)
        # Make v_gps_cat shape = [batch_size, head_size, 1,  n_key_gp, n_key_groups*dim]
        v_gps_cat.unsqueeze_(2)

        # Step 0: Compute the decomposable Kronecker product component P0
        # Make num_p0 shape = [batch_size, head_size, 1, n_query_gp, n_key_groups*dim]
        attn_p0 = torch.bmm(
            attn_rep.reshape(-1, *attn_rep.shape[-2:]),
            v_gps_cat.reshape(-1, *v_gps_cat.shape[-2:]),
        )
        # Make num_p0 shape = [batch_size, head_size, 1, n_query_gp, dim]
        attn_p0 = attn_p0.reshape(batch_size, head_size, 1, -1, n_key_groups, dim).mean(
            dim=-2, keepdim=False
        )
        # den_p0 shape = [batch_size, head_size, 1, n_query_gp, 1]
        lse_p0 = lse_rep + math.log(n_key_groups)

        # Step 1: Column sampling
        # Step 1.0: Key residule guided column sampling
        # k_res shape: [batch_size, head_size, n_key_groups, n_key_gp, dim]
        k_sub_ind_gps = KroneckerDecompAttention.normGuidedSampling(
            k_res, ratio=self.sampling_ratio
        )
        k_sub_ind = k_sub_ind_gps.view(batch_size * head_size, n_key_groups, -1)

        k_rep_sub = utils.indexing(
            torch.stack(
                [k_rep.view(batch_size * head_size, -1, dim)] * n_key_groups, dim=1
            ),
            k_sub_ind,
        ).view(batch_size, head_size, -1, dim)
        k_sub = utils.indexing(
            key.view(batch_size * head_size, n_key_groups, -1, dim), k_sub_ind
        ).view(batch_size, head_size, -1, dim)
        v_sub = utils.indexing(
            value.view(batch_size * head_size, n_key_groups, -1, dim), k_sub_ind
        ).view(batch_size, head_size, -1, dim)

        # q_rep shape: [batch_size, head_size, 1, n_query_gp, dim]
        # num_p1_del shape: [batch_size, head_size, n_query_gp, dim]
        q_rep_ = q_rep.view(batch_size, head_size, -1, dim)
        attn_p1_del, lse_p1_del = self.attn_calculator(
            q_rep_,
            k_rep_sub,
            v_sub,
            scale,
            causal=False,
            return_lse=return_lse,
        )
        # num_p1_del shape: [batch_size, head_size, 1, n_query_gp, dim]
        # den_p1_del shape: [batch_size, head_size, 1, n_query_gp, 1]
        attn_p1_del.unsqueeze_(2)
        lse_p1_del.unsqueeze_(2)

        attn_p1_add, lse_p1_add = self.attn_calculator(
            q_rep_,
            k_sub,
            v_sub,
            scale,
            causal=False,
            return_lse=return_lse,
        )
        attn_p1_add.unsqueeze_(2)
        lse_p1_add.unsqueeze_(2)

        # Step 1.1: remove the sampled column from the P0 result
        attn_p1, lse_p1 = utils.subtract_self_attentions(
            attn_p0, lse_p0, attn_p1_del, lse_p1_del
        )
        # Step 1.2: add the sampled column to the modified P0 result
        attn_p1, lse_p1 = utils.add_self_attentions(
            attn_p1, lse_p1, attn_p1_add, lse_p1_add
        )

        # Step 2: Row sampling
        # Step 2.0: Query residule guided row sampling
        # q_res shape: [batch_size, head_size, n_query_groups, n_query_gp, dim]
        q_sub_ind_gps = KroneckerDecompAttention.normGuidedSampling(
            q_res, ratio=self.sampling_ratio
        )
        q_sub_ind = q_sub_ind_gps.view(batch_size * head_size, n_query_groups, -1)
        q_sub = utils.indexing(
            query.view(batch_size * head_size, n_query_groups, -1, dim), q_sub_ind
        ).view(batch_size, head_size, -1, dim)
        attn_p2_new, lse_p2_new = self.attn_calculator(
            q_sub,
            key,
            value,
            scale,
            causal=False,
            return_lse=return_lse,
        )

        attn = torch.cat([attn_p1] * n_query_groups, dim=2)
        q_sub_ind_1d = nd_to_1d_index(q_sub_ind, q_res.shape[:-1])
        attn_view = attn.view(-1, dim)

        attn_view[q_sub_ind_1d] = attn_p2_new.view(-1, dim)
        attn = attn.reshape(batch_size, head_size, n_query, dim)

        if not return_lse:
            return attn
        else:
            lse = torch.cat([lse_p1] * n_query_groups, dim=2)
            lse_view = lse.view(-1, 1)
            lse_view[q_sub_ind_1d] = lse_p2_new.view(-1, 1)
            lse = lse.reshape(batch_size, head_size, n_query, 1)
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
    sampling_ratio=0.25,
):
    batch_size = 2
    head_size = 4
    dim = 16
    n_query = n_query_group_size * n_query_groups
    n_key = n_key_group_size * n_key_groups

    n_gp = [n_query // n_query_groups, n_key // n_key_groups]

    q_rep_gp = torch.randn(
        batch_size, head_size, 1, n_gp[0], dim, dtype=torch.bfloat16, device="cuda"
    )
    k_rep_gp = torch.randn(
        batch_size, head_size, 1, n_gp[1], dim, dtype=torch.bfloat16, device="cuda"
    )

    ratio = min(max(0.0, sampling_ratio), 1.0)
    n_q_res_gp = max(1, int(n_query_group_size * ratio))
    n_k_res_gp = max(1, int(n_key_group_size * ratio))

    # Hack to have same probability for each key column
    q_sample_prob = torch.ones(1, device=q_rep_gp.device).as_strided_(
        (batch_size * head_size * n_query_groups, n_gp[0]), (0, 0)
    )
    q_sample_set = torch.multinomial(
        q_sample_prob, n_q_res_gp, replacement=False
    ).reshape(batch_size, head_size, n_query_groups, -1)
    q_indices_1d = nd_to_1d_index(
        q_sample_set, (batch_size, head_size, n_query_groups, n_gp[0])
    )

    k_sample_prob = torch.ones(1, device=k_rep_gp.device).as_strided_(
        (batch_size * head_size * n_key_groups, n_gp[1]), (0, 0)
    )
    k_sample_set = torch.multinomial(
        k_sample_prob, n_k_res_gp, replacement=False
    ).reshape(batch_size, head_size, n_key_groups, -1)
    k_indices_1d = nd_to_1d_index(
        k_sample_set, (batch_size, head_size, n_key_groups, n_gp[1])
    )

    query = torch.cat([q_rep_gp] * n_query_groups, dim=2).reshape(-1, dim)
    key = torch.cat([k_rep_gp] * n_key_groups, dim=2).reshape(-1, dim)

    res_scale = 0.0
    query[q_indices_1d] += res_scale * torch.randn(
        batch_size * head_size * n_query_groups * n_q_res_gp,
        dim,
        dtype=query.dtype,
        device=query.device,
    )
    key[k_indices_1d] += res_scale * torch.randn(
        batch_size * head_size * n_key_groups * n_k_res_gp,
        dim,
        dtype=key.dtype,
        device=key.device,
    )

    query = query.reshape(batch_size, head_size, n_query, dim)
    key = key.reshape(batch_size, head_size, n_key, dim)

    print(f"query[0,0] = {query[0,0]}")
    print(f"key[0,0] = {key[0,0]}")
    value = torch.randn(
        batch_size, head_size, n_key, dim, dtype=key.dtype, device=key.device
    )

    return query, key, value, n_query_groups, n_key_groups


def load_real_qkv(path_prefix, n_query_groups, n_key_groups):
    query = torch.permute(torch.load(path_prefix + "q.pt"), (0, 2, 1, 3))
    key = torch.permute(torch.load(path_prefix + "k.pt"), (0, 2, 1, 3))
    value = torch.permute(torch.load(path_prefix + "v.pt"), (0, 2, 1, 3))
    return query, key, value, n_query_groups, n_key_groups


def create_uniform_kronecker_qkv(
    path_prefix, n_query_groups, n_key_groups, sampling_ratio=1 / 40
):
    query, key, value, n_query_groups, n_key_groups = load_real_qkv(
        path_prefix, n_query_groups, n_key_groups
    )

    b, h, n_query, dim = query.shape
    n_key = key.shape[2]

    n_gp = [n_query // n_query_groups, n_key // n_key_groups]
    c_gp = [n_query_groups // 2, n_key_groups // 2]
    q_gp = query[..., c_gp[0] * n_gp[0] : c_gp[0] * n_gp[0] + n_gp[0], :]
    k_gp = key[..., c_gp[1] * n_gp[1] : c_gp[1] * n_gp[1] + n_gp[1], :]

    n_query_group_size = n_query // n_query_groups
    n_key_group_size = n_key // n_key_groups

    ratio = min(max(0.0, sampling_ratio), 1.0)
    n_q_res_gp = max(1, int(n_query_group_size * ratio))
    n_k_res_gp = max(1, int(n_key_group_size * ratio))

    # Hack to have same probability for each key column
    q_sample_prob = torch.ones(1, device=query.device).as_strided_(
        (b, h, 1, n_gp[0]), (0, 0, 0, 0)
    )
    q_sample_set = (
        torch.multinomial(q_sample_prob, n_q_res_gp, replacement=False)
        .reshape(b, h, 1, -1)
        .expand(-1, -1, n_query_groups, -1)
    )
    print(f"created q_sample_set[0,0,0] = {q_sample_set[0,0,0]}")

    k_sample_prob = torch.ones(1, device=key.device).as_strided_(
        (b, h, 1, n_gp[1]), (0, 0, 0, 0)
    )
    k_sample_set = (
        torch.multinomial(k_sample_prob, n_k_res_gp, replacement=False)
        .reshape(b, h, 1, -1)
        .expand(-1, -1, n_key_groups, -1)
    )
    print(f"created k_sample_set[0,0,0] = {k_sample_set[0,0,0]}")

    query_ = torch.stack([q_gp] * n_query_groups, dim=2)
    key_ = torch.stack([k_gp] * n_key_groups, dim=2)

    query_[q_sample_set] = query.reshape(b, h, n_query_groups, -1, dim)[q_sample_set]
    key_[q_sample_set] = key.reshape(b, h, n_key_groups, -1, dim)[k_sample_set]

    return query, key, value, n_query_groups, n_key_groups


def test_kronecker_attn(
    query,
    key,
    value,
    n_query_groups,
    n_key_groups,
    sampling_ratio=1 / 30,
    threshold=0.1,
):
    batch_size, head_size, n_query, dim = query.shape
    attn_module = KroneckerDecompAttention(attn_calculator_func, sampling_ratio)

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

if __name__ == "__main__":
    # Set the seed
    torch.manual_seed(9)

    qkv_id = 1
    data = load_random_qkv()
    test_kronecker_attn(*data, sampling_ratio=0.5, threshold=0.5)

    # data = load_real_qkv(QKV_LIST[qkv_id], 6, 6)
    # data = create_uniform_kronecker_qkv(QKV_LIST[qkv_id], 6, 6)
    # test_kronecker_attn(*data, sampling_ratio=1 / 30, threshold=0.5)
    print("All tests passed!")
