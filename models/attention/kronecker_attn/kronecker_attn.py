import random

import einops
import torch


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
        A_{pm x qn} = S_{m x n} ⊗ B_{p x q}
            [s_00*B, s_01*B, ..., s_0n*B]
        =   [s_10*B, s_11*B, ..., s_1n*B]
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

        # attn_gps shape = [batch_size, head_size, n_gp[0], dim * n_key_groups]
        # lsp_gp shape = [batch_size, head_size, n_gp[0], 1]
        attn_rst = self.attn_calculator(
            q_gp, k_gp, v_gp, scale, causal=False, return_lse=return_lse
        )

        if not return_lse:
            attn_gp = attn_rst
        else:
            attn_gp, lse_gp = attn_rst

        # 2. Estimate the scaling factor matrix: S = C_{m x 1} * S'_{m x n}
        # where C_{m x 1} is the scaling factors for one picked key group.
        k_picked = random.randint(0, n_gp[1] - 1)
        k_sample = key[:, k_picked :: n_gp[1], :, :]

        q_ = einops.rearrange(query, "b l h d -> (b h) l d")
        k_ = einops.rearrange(k_sample, "b l h d -> (b h) d l")
        w_ = torch.bmm(q_, k_).reshape(batch_size, head_size, n_query, -1)
        w_gps = torch.cat(value.chunk(w_, dim=-2), dim=-1).transpose(-1, -2)
        w_gps = w_gps.reshape(batch_size, head_size, n_query_groups, n_key_groups, -1)
        norm_gps = w_gps.norm(p=2, dim=-1)
        # s_gps shape = [batch_size, head_size, n_query_groups, n_key_groups]
        s_gps = norm_gps / norm_gps[..., c_gp[0], c_gp[1]]

        # 3. Compute the kronecker product of S and B
        # attn_gps shape = [batch_size, head_size, 1, n_gp[0], n_key_groups, dim]
        attn_gps = torch.stack(attn_gp.chunk(n_key_groups, dim=-1), dim=-2).unsqueeze_(
            2
        )
        # s_gps shape = [batch_size, head_size, n_query_groups, 1, n_key_groups]
        s_gps = s_gps.unsqueeze_(3)
        s_gps_sum = torch.sum(s_gps, dim=-1, keepdim=True)

        # attn shape = [batch_size, head_size, n_query_groups, n_gp[0], dim]
        attn = torch.matmul(attn_gps, s_gps) / s_gps_sum
        attn = attn.reshape(batch_size, head_size, n_query, -1)
        if not return_lse:
            return attn
        else:
            # lsp_gp shape = [batch_size, head_size, n_gp[0], 1]
            lse = lse_gp.unsqueeze_(2) + torch.log(s_gps_sum.squeeze_(-1))
            return attn, lse