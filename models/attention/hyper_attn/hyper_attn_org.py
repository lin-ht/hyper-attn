import math

import torch
from attention.hyper_attn.angular_lsh import AngularLSH
from attention.hyper_attn.utils import (
    add_self_attentions,
    exact_attention_cuda,
    exact_attention_xformers,
    indexing,
)
from einops import rearrange


class HyperAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim=64,
        lsh_num_projs=7,
        block_size=256,
        sample_size=256,
        min_seq_len=4096,
        impl="triton",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.cuda = impl == "cuda"
        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        scale=None,
        causal=False,
        return_lse=False,
    ):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = dim ** (-0.5) if scale is None else scale

        # Without causal masking
        if not causal:
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        # With causal masking
        else:
            if n_key <= self.min_seq_len:
                if self.cuda:
                    attn, lse = exact_attention_cuda(
                        query, key, value, scale, causal=True
                    )
                else:
                    attn, lse = exact_attention_xformers(
                        query, key, value, scale, causal=True
                    )
            else:
                # If n_query is odd we pad inputs by adding all-zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(
                        query, (0, 0, 0, 1), mode="constant", value=0.0
                    )
                    key = torch.nn.functional.pad(
                        key, (0, 0, 0, 1), mode="constant", value=0.0
                    )
                    value = torch.nn.functional.pad(
                        value, (0, 0, 0, 1), mode="constant", value=0.0
                    )

                q_bd = query.view(
                    batch_size, 2 * n_heads, query.shape[2] // 2, query.shape[-1]
                )
                k_bd = key.view(
                    batch_size, 2 * n_heads, key.shape[2] // 2, key.shape[-1]
                )
                v_bd = value.view(
                    batch_size, 2 * n_heads, key.shape[2] // 2, value.shape[-1]
                )

                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)

                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2] // 2 :, :],
                    key[:, :, : key.shape[2] // 2, :],
                    value[:, :, : key.shape[2] // 2, :],
                    scale,
                )

                attn_up, lse_up = (
                    attn_bd[:, :, : query.shape[2] // 2, :],
                    lse_bd[:, :, : query.shape[2] // 2, :],
                )
                attn_down, lse_down = add_self_attentions(
                    attn_bd[:, :, query.shape[2] // 2 :, :],
                    lse_bd[:, :, query.shape[2] // 2 :, :],
                    attn_unmasked,
                    lse_unmasked,
                )

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                # If n_query was odd exclude the last rows
                if n_query % 2:
                    attn = attn[:, :, :-1, :]
                    lse = lse[:, :, :-1, :]

        if not return_lse:
            return attn
        else:
            return attn, lse

    def forward_no_causal_mask(self, query, key, value, scale):
        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        if self.min_seq_len >= n_query:
            if self.cuda:
                return exact_attention_cuda(query, key, value, scale, causal=False)
            else:
                return exact_attention_xformers(query, key, value, scale, causal=False)

        # 1. Sorted block-diagonal via sortLSH
        _, query_sort_idx = torch.sort(
            self.lsh.hash(query), dim=2, stable=True
        )  # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash(key), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(
            query_sort_idx, dim=2, stable=True
        )  # for recovering the row order

        key_block_size = self.block_size

        query_sorted = indexing(query, query_sort_idx, key_block_size)
        # print(f"query_sorted[0, 0, 0, :] = {query_sorted.shape}")
        key_sorted = indexing(key, key_sort_idx, key_block_size)
        value_sorted = indexing(value, key_sort_idx, key_block_size)

        if key_block_size > 0:
            num_blocks = key_sorted.shape[2] // key_block_size
            query_block_size = query_sorted.shape[2] // num_blocks
            print(f"query_sort_idx[0, 0, -20:] = {query_sort_idx[0, 0, -20:]}")

            # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
            query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
            key_split_per_block = key_sorted.view(-1, 1, key_block_size, dim)
            value_split_per_block = value_sorted.view(-1, 1, key_block_size, dim)

            if self.cuda:
                attn_block, lse_block = exact_attention_cuda(
                    query_split_per_block,
                    key_split_per_block,
                    value_split_per_block,
                    softmax_scale=scale,
                    causal=False,
                )
            else:
                attn_block, lse_block = exact_attention_xformers(
                    query_split_per_block,
                    key_split_per_block,
                    value_split_per_block,
                    softmax_scale=scale,
                    causal=False,
                )

            if attn_block.shape[2] not in attn_block.stride():
                attn_block = attn_block.contiguous()
            attn_block = attn_block.view(
                batch_size, head_size, query_sorted.shape[2], -1
            )

            if lse_block.shape[2] not in lse_block.stride():
                lse_block = lse_block.contiguous()
            lse_block = lse_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            # When inputs are padded, then unpad them
            if query_sorted.shape[2] != n_query:  # query.shape[2]:
                attn_block, lse_block = (
                    attn_block[:, :, :n_query, :],
                    lse_block[:, :, :n_query, :],
                )
                query_sorted = query_sorted[:, :, :n_query, :]
                key_sorted = key_sorted[:, :, :n_key, :]
                value_sorted = value_sorted[:, :, :n_key, :]

        else:
            query_block_size = -1
            query_block_size = -1
            attn_block, lse_block = 0, 0

        # 2. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if (
            sample_size > 0
            and (n_query > query_block_size)
            and (n_key > key_block_size)
        ):
            # torch.manual_seed(42)
            print(
                f"initial_seed = {torch.initial_seed()}, status = {torch.get_rng_state()}"
            )
            sampled_set = torch.randint(
                n_key,
                size=(batch_size, head_size, sample_size),
                device=query_sorted.device,
            )
            print(f"sampled_set[0,0,-20:]={sampled_set[0,0,-20:]}")
            # Compute mask for hiding A_ij computed in block-diagonal attention
            offset_n = rearrange(
                torch.arange(n_query, device=query_sorted.device), "n -> 1 n 1"
            )
            weights = n_key / sample_size
            value_subset = indexing(value_sorted, sampled_set)
            key_subset = indexing(key_sorted, sampled_set)
            print(f"key_subset[0,0,0, -20:]={key_subset[0,0,0,-20:]}")

            if not self.cuda:
                block_mask = (offset_n // query_block_size) == (
                    sampled_set // key_block_size
                ).view(-1, 1, sample_size)
                block_mask = block_mask.view(batch_size, head_size, -1, sample_size)
                block_mask = block_mask.to(query_sorted.dtype)
                block_mask *= torch.finfo(
                    query_sorted.dtype
                ).min  # adding -inf added to QK^T

                attn_res, lse_res = exact_attention_xformers(
                    query_sorted,
                    key_subset,
                    value_subset,
                    scale,
                    causal=False,
                    bias=block_mask,
                )
            else:
                attn_res, lse_res = exact_attention_cuda(
                    query_sorted, key_subset, value_subset, scale, causal=False
                )
            lse_res = lse_res + math.log(weights)

            # Add two attentions
            if key_block_size > 0:
                attn, lse = add_self_attentions(
                    attn_block, lse_block, attn_res, lse_res
                )
            else:
                attn, lse = attn_res, lse_res
        else:
            attn, lse = attn_block, lse_block

        # Re-order rows with the inverse order for query_sorted -> query
        attn = indexing(attn, query_sort_idx_inv)
        lse = indexing(lse, query_sort_idx_inv)
        return attn, lse


def print_std_mean(data, log_prefix=""):
    data_std_mean = torch.std_mean(data.reshape(-1), dim=-1)
    print(
        f"{log_prefix} mean: "
        f"{data_std_mean[1].item()*100: 4.02f}%, "
        f"{log_prefix} std: {data_std_mean[0].item()*100: 4.02f}% | "
    )
    return data_std_mean


def compute_error_ratio(
    attn_approx, lse_approx, exact_attn, exact_lse, ord="fro", log_prefix=""
):
    diff_attn = attn_approx - exact_attn
    diff_lse = lse_approx - exact_lse

    diff_attn_norm = torch.norm(diff_attn, p=ord, dim=-1)
    exact_attn_norm = torch.norm(exact_attn, p=ord, dim=-1)
    spectral_error_ratio = diff_attn_norm / exact_attn_norm

    lse_error_ratio = torch.abs(diff_lse) / torch.abs(exact_lse)

    attn_std_mean = print_std_mean(
        spectral_error_ratio, f"{log_prefix} relative Attn error"
    )
    lse_std_mean = print_std_mean(lse_error_ratio, f"{log_prefix} relative Lse  error")

    return attn_std_mean, lse_std_mean


def load_qkv_from_file(path_prefix):
    query = (
        torch.permute(torch.load(path_prefix + "q.pt"), (0, 2, 1, 3))
        .to(torch.bfloat16)
        .to(torch.device("cuda"))
    )
    key = (
        torch.permute(torch.load(path_prefix + "k.pt"), (0, 2, 1, 3))
        .to(torch.bfloat16)
        .to(torch.device("cuda"))
    )
    value = (
        torch.permute(torch.load(path_prefix + "v.pt"), (0, 2, 1, 3))
        .to(torch.bfloat16)
        .to(torch.device("cuda"))
    )
    return query, key, value


def test_hyper_attn(query, key, value):
    batch_size, head_size, n_query, dim = query.shape
    attn_module = HyperAttention(
        input_dim=dim, lsh_num_projs=7, block_size=256, min_seq_len=256
    )

    scale = dim ** (-0.5)
    attn_approx, lse_approx = attn_module(
        query, key, value, scale, causal=False, return_lse=True
    )

    torch.set_printoptions(sci_mode=False)

    attn_calculator_func = exact_attention_xformers

    print(f"attn_approx[0,0] = \n{attn_approx[0,0]}")
    print(f"lse_approx[0,0] = \n{lse_approx[0,0]}")
    attn_exact, lse_exact = attn_calculator_func(query, key, value, scale, False, None)
    print(f"attn_exact[0,0] = \n{attn_exact[0,0]}")
    print(f"lse_exact[0,0] = \n{lse_exact[0,0]}")

    torch.set_printoptions(profile="default")

    relative_attn_err_std_mean, relative_lse_err_std_mean = compute_error_ratio(
        attn_approx, lse_approx, attn_exact, lse_exact, ord="fro"
    )


QKV_LIST = [
    "tests/data/set_1/layer04_self_attn_it538_20240521162520148861_",
    "tests/data/set_2/layer37_self_attn_it307_20240521162722916847_",
    "tests/data/set_3/layer04_self_attn_it999_20240521162129306018_",
]

if __name__ == "__main__":
    # Set the seed
    torch.manual_seed(9)

    qkv_id = 0
    data = load_qkv_from_file(QKV_LIST[qkv_id])
    test_hyper_attn(*data)
    print("All tests passed!")
