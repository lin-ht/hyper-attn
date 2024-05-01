import math
import torch
from einops import rearrange
from typing import Optional

from attention.hyper_attn.utils import (
    exact_attention,
    exact_attention_cuda,
    exact_attention_xformers,
    add_self_attentions,
    indexing,
)
from attention.hyper_attn.angular_lsh import AngularLSH


class HyperAttention(torch.nn.Module):

    def __init__(self, input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=4096, pairing_method='lsh', approximate_unsampled=True, impl='xformers'):
        """
        Hyper attention module.
        Input parameters:
            - input_dim: int, the dimension of input query and key
            - lsh_num_projs: int, the number of projections for LSH
            - block_size: int, the block size for block-diagonal approximation
            - sample_size: int, the number of sampled columns in the attention matrix A
            - min_seq_len: int, the minimum sequence length the hyper_attn is applied to
            - approximate_unsampled: bool, whether to approximate the unseen part of the attention matrix from uniformly sampled ones
            - impl: str, the implementation of the exact attention
        """
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs  # 2 ** 7 = 128 slots in the hash table
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.pairing_method = pairing_method  # 'lsh' or 'anns'
        self.impl = impl
        self.approximate_unsampled = approximate_unsampled
        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))  # dim: (heads, seq_len, query/key_dim)

        if impl == 'xformers':
            self.exact_attn = exact_attention_xformers
        elif impl == 'cuda':
            self.exact_attn = exact_attention_cuda
        else:
            self.exact_attn = exact_attention


    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, scale=None, causal=False, return_lse=False):
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
                attn, lse = self.exact_attn(query, key, value, scale, causal=True)
            else:
                # If n_query is odd we pad inputs by adding all-zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(query, (0,0,0,1), mode='constant', value=0.)
                    key = torch.nn.functional.pad(key, (0,0,0,1), mode='constant', value=0.)
                    value = torch.nn.functional.pad(value, (0,0,0,1), mode='constant', value=0.)

                q_bd = query.view(batch_size, 2*n_heads, query.shape[2]//2, query.shape[-1])
                k_bd = key.view(batch_size, 2*n_heads, key.shape[2]//2, key.shape[-1])
                v_bd = value.view(batch_size, 2*n_heads, key.shape[2]//2, value.shape[-1])

                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)

                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2]//2:, :],
                    key[:, :, :key.shape[2]//2, :],
                    value[:, :, :key.shape[2]//2, :], scale)

                attn_up, lse_up = attn_bd[:,:,:query.shape[2]//2,:], lse_bd[:,:,:query.shape[2]//2,:]
                attn_down, lse_down = add_self_attentions(
                    attn_bd[:,:,query.shape[2]//2:,:],
                    lse_bd[:,:,query.shape[2]//2:,:],
                    attn_unmasked,
                    lse_unmasked)

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                # If n_query was odd exclude the last rows
                if n_query % 2:
                    attn = attn[:,:,:-1,:]
                    lse = lse[:,:,:-1,:]

        if not return_lse:
            return attn
        else:
            return attn, lse

    def attention_by_lsh_sort(self, query, key, value, scale):
        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        # Sorted block-diagonal via sortLSH
        _, query_sort_idx = torch.sort(self.lsh.hash(query), dim=2, stable=True) # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash(key), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True) # for recovering the row order

        key_block_size = self.block_size
        key_sorted = indexing(key, key_sort_idx, key_block_size)
        value_sorted = indexing(value, key_sort_idx, key_block_size)

        if key_block_size > 0:

            num_blocks = key_sorted.shape[2] // key_block_size
            query_block_size = math.ceil(n_query / num_blocks)
            query_sorted = indexing(query, query_sort_idx, query_block_size)

            # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
            query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
            key_split_per_block = key_sorted.view(-1, 1, key_block_size, dim)
            value_split_per_block = value_sorted.view(-1, 1, key_block_size, dim)

            # This attn_block = (D^{-1}A)(V) and the D^{-1}A does softmax locally according to the block.
            attn_block, lse_block = self.exact_attn(
                query_split_per_block, key_split_per_block, value_split_per_block,
                softmax_scale=scale, causal=False)

            if attn_block.shape[2] not in attn_block.stride():
                attn_block = attn_block.contiguous()
            attn_block = attn_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            if lse_block.shape[2] not in lse_block.stride():
                lse_block = lse_block.contiguous()
            lse_block = lse_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            # When inputs are padded, then unpad them
            if query_sorted.shape[2] != n_query: #query.shape[2]:
                attn_block, lse_block = attn_block[:,:,:n_query,:], lse_block[:,:,:n_query,:]
                query_sorted = query_sorted[:,:,:n_query,:]
            # key and query could be padded differently.
            if key_sorted.shape[2] != n_key:
                key_sorted = key_sorted[:,:,:n_key,:]
                value_sorted = value_sorted[:,:,:n_key,:]

        else:
            # Fall back to flash_attn2
            query_block_size = -1
            query_sorted = indexing(query, query_sort_idx)
            attn_block, lse_block = 0, 0

        return attn_block, lse_block, query_sorted, key_sorted, value_sorted, query_block_size, key_block_size, query_sort_idx_inv

    # def sampled_set_of_diagonal_blocks(self, n_query, query_block_size, n_key, key_block_size, device=None) -> torch.tensor|None:
    #     if key_block_size <=0 or query_block_size <= 0:
    #         return None
    #     n_key_sup = (n_key+key_block_size-1)/key_block_size * key_block_size
    #     key_range = torch.clamp(torch.arange(n_key_sup, device=device), max=n_key-1).reshape([-1, key_block_size])
    #     sampled_set = key_range.repeat([1, query_block_size]).reshape([-1, key_block_size])
    #     return sampled_set[:n_query, :]

    def get_block_mask(self, query: torch.tensor, new_samples:torch.tensor, sampled_set:Optional[torch.tensor]=None, query_block_size=1, key_block_size=1) -> tuple[Optional[torch.tensor], torch.tensor]:
        """
        The shape of new_samples (sampled_set) could be [batch_size * head_size, n_query | 1, sample_size]
        """
        batch_size, head_size, n_query, dim = query.shape
        sample_size = new_samples.shape[-1]

        block_mask = None
        # Exclude samples already covered by diagonal blocks
        if self.pairing_method == 'lsh':
            offset_n = torch.arange(n_query, device=query.device).reshape(1, -1, 1)
            if key_block_size > 0 and query_block_size > 0:
                # Final block_mask is a 4d-tensor with shape [batch_size * head_size, n_query, sample_size]
                block_mask = (offset_n // query_block_size) == (new_samples // key_block_size).view(-1, 1, sample_size)
                block_mask = block_mask.view(batch_size * head_size, -1, sample_size)

        # Exclude samples covered by existing samples
        # Caution: ensure the samples of each row of sampled_set are unique.
        if sampled_set is not None:
            assert sampled_set.shape[0] == batch_size * head_size
            double_sampled = torch.zeros_like(new_samples, dtype=torch.bool)
            for i in range(double_sampled.shape[0]):
                matched = sampled_set.shape[1] == new_samples.shape[1]
                for j in range(double_sampled.shape[1]):
                    sampled_set_j = sampled_set[i, j, :] if matched else sampled_set[i, 0, :]
                    double_sampled[i, j, :] = torch.isin(new_samples[i, j, :], sampled_set_j, assume_unique=True).view(1, 1, sample_size)

            if block_mask is None:
                block_mask = double_sampled
            else:
                block_mask = block_mask | double_sampled

        if block_mask is not None:
            block_mask.view(batch_size, head_size, -1, sample_size)
            new_sample_cnt = sample_size - block_mask.sum(dim=-1, keepdim=True)
            block_mask *= torch.finfo(query.dtype).min # adding -inf to QK^T to mask out
        else:
            new_sample_cnt = torch.ones(1) * sample_size
        return block_mask, new_sample_cnt

    def forward_no_causal_mask(self, query, key, value, scale):
        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        if self.min_seq_len >= n_query:
            return self.exact_attn(query, key, value, scale, causal=False)

        # 1. Significant correlation guided sampling
        if self.pairing_method == 'lsh':
            attn_paired, lse_paired, query_, key_, value_, query_block_size, key_block_size, query_sort_idx_inv = self.attention_by_lsh_sort(query, key, value, scale)
        else:
            raise ValueError(f"Unknown pairing method: {self.pairing_method}")

        # 2. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if sample_size > 0 and (n_query > query_block_size) and (n_key > key_block_size):
            # Hack to have same probability for each key column
            sample_prob = torch.ones(1, device=query_.device).as_strided_((batch_size * head_size, n_key), (0, 0))
            sampled_set = torch.multinomial(sample_prob, sample_size, replacement=False).reshape(batch_size, head_size, sample_size)
            value_subset = indexing(value_, sampled_set)
            key_subset = indexing(key_, sampled_set)

            # Compute mask for hiding A_ij computed in block-diagonal attention
            # offset_n = rearrange(torch.arange(n_query, device=query_sorted.device), 'n -> 1 n 1')
            offset_n = torch.arange(n_query, device=query_.device).reshape(1, -1, 1)
            if self.impl != "cuda":
                if key_block_size > 0:
                    # block_mask is a 4d-tensor with shape [batch_size, head_size, n_query, sample_size]
                    block_mask = (offset_n // query_block_size) == (sampled_set // key_block_size).view(-1, 1, sample_size)
                    block_mask = block_mask.view(batch_size, head_size, -1, sample_size)
                    block_mask = block_mask.to(query_.dtype)
                    sampled_cnt =  sample_size - block_mask.sum(dim=-1, keepdim=True)
                    block_mask *= torch.finfo(query_.dtype).min # adding -inf to QK^T to mask out
                else:
                    sampled_cnt = torch.ones(1) * sample_size
                    block_mask = None

                attn_res, lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False, bias=block_mask)
            else:
                sampled_cnt = torch.ones(1) * sample_size
                attn_res, lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False)

            # Add only sampled residual attentions:
            if key_block_size > 0:
                attn_, lse_ = add_self_attentions(attn_paired, lse_paired, attn_res, lse_res)
            else:
                attn_, lse_ = attn_res, lse_res

        # 3. Significant sampling according to Value
            value_norms = torch.norm(value_, dim=-1)
            _, topk_sampled_set = torch.topk(value_norms, self.sample_size, dim=-1, largest=True, sorted=False)
            topk_sampled_set.reshape(batch_size, head_size, sample_size)
            value_subset = indexing(value_, topk_sampled_set)
            key_subset = indexing(key_, topk_sampled_set)

            # Compute mask for hiding A_ij computed in block-diagonal attention and previously sampled attentions
            if self.impl != "cuda":
                if key_block_size > 0:
                    # block_mask is a 4d-tensor with shape [batch_size, head_size, n_query, sample_size]
                    sampled_set = sampled_set.view(-1, 1, sample_size)
                    topk_sampled_set = topk_sampled_set.view(-1, 1, sample_size)
                    topk_block_mask = (offset_n // query_block_size) == (topk_sampled_set // key_block_size)
                    double_sampled = torch.zeros_like(topk_sampled_set, dtype=torch.bool)
                    for i in range(topk_block_mask.shape[0]):
                        double_sampled[i, 0, :] = torch.isin(topk_sampled_set[i, 0, :], sampled_set[i, 0, :], assume_unique=True).view(1, 1, sample_size)
                    topk_block_mask = topk_block_mask | double_sampled
                    topk_block_mask = topk_block_mask.view(batch_size, head_size, -1, sample_size)
                    topk_block_mask = topk_block_mask.to(query_.dtype)
                    topk_sampled_cnt = sample_size - topk_block_mask.sum(dim=-1, keepdim=True)
                    topk_block_mask *= torch.finfo(query_.dtype).min # adding -inf to QK^T to mask out
                else:
                    topk_sampled_cnt = torch.ones(1) * sample_size
                    topk_block_mask = None

                topk_attn_res, topk_lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False, bias=topk_block_mask)
            else:
                topk_sampled_cnt = torch.ones(1) * sample_size
                topk_attn_res, topk_lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False)

            # Add only topk sampled residual attentions:
            attn_, lse_ = add_self_attentions(attn_, lse_, topk_attn_res, topk_lse_res)

            # Unseen part approximation
            weights = torch.clamp((n_key - sampled_cnt - topk_sampled_cnt - key_block_size) / (sampled_cnt + 1e-6), min=0.0) # weights >= 0.0
            lse_res_unseen = lse_res + torch.log(weights)
            # Treat the unseen part as zero attentions if unseen_estimation_type is 0,
            # i.e. assuming the mean of the rest of residual attentions is zero.
            attn_res_unseen = attn_res if self.approximate_unsampled else 0
            # Add the approximated unseen residual attentions from uniformly sampled ones:
            attn, lse = add_self_attentions(attn_, lse_, attn_res_unseen, lse_res_unseen)
        else:
            # Only one block, no approximation
            attn, lse = attn_paired, lse_paired

        # Re-order rows with the inverse order for query_sorted -> query
        attn = indexing(attn, query_sort_idx_inv)
        lse = indexing(lse, query_sort_idx_inv)
        return attn, lse
