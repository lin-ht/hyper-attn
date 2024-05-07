import math
import torch
from typing import Optional

from attention.hyper_attn.utils import (
    exact_attention,
    exact_attention_cuda,
    exact_attention_xformers,
    add_self_attentions,
    indexing,
)
from attention.hyper_attn.angular_lsh import AngularLSH
from attention.hyper_attn.anns_hnsw import AnnsHNSW

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
        self.apply_2d_local_sampling = False
        self.pairing_method = pairing_method  # 'lsh' or 'anns'
        self.impl = impl
        self.approximate_unsampled = approximate_unsampled

        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim + 1))  # dim: (heads, seq_len, query/key_dim)

        if impl == 'xformers':
            self.exact_attn = exact_attention_xformers
        elif impl == 'cuda':
            self.exact_attn = exact_attention_cuda
        else:
            self.exact_attn = exact_attention

    def treat_sequence_as_2d(self, aspect_ratio: float):
        self.apply_2d_local_sampling = True
        self.aspect_ratio = aspect_ratio


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

    def attention_by_spatial_pairing(self, query, key, value, scale, aspect_ratio, lsh_query_sort_idx_inv, lsh_key_sort_idx_inv, lsh_query_block_size, lsh_key_block_size):
        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        h_query = int(math.sqrt(n_query / aspect_ratio))
        w_query = int(n_query / h_query)

        h_key = int(math.sqrt(n_key / aspect_ratio))
        w_key = int(n_key / h_key)

        sample_size = self.block_size
        h_sample_size_key = int(math.sqrt(sample_size/aspect_ratio))
        w_sample_size_key = int(sample_size / h_sample_size_key)

        n_h_blocks = h_key // h_sample_size_key
        n_w_blocks = w_key // w_sample_size_key
        n_blocks = n_h_blocks * n_w_blocks

        # TODO: handle the case with padding.
        assert n_h_blocks * h_sample_size_key == h_key, f"Invalid key height: {n_h_blocks} * {h_sample_size_key} != {h_key}"
        assert n_w_blocks * w_sample_size_key == w_key, f"Invalid key width: {n_w_blocks} * {w_sample_size_key} != {w_key}"

        h_sample_size_query = h_query // n_h_blocks
        w_sample_size_query = w_query // n_w_blocks

        assert n_h_blocks * h_sample_size_query == h_query, f"Invalid query height: {n_h_blocks} * {h_sample_size_query} != {h_query}"
        assert n_w_blocks * w_sample_size_query == w_query, f"Invalid query width: {n_w_blocks} * {w_sample_size_query} != {w_query}"

        query_sort_idx = torch.arange(n_query, device=query.device).reshape(n_h_blocks, h_sample_size_query, n_w_blocks, w_sample_size_query)
        query_sort_idx = torch.permute(query_sort_idx, (0, 2, 1, 3)).reshape(1, 1, n_query)
        query_sort_idx_full = query_sort_idx.expand(batch_size, head_size, -1)
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True) # for recovering the row order

        key_sort_idx = torch.arange(n_key, device=key.device).reshape(n_h_blocks, h_sample_size_key, n_w_blocks, w_sample_size_key)
        key_sort_idx = torch.permute(key_sort_idx, (0, 2, 1, 3)).reshape(1, 1, n_key)
        key_sort_idx_full = key_sort_idx.expand(batch_size, head_size, -1)
        key_sort_idx_inv = torch.argsort(key_sort_idx, dim=2, stable=True) # for recovering the row order

        key_block_size = h_sample_size_key * w_sample_size_key
        key_sorted = indexing(key, key_sort_idx_full, key_block_size)
        value_sorted = indexing(value, key_sort_idx_full, key_block_size)

        query_block_size = h_sample_size_query * w_sample_size_query
        query_sorted = indexing(query, query_sort_idx_full, query_block_size)

        # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
        query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
        key_split_per_block = key_sorted.view(-1, 1, key_block_size, dim)
        value_split_per_block = value_sorted.view(-1, 1, key_block_size, dim)

        # let data_1[i] = data[sort_1[i]], data_2[i] = data[sort_2[i]]
        # and inv_1[sort_1[i]] = i = inv_2[sort_2[i]], where inv[i] means
        # i-th data item is put in inv[i]-th position of the sorted data,
        # i.e. data[i] = data_1[inv_1[i]] = data_2[inv_2[i]]
        # now we want to find the index j in data_1 that corresponds to the value of data_2[k]
        # because data_2[k] = data[sort_2[k]] = data_1[inv_1[sort_2[k]]]
        # so j = inv_1[sort_2[k]]
        block_mask = lsh_query_sort_idx_inv.gather(-1, query_sort_idx_full).view(-1, 1, query_block_size, 1) // lsh_query_block_size == lsh_key_sort_idx_inv.gather(-1, key_sort_idx_full).view(-1, 1, 1, key_block_size) // lsh_key_block_size

        block_mask, sampled_cnt = self.finalize_block_mask(batch_size*head_size*n_blocks, 1, key_block_size, query.dtype, block_mask)

        # This attn_block = (D^{-1}A)(V) and the D^{-1}A does softmax locally according to the block.
        attn_block, lse_block = self.exact_attn(
            query_split_per_block, key_split_per_block, value_split_per_block,
            softmax_scale=scale, causal=False, bias=block_mask)

        if attn_block.shape[2] not in attn_block.stride():
            attn_block = attn_block.contiguous()
        attn_block = attn_block.view(batch_size, head_size, query_sorted.shape[2], -1)
        sampled_cnt = sampled_cnt.view(batch_size, head_size, query_sorted.shape[2], -1)

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

        # Return reverse sort idxes for block mask generation
        return attn_block, lse_block, sampled_cnt, query_sorted, key_sorted, value_sorted, query_block_size, key_block_size, query_sort_idx, key_sort_idx, query_sort_idx_inv, key_sort_idx_inv


    def attention_by_lsh_sort(self, query, key, value, scale):
        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        # Sorted block-diagonal via sortLSH
        if self.lsh.feature_dim == key.shape[-1] + 1:
            key_, key_norm_max = self.lsh.transform_key(key)
            query_ = self.lsh.transform_query(query, key_norm_max)
        else:
            key_, query_ = key, query

        _, query_sort_idx = torch.sort(self.lsh.hash(query_), dim=2, stable=True) # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash(key_), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True) # for recovering the row order
        key_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True) # for recovering the key order

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
                # query_sorted = query_sorted[:,:,:n_query,:]
            # key and query could be padded differently.
            # if key_sorted.shape[2] != n_key:
            #     key_sorted = key_sorted[:,:,:n_key,:]
            #     value_sorted = value_sorted[:,:,:n_key,:]

            # Restore the original order
            attn_block = indexing(attn_block, query_sort_idx_inv)
            lse_block = indexing(lse_block, query_sort_idx_inv)
        else:
            # Fall back to flash_attn2
            query_block_size = -1
            # query_sorted = indexing(query, query_sort_idx)
            attn_block, lse_block = 0, 0

        # Return reverse sort idxes for block mask generation
        return attn_block, lse_block, query_block_size, key_block_size, query_sort_idx_inv, key_sort_idx_inv

    def finalize_block_mask(self, batch_size, head_size, sample_size, dtype, block_mask:Optional[torch.tensor]=None):
        if block_mask is not None:
            block_mask = block_mask.view(batch_size, head_size, -1, sample_size)
            new_sample_cnt = sample_size - block_mask.sum(dim=-1, keepdim=True)
            block_mask = block_mask.to(dtype) * torch.finfo(dtype).min # adding -inf to QK^T to mask out
        else:
            new_sample_cnt = torch.ones(1) * sample_size
        return block_mask, new_sample_cnt

    def forward_no_causal_mask(self, query, key, value, scale):
        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        if self.min_seq_len >= n_query:
            return self.exact_attn(query, key, value, scale, causal=False)

        # 1. Significant correlation guided blockwise pairing through LSH
        attn_paired, lse_paired, lsh_query_block_size, lsh_key_block_size, lsh_query_sort_idx_inv, lsh_key_sort_idx_inv = self.attention_by_lsh_sort(query, key, value, scale)
        query_, key_, value_ = query, key, value

        # 2. Spatial local sampling
        if self.apply_2d_local_sampling:
            rst_local = self.attention_by_spatial_pairing(query, key, value, scale, self.aspect_ratio, lsh_query_sort_idx_inv, lsh_key_sort_idx_inv, lsh_query_block_size, lsh_key_block_size)
            attn_local, lse_local, sampled_cnt_local, query_, key_, value_, local_query_block_size, local_key_block_size, local_query_sort_idx, local_key_sort_idx, local_query_sort_idx_inv, local_key_sort_idx_inv = rst_local

            # local_query_block_size = 256
            # local_key_block_size = 256
            # query_, key_, value_ = query, key, value
            # local_key_sort_idx = torch.arange(n_key, device=key.device).reshape(1, 1, -1)
            # local_query_sort_idx = torch.arange(n_query, device=key.device).reshape(1, 1, -1)
            # local_query_sort_idx_inv = local_query_sort_idx
            # local_key_sort_idx_inv = local_key_sort_idx

            # Follow the order of spatial pairing
            local_query_sort_idx_full = local_query_sort_idx.expand(batch_size, head_size, -1)
            attn_paired = indexing(attn_paired, local_query_sort_idx_full)
            lse_paired = indexing(lse_paired, local_query_sort_idx_full)

            lsh_query_in_order = lsh_query_sort_idx_inv.gather(-1, local_query_sort_idx_full).view(batch_size, head_size, -1)

            attn_paired, lse_paired = add_self_attentions(attn_paired, lse_paired, attn_local, lse_local)
        else:
            local_query_block_size = -1
            local_key_block_size = -1
            # local_query_block_size = 256
            # local_key_block_size = 256
            # query_, key_, value_ = query, key, value
            sampled_cnt_local = torch.zeros(1, device=query.device)
            local_key_sort_idx = torch.arange(n_key, device=key.device).reshape(1, 1, -1)
            local_query_sort_idx = torch.arange(n_query, device=key.device).reshape(1, 1, -1)
            local_key_sort_idx_inv = local_key_sort_idx
            local_query_sort_idx_inv = local_query_sort_idx

            lsh_query_in_order = torch.arange(n_query, device=query_.device).reshape(1, -1, 1)

        # 3. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if sample_size > 0 and (n_query > lsh_query_block_size) and (n_key > lsh_key_block_size):
            # Hack to have same probability for each key column
            sample_prob = torch.ones(1, device=query_.device).as_strided_((batch_size * head_size, n_key), (0, 0))
            sampled_set = torch.multinomial(sample_prob, sample_size, replacement=False).reshape(batch_size, head_size, sample_size)
            # sampled_set = (local_key_sort_idx_inv[0,0,:sample_size]).reshape(1,1,-1)
            # sampled_set[0,0,sample_size-1] = local_key_sort_idx_inv[0,0,sample_size]
            sampled_set = sampled_set.expand(batch_size, head_size, sample_size)
            value_subset = indexing(value_, sampled_set)
            key_subset = indexing(key_, sampled_set)

            # Compute mask for hiding A_ij computed in block-diagonal attention
            if self.impl != "cuda":
                block_mask = None
                # Exclude samples already covered by diagonal blocks from local sampling
                offset_n = torch.arange(n_query, device=query_.device).reshape(1, -1, 1)
                if local_key_block_size > 0 and local_query_block_size > 0:
                    # Final block_mask is a 4d-tensor with shape [batch_size * head_size, n_query, sample_size]
                    block_mask = (offset_n // local_query_block_size) == (sampled_set // local_key_block_size).view(-1, 1, sample_size)
                    block_mask = block_mask.view(-1, n_query, sample_size)
                # Exclude samples already covered by diagonal blocks from lsh sorting
                if lsh_key_block_size > 0 and lsh_query_block_size > 0:
                    tmp_2 = local_key_sort_idx.expand(batch_size, head_size, -1).view(batch_size, head_size, -1, 1)
                    tmp_1 = indexing(tmp_2, sampled_set).view(batch_size, head_size, -1)  # index in the original data
                    sampled_set_lsh = indexing(lsh_key_sort_idx_inv.view(batch_size, head_size, -1, 1), tmp_1).view(batch_size, head_size, -1)
                    block_mask_lsh = (lsh_query_in_order.view(-1, n_query, 1) // lsh_query_block_size) == (sampled_set_lsh // lsh_key_block_size).view(-1, 1, sample_size)
                    block_mask_lsh = block_mask_lsh.view(-1, n_query, sample_size)
                    block_mask = block_mask | block_mask_lsh if block_mask is not None else block_mask_lsh

                block_mask, sampled_cnt_rand = self.finalize_block_mask(batch_size, head_size, sample_size, query_.dtype, block_mask)
                attn_res, lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False, bias=block_mask)
            else:
                sampled_cnt_rand = torch.ones(1) * sample_size
                attn_res, lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False)

            # Add only sampled residual attentions:
            if local_key_block_size > 0 or lsh_key_block_size > 0:
                attn_, lse_ = add_self_attentions(attn_paired, lse_paired, attn_res, lse_res)
            else:
                attn_, lse_ = attn_res, lse_res

        # 4. Significant sampling according to Value
            value_norms = torch.norm(value_, dim=-1)
            _, topk_sampled_set = torch.topk(value_norms, self.sample_size, dim=-1, largest=True, sorted=False)
            topk_sampled_set.reshape(batch_size, head_size, sample_size)
            value_subset = indexing(value_, topk_sampled_set)
            key_subset = indexing(key_, topk_sampled_set)

            # Compute mask for hiding A_ij computed in block-diagonal attention and previously sampled attentions
            if self.impl != "cuda":
                block_mask = None
                # Exclude samples already covered by diagonal blocks from local sampling
                offset_n = torch.arange(n_query, device=query_.device).reshape(1, -1, 1)
                if local_key_block_size > 0 and local_query_block_size > 0:
                    # Final block_mask is a 4d-tensor with shape [batch_size * head_size, n_query, sample_size]
                    block_mask = (offset_n // local_query_block_size) == (topk_sampled_set // local_key_block_size).view(-1, 1, sample_size)
                    block_mask = block_mask.view(-1, n_query, sample_size)

                # Exclude samples already covered by diagonal blocks from lsh sorting
                if lsh_key_block_size > 0 and lsh_query_block_size > 0:
                    tmp_2 = local_key_sort_idx.expand(batch_size, head_size, -1).view(batch_size, head_size, -1, 1)
                    tmp_1 = indexing(tmp_2, topk_sampled_set).view(batch_size, head_size, -1)
                    sampled_set_lsh = indexing(lsh_key_sort_idx_inv.view(batch_size, head_size, -1, 1), tmp_1).view(batch_size, head_size, -1)
                    block_mask_lsh = (lsh_query_in_order.view(-1, n_query, 1) // lsh_query_block_size) == (sampled_set_lsh // lsh_key_block_size).view(-1, 1, sample_size)
                    block_mask_lsh = block_mask_lsh.view(-1, n_query, sample_size)
                    block_mask = block_mask | block_mask_lsh if block_mask is not None else block_mask_lsh

                # Exclude samples already covered by uniform sampling
                double_sampled = torch.zeros_like(topk_sampled_set, dtype=torch.bool)
                for i in range(double_sampled.shape[0]):
                    for j in range(double_sampled.shape[1]):
                        sampled_set_j = sampled_set[i, j, :]
                        double_sampled[i, j, :] = torch.isin(topk_sampled_set[i, j, :], sampled_set_j, assume_unique=True).view(1, 1, sample_size)
                block_mask = block_mask | double_sampled.view(-1, 1, sample_size) if block_mask is not None else double_sampled

                block_mask, topk_sampled_cnt = self.finalize_block_mask(batch_size, head_size, sample_size, query_.dtype, block_mask)
                topk_attn_res, topk_lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False, bias=block_mask)
            else:
                topk_sampled_cnt = torch.ones(1) * sample_size
                topk_attn_res, topk_lse_res = self.exact_attn(query_, key_subset, value_subset, scale, causal=False)

            # Add only topk sampled residual attentions:
            attn_, lse_ = add_self_attentions(attn_, lse_, topk_attn_res, topk_lse_res)

        # 5. Unseen part approximation
            weights = torch.clamp((n_key - sampled_cnt_local - sampled_cnt_rand - topk_sampled_cnt - lsh_key_block_size) / (sampled_cnt_rand + 1e-6), min=0.0) # weights >= 0.0
            lse_res_unseen = lse_res + torch.log(weights)
            # Treat the unseen part as zero attentions if unseen_estimation_type is 0,
            # i.e. assuming the mean of the rest of residual attentions is zero.
            attn_res_unseen = attn_res if self.approximate_unsampled else 0
            # Add the approximated unseen residual attentions from uniformly sampled ones:
            attn, lse = add_self_attentions(attn_, lse_, attn_res_unseen, lse_res_unseen)
        else:
            # Only one block, no approximation
            attn, lse = attn_paired, lse_paired

        # if local_query_sort_idx_inv is not None:
        if self.apply_2d_local_sampling:
            # Re-order rows with the inverse order for query_sorted -> query
            attn = indexing(attn, local_query_sort_idx_inv.expand(batch_size, head_size, -1))
            lse = indexing(lse, local_query_sort_idx_inv.expand(batch_size, head_size, -1))
        return attn, lse
