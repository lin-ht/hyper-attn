import torch
import pytest

import triton
import triton.language as tl
# from .utils import get_shape_from_layout, get_strides_from_layout, is_cdna, is_rdna, DEBUG, AUTOTUNE

try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError as e:
    flash_attn_func_cuda = None
    print(f"flash_attn importError: {e}")

DEBUG = True
AUTOTUNE = False

def get_shape_from_layout(q, k, layout, cu_seqlens_q = None, cu_seqlens_k = None, max_seqlen_q=None, max_seqlen_k=None):
    if layout == 'bhsd':
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
    elif layout == 'bshd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
    elif  layout == 'thd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        batch_k, max_seqlen_k, nheads_k, head_size_k = len(cu_seqlens_k) - 1, max_seqlen_k, k.shape[1], k.shape[2]
    else:
        assert False, "Got unsupported layout."
    
    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, max_seqlen_q, max_seqlen_k


def get_strides_from_layout(q, k, v, o, layout):
    if layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, start_m,
                    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed, batch_philox_offset, exp_scores_ptrs,
                    block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, score_ptrs, scores_scaled_shifted_ptrs,
                    IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    OFFS_M: tl.constexpr, OFFS_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, PADDED_HEAD: tl.constexpr,
                    ACTUAL_BLOCK_DMODEL: tl.constexpr, SM_SCALE: tl.constexpr, USE_EXP2: tl.constexpr,
                    RETURN_SCORES: tl.constexpr):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
    
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        
        # -- compute qk ----
        qk += tl.dot(q, k)
        qk_scaled =  qk * SM_SCALE
        if RETURN_SCORES:
            score_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(score_ptrs, qk_scaled, mask=score_mask)

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            qk_scaled += bias

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            qk_scaled += alibi_block
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[:, None]
        if RETURN_SCORES: 
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            scores_scaled_shifted_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(scores_scaled_shifted_ptrs, q_shifted, mask=scores_scaled_shifted_mask)
        
        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_SCORES:
                # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
                exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
                tl.store(exp_scores_ptrs, tl.where(keep, p, -p), mask=exp_score_mask)
            p = tl.where(keep, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(exp_scores_ptrs, p, mask=exp_score_mask)
        
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(v.type.element_ty), v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += BLOCK_N
            scores_scaled_shifted_ptrs += BLOCK_N
            exp_scores_ptrs += BLOCK_N
    return acc, l_i, m_i


def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']


def get_rdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
    ]


def get_autotune_configs():
    return get_cuda_autotune_config(), ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']

    if AUTOTUNE:
        # if is_rdna():
        #     return get_rdna_autotune_configs()
        # elif is_cdna():
        #     return get_cdna_autotune_configs()
        # else:
        #     raise ValueError("Unknown Device Type")
        pass
    else:
        return [
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "waves_per_eu": 1, "PRE_LOAD_V": False},
                num_stages=1,
                num_warps=4,
            ),
        ], [
            "IS_CAUSAL",
            "dropout_p",
            "MAX_SEQLENS_Q",
            "MAX_SEQLENS_K",
            "ACTUAL_BLOCK_DMODEL",
            "VARLEN",
            "HQ",
            "HK",
        ]




autotune_configs, autotune_keys = get_autotune_configs()

# @triton.autotune(
#     configs=autotune_configs,
#     key=autotune_keys,
#     use_cuda_graph=True,
# )
@triton.jit
def attn_fwd(Q, K, V, bias, SM_SCALE: tl.constexpr, LSE, Out, stride_qz, stride_qh, stride_qm, stride_qk, 
             stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, 
             stride_oz, stride_oh, stride_om, stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az, stride_ah,
             stride_sz, stride_sh, stride_sm, stride_sn, stride_lse_z, stride_lse_h, stride_lse_m, cu_seqlens_q, cu_seqlens_k,
             dropout_p, philox_seed, philox_offset_base, scores, scores_scaled_shifted, exp_scores, alibi_slopes,  HQ: tl.constexpr,
             HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, RETURN_SCORES: tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        # print("cu_seqlens_q_start:", cu_seqlens_q_start)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if (IS_CAUSAL):
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            # We still need to write 0s to the result
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
            # statically known.
            l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
            l_ptrs = l_offset + offs_m * stride_lse_m 
            
            l = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
           
            # mask_m_offsets = start_m + tl.arange(0, BLOCK_M)
            # lse_mask = mask_m_offsets < causal_start_idx
            # softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)
            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    n_extra_tokens = 0
    # print("n_extra_tokens:", n_extra_tokens)
    # print("seqlen_k:", seqlen_k)
    # print("BLOCK_N:", BLOCK_N)
    # return
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    if USE_BIAS:
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_h_q * stride_bh
        bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if RETURN_SCORES:
        scores_offset = scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        score_ptrs = scores_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn

        scores_scaled_shifted_offset = scores_scaled_shifted + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        scores_scaled_shifted_ptrs = scores_scaled_shifted_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    
        exp_scores_offset = exp_scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        exp_scores_ptrs = exp_scores_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        score_ptrs = None
        scores_scaled_shifted_ptrs = None
        exp_scores_ptrs = None

    if ENABLE_DROPOUT:
        off_hz = off_z * HQ + off_h_q
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn,
                                        start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, batch_philox_offset,
                                        exp_scores_ptrs,
                                        # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                                        block_min, block_max, 0, 0, 0, alibi_slope, score_ptrs, scores_scaled_shifted_ptrs,
                                        # IS_CAUSAL, ....
                                        False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, False, ENABLE_DROPOUT, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL, SM_SCALE,  USE_EXP2=USE_EXP2, RETURN_SCORES=RETURN_SCORES)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if USE_BIAS:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += n_full_blocks * BLOCK_N
            scores_scaled_shifted_ptrs += n_full_blocks * BLOCK_N
            exp_scores_ptrs += n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn,
                                        start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, batch_philox_offset,
                                        exp_scores_ptrs, block_min, block_max, offs_n_causal, masked_blocks,
                                        n_extra_tokens, alibi_slope, score_ptrs, scores_scaled_shifted_ptrs,
                                        IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, True, ENABLE_DROPOUT, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL, SM_SCALE, USE_EXP2=USE_EXP2, RETURN_SCORES=RETURN_SCORES)
    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
    
    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
    l_ptrs = l_offset + offs_m * stride_lse_m 
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i)
    
    if IS_CAUSAL:
        # zero out nans caused by -infs when doing causal
        lse_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask) # the log of the normalization constant
    else:
        tl.store(l_ptrs, softmax_lse) # the log of the normalization constant

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


def attention_prefill_forward_triton_impl(
                                        q,
                                        k,
                                        v,
                                        o,
                                        sm_scale,
                                        alibi_slopes,
                                        causal,
                                        bias,
                                        dropout_p,
                                        layout,
                                        cu_seqlens_q, 
                                        cu_seqlens_k,
                                        max_seqlens_q, 
                                        max_seqlens_k, 
                                        return_scores, 
                                        use_exp2):

    if DEBUG:
        print()
        print("attention_prefill_forward_triton_impl")
        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)
        print("o:", o.shape)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)

    # check if varlen
    is_varlen = layout == "thd"

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if (bias is not None):
        assert (bias.numel() < 2**31)

    batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k = get_shape_from_layout(q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)

    grid = lambda META: (triton.cdiv(max_seqlens_q, META['BLOCK_M']), nheads_q, batch)

    if return_scores:
        scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        scores_scaled_shifted = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        scores_strides = (scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3))
    else:
        scores = None
        scores_scaled_shifted = None
        scores_strides = (0, 0 , 0 , 0)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if return_scores:
        exp_scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
    else:
        exp_scores = None

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        softmax_lse = torch.empty((q.shape[0], nheads_q), device=q.device, dtype=torch.float32)
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # Seed the RNG so we get reproducible results for testing.
    philox_seed = 0x1BF52
    philox_offset = 0x1D4B42

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1),bias.stride(2),
                        bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)


    attn_fwd[grid](q, k, v, bias, sm_scale, softmax_lse, o, *q_strides, *k_strides, *v_strides, *o_strides,
                    *bias_strides, *alibi_strides, *scores_strides, stride_lse_z, stride_lse_h, stride_lse_m, cu_seqlens_q, cu_seqlens_k,
                    dropout_p=dropout_p, philox_seed=philox_seed, philox_offset_base=philox_offset, scores=scores, 
                    scores_scaled_shifted=scores_scaled_shifted, exp_scores=exp_scores, alibi_slopes=alibi_slopes, 
                    HQ=nheads_q, HK=nheads_k, ACTUAL_BLOCK_DMODEL=head_size, MAX_SEQLENS_Q=max_seqlens_q,
                    MAX_SEQLENS_K=max_seqlens_k, IS_CAUSAL=causal, VARLEN=is_varlen,
                    BLOCK_DMODEL=padded_d_model, USE_BIAS=False if bias is None else True,
                    USE_ALIBI=False if alibi_slopes is None else True, ENABLE_DROPOUT=dropout_p>0.0, USE_EXP2=use_exp2, RETURN_SCORES=return_scores,
                    BLOCK_M=128, BLOCK_N=128, PRE_LOAD_V=False)

    return o, softmax_lse, exp_scores, grid, head_size, philox_seed, philox_offset, scores, scores_scaled_shifted


# defailt fp16 tolerance is ATOL, RTOL = 1e-5, 1e-3. See table https://pytorch.org/docs/stable/testing.html
ATOL, RTOL = 1e-2, 1e-2 # old standard. maybe to lose. 
# ATOL, RTOL = 1e-3, 1e-3  # catchs fa mismatch issues
# ATOL, RTOL = 1e-4, 1e-3 # to strict. there will be small diffs
# ATOL, RTOL = 1e-5, 1e-3 # # default fp16. there will be small diffs
EQUAL_NAN = True


class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    layout = None
    cache_seqlens = None
    cache_batch_idx = None
    new_kv = False
    seqlen_new = None
    k_new = None
    v_new = None
    dropout_p, return_scores= 0.0, False
    # NOTE: scale sm_scale by log_2(e) and use 2^x in the loop as we do not have native e^x support in HW.
    use_exp2 = False
    

    def __repr__(self) -> str:
        return (f"MetaData(\n"
                f"  sm_scale={self.sm_scale},\n"
                f"  cu_seqlens_q={self.cu_seqlens_q},\n"
                f"  cu_seqlens_k={self.cu_seqlens_k},\n"
                f"  max_seqlens_q={self.max_seqlens_q},\n"
                f"  max_seqlens_k={self.max_seqlens_k},\n"
                f"  bias={self.bias},\n"
                f"  alibi_slopes={self.alibi_slopes},\n"
                f"  causal={self.causal},\n"
                f"  num_contexts={self.num_contexts},\n"
                f"  varlen={self.varlen},\n"
                f"  layout={self.layout},\n"
                f"  cache_seqlens={self.cache_seqlens},\n"
                f"  cache_batch_idx={self.cache_batch_idx},\n"
                f"  new_kv={self.new_kv},\n"
                f"  seqlen_new={self.seqlen_new},\n"
                f"  k_new={self.k_new},\n"
                f"  v_new={self.v_new},\n"
                f"  dropout_p={self.dropout_p},\n"
                f"  return_scores={self.return_scores}\n"
                f")")

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p, return_scores):
        self.dropout_p = dropout_p
        self.return_scores = return_scores

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size, _, _ = get_shape_from_layout(q, k, self.layout, self.cu_seqlens_q, self.cu_seqlens_k, self.max_seqlens_q, self.max_seqlens_k)
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            # assert not self.return_scores
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device="cuda", DEBUG_INPUT=False):
    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, f'Got unsupported tensor layout: {layout}'

    if DEBUG_INPUT:
        if layout == "bhsd":
            q = torch.arange(N_CTX_Q, dtype=dtype, device=device).view(1, 1, N_CTX_Q, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
        elif layout == "bshd":
            q = torch.arange(N_CTX_Q, dtype=dtype, device=device).view(1, N_CTX_Q, 1, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
    else:
        q = torch.randn(q_tensor_shape, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(k_tensor_shape, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(k_tensor_shape, dtype=dtype, device=device, requires_grad=True)
    
    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def get_input_metadata(q, k):
    # assume layout bshd as the same as flash_attn2
    layout = "bshd"
    (Z, N_CTX_Q, HQ, D_HEAD) = q.shape
    (Z, N_CTX_K, HK, D_HEAD) = k.shape

    sm_scale = D_HEAD**-0.5

    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return input_metadata


def attention_prefill(q, k, v, input_metadata):
    o = torch.empty_like(q)

    (output, 
    softmax_lse, 
    exp_scores, 
    grid, 
    head_size, 
    philox_seed, 
    philox_offset, 
    _, 
    _) = attention_prefill_forward_triton_impl(
        q, k, v, o, 
        input_metadata.sm_scale, 
        input_metadata.alibi_slopes,
        input_metadata.causal, 
        input_metadata.bias, 
        input_metadata.dropout_p, 
        input_metadata.layout, 
        input_metadata.cu_seqlens_q, 
        input_metadata.cu_seqlens_k, 
        input_metadata.max_seqlens_q, 
        input_metadata.max_seqlens_k, 
        input_metadata.return_scores, 
        input_metadata.use_exp2,
    )
    return output, softmax_lse, exp_scores


def flash_attn_func(q, k, v, causual: bool = False):
    input_metadata = get_input_metadata(q, k)
    if causual:
        input_metadata.need_causal()
    return attention_prefill(q, k, v, input_metadata)[:2]

def ref_flash_attn2(q, k, v, causal, layout, input_metadata):
    # assert layout == 'bshd'
    if layout == "bhsd":
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    
    ref = flash_attn_func_cuda(q, k, v, causal)

    if layout == "bhsd":
        ref = ref.transpose(1, 2).clone()
    return ref


def ref_torch_attention(q, k, v, causal, layout, input_metadata):
    # Transpose here if layout is bshd so we have same reference code for all layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()

    _, HQ, N_CTX_Q, _ = q.shape
    _, HK, N_CTX_K, _ = k.shape

    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    # if use_alibi:
    #     scores += compute_alibi_tensor_ref(alibi_slopes, N_CTX_Q, N_CTX_K)

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    return ref_out, scores

def check_allclose(rst, ref, label=""):
    is_allclose = torch.allclose(rst, ref)
    max_err = (rst - ref).abs().max()
    mean_err = (rst - ref).abs().mean()
    print(f"[{label}]Maximum error: {max_err}, Mean error: {mean_err}, Allclose: {is_allclose}")
    return is_allclose


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (1, 32, 32, 65536, 65536, 64),
    # (4, 32, 32, 1024, 1024, 64),
    # (4, 48, 24, 1024, 1024, 64),
    # (1, 24, 6, 8192, 8192, 64),
    # (1, 4, 2, 16384, 16384, 128),
    # (2, 16, 4, 1020, 987, 128),
    # (2, 16, 4, 15498, 2, 128),
    # (2, 16, 2, 7, 16219, 64),
    # (4, 48, 12, 1, 1, 64),
    # (4, 48, 48, 1, 1, 128),
    # (4, 48, 24, 3, 3, 128),
    # (4, 48, 48, 1001, 990, 64),
    # (1, 8, 8, 8081, 7099, 64),
    # (1, 4, 4, 16330, 15989, 128),
    # (4, 4, 1, 1024, 1024, 33),
    # (4, 4, 2, 65, 1018, 65),
    # (4, 4, 4, 128, 128, 65),
    # (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('use_alibi', [False])
@pytest.mark.parametrize('layout', ['bhsd']) # @pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
def test_op_fwd_prefill(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_alibi, layout, dtype=torch.bfloat16):
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / HQ * i) for i in range(1, HQ + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    # triton implementation
    ref_out = ref_flash_attn2(q.clone(), k.clone(), v.clone(), causal, layout, input_metadata)
    tri_out_0, _ = flash_attn_func(q.clone(), k.clone(), v.clone(), causal)
    # tri_out, _, _ = attention_prefill(q, k, v, input_metadata)
    # # ref_out, _ = ref_torch_attention(q.clone(), k.clone(), v.clone(), causal, layout, input_metadata)
    
    check_allclose(tri_out_0, ref_out, "0 vs ref")
    # check_allclose(tri_out, ref_out, "1 vs ref")
    # check_allclose(tri_out_0, tri_out, "0 vs 1")

    # torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ref_out, tri_out_0, atol=2e-2, rtol=2e-2)