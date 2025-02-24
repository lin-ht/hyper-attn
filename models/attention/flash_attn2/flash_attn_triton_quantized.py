"""
*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import math

import torch
import triton
import triton.language as tl


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: True,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,  # b, h, s
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    deb,
    K_scales,  # b
    K_zero_points,  # b
    V_scales,  # b
    V_zero_points,  # b
    softmax_scale,
    stride_qb, # batch
    stride_qh, # nheads
    stride_qm, # seqlen
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_debb,
    stride_debh,
    stride_debn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    K_BITS: tl.constexpr,
    K_CNTS: tl.constexpr,
    V_BITS: tl.constexpr,
    V_CNTS: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = 0  # seqlen // BLOCK_M
    off_hb = tl.program_id(1)  # nheads * batch
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d_k = tl.arange(0, BLOCK_HEADDIM//K_CNTS)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM//V_CNTS)
    offs_d_k_cnts = tl.arange(0, K_CNTS)
    offs_d_v_cnts = tl.arange(0, V_CNTS)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    ) # [1, BLOCK_HEADDIM]
    # # Encoded:
    # k_ptrs = (
    #     K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d_k[None, :])
    # ) # [BLOCK_N, BLOCK_HEADDIM]
    # No encoding:
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    ) # [BLOCK_N, BLOCK_HEADDIM]

    # # No Encoding:
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    # # Encoded:
    # v_ptrs = (
    #     V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d_v[None, :])
    # )
    deb_ptrs = (
        deb + off_b * stride_debb + off_h * stride_debh + (offs_n[:, None] * stride_debn + offs_d[None, :])
    )

    # # Encoded:
    # k_scales_ptr = K_scales + off_b
    # k_zero_points_ptr = K_zero_points + off_b
    # k_scales_data = tl.load(k_scales_ptr)
    # k_zero_points_data = tl.load(k_zero_points_ptr)

    # # Encoded:
    # v_scales_ptr = V_scales + off_b
    # v_zero_points_ptr = V_zero_points + off_b
    # v_scales_data = tl.load(v_scales_ptr)
    # v_zero_points_data = tl.load(v_zero_points_ptr)

    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    
    if EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)

    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                # # No encoding:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0)
                # # Encoded:
                # k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d_k[None, :] * K_CNTS < headdim, other=0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0,
                )
            else:
                # # No Encoding:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0,
                )
                # # Encoded:
                # k = tl.load(
                #     k_ptrs + start_n * stride_kn,
                #     mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d_k[None, :] * K_CNTS < headdim),
                #     other=0,
                # )
        # # Encoded:
        # dequantize k
        # tl.debug_barrier()
        # K_BITS_MASK: tl.constexpr = ((1 << K_BITS) - 1)
        # K_BITS_BASE: tl.constexpr = (K_CNTS - 1) * K_BITS
        # k_ind = (k[:, :, None] >> (K_BITS_BASE - offs_d_k_cnts[None, None, :]* K_BITS)) & K_BITS_MASK
        # k_ind = tl.reshape(k_ind, [BLOCK_N, BLOCK_HEADDIM])
        # k_decoded = (k_ind - k_zero_points_data) / k_scales_data
        # deb_data = k_decoded
        # if EVEN_N:  # If we just do "if EVEN_N", there seems to be some race condition
        #     if EVEN_HEADDIM:
        #         tl.store(deb_ptrs + start_n * stride_debn, deb_data)
        #     else:
        #         tl.store(deb_ptrs + start_n * stride_debn, deb_data, mask=offs_d[None, :] < headdim)
        # else:
        #     if EVEN_HEADDIM:
        #         tl.store(
        #             deb_ptrs + start_n * stride_debn,
        #             deb_data,
        #             mask=(start_n + offs_n)[:, None] < seqlen_k,
        #         )
        #     else:
        #         tl.store(
        #             deb_ptrs + start_n * stride_debn,
        #             deb_data,
        #             mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :]< headdim),
        #         )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # qk += tl.dot(q, tl.trans(k))
        # # No encoding:
        qk += tl.reshape(tl.sum(q * k, axis=-1), [BLOCK_M, BLOCK_N])
        # # Encoded:
        # qk += tl.reshape(tl.sum(q * k_decoded, axis=-1), [BLOCK_M, BLOCK_N])
        # qk += tl.reshape(tl.sum(q * k_ind, axis=-1), [BLOCK_M, BLOCK_N])
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i) # [BLOCK_M]
        p = tl.exp(qk * softmax_scale - m_ij[:, None]) # [BLOCK_M, BLOCK_N]
        l_ij = tl.sum(p, 1) # [BLOCK_M]

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                # # No encoding:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
                # # Encoded:
                # v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d_v[None, :] * V_CNTS < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                # # No Encoding:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                # # Encoded:
                # v = tl.load(
                #     v_ptrs + start_n * stride_vn,
                #     mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d_v[None, :] * V_CNTS < headdim),
                #     other=0.0,
                # )
        # # dequantize v
        # tl.debug_barrier()
        # V_BITS_MASK: tl.constexpr = ((1 << V_BITS) - 1)
        # V_BITS_BASE: tl.constexpr = (V_CNTS - 1) * V_BITS
        # v_ind = (v[:, :, None] >> (V_BITS_BASE - offs_d_v_cnts[None, None, :]* V_BITS)) & V_BITS_MASK
        # v_ind = tl.reshape(v_ind, [BLOCK_N, BLOCK_HEADDIM])
        # v_decoded = (v_ind - v_zero_points_data) / v_scales_data
        # if not EVEN_N:  # Need to mask out otherwise the acc_o is wrong
        #     v_decoded *= tl.where((start_n + offs_n)[:, None] < seqlen_k, 1, 0) # [BLOCK_N, BLOCK_HEADDIM]

        # p = p.to(v.dtype)
        # # No encoding:
        acc_o += tl.sum(tl.trans(p) * v, 0) # acc_o += tl.dot(p, v) # v: [BLOCK_N, BLOCK_HEADDIM]
        # # Encoded:
        # acc_o += tl.sum(tl.trans(p) * v_decoded, 0) # acc_o += tl.dot(p, v) # v: [BLOCK_N, BLOCK_HEADDIM]

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

        # if EVEN_N:  # If we just do "if EVEN_N", there seems to be some race condition
        #     if EVEN_HEADDIM:
        #         tl.store(deb_ptrs + start_n * stride_debn, v_decoded)
        #     else:
        #         tl.store(deb_ptrs + start_n * stride_debn, v_decoded, mask=offs_d[None, :] < headdim)
        # else:
        #     if EVEN_HEADDIM:
        #         tl.store(
        #             deb_ptrs + start_n * stride_debn,
        #             v_decoded,
        #             mask=(start_n + offs_n)[:, None] < seqlen_k,
        #         )
        #     else:
        #         tl.store(
        #             deb_ptrs + start_n * stride_debn,
        #             v_decoded,
        #             mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :]< headdim),
        #         )
    # end of for start_n in range(0, end_n, BLOCK_N):

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = 0
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    
    if EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o.to(tl.float16))
    else:
        tl.store(out_ptrs, acc_o.to(tl.float16), mask=offs_d[None, :] < headdim)

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dx(
    dx_ptrs,
    dx,
    offs_n,
    offs_d,
    seqlen,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    even_headdim,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if even_headdim:
            tl.store(dx_ptrs, dx)
        else:
            tl.store(dx_ptrs, dx, mask=offs_d[None, :] < headdim)
    else:
        if even_headdim:
            tl.store(dx_ptrs, dx, mask=offs_n[:, None] < seqlen)
        else:
            tl.store(dx_ptrs, dx, mask=(offs_n[:, None] < seqlen) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dx(
            dk_ptrs,
            dk,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            even_headdim=EVEN_HEADDIM,
        )
        _bwd_store_dx(
            dv_ptrs,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            even_headdim=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dx(
        dk_ptrs,
        dk,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        even_headdim=EVEN_HEADDIM,
    )
    _bwd_store_dx(
        dv_ptrs,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        even_headdim=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_attn_forward(q, k, v, k_bits, k_scales, k_zero_points, v_bits, v_scales, v_zero_points, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    k_cnts = 8 // k_bits
    v_cnts = 8 // v_bits
    assert seqlen_q == 1, "Only support special case: seqlen_q=1"
    assert bias is None, "Bias not supported"
    
    _, seqlen_k, _, d_k = k.shape
    _, seqlen_v, _, d_v = v.shape
    # # No encoding:
    assert d == d_k, f"d(={d}) must be d_k(={d_k})"
    assert d == d_v, f"d(={d}) must be d_v(={d_v})"
    # # Encoded:
    # assert d == d_k * k_cnts, f"d(={d}) must be d_k(={d_k}) * k_cnts(={k_cnts})"
    # assert d == d_v * v_cnts, f"d(={d}) must be d_v(={d_v}) * v_cnts(={v_cnts})"
    assert k.shape == (batch, seqlen_k, nheads, d_k)
    assert v.shape == (batch, seqlen_k, nheads, d_v)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    # assert q.dtype == k.dtype, "Tensor q and k must have the same type"
    # # Encoded:
    # assert k.dtype in [torch.uint8], "k encoded type must be uint8"
    # assert v.dtype in [torch.uint8], "v encoded type must be uint8"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = seqlen_q # math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded * 128), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded * 128), device=q.device, dtype=torch.float32)
    
    o = torch.empty_like(q)
    deb = torch.empty((batch, math.ceil(seqlen_k/128) * 128, nheads, d), device=k.device, dtype=q.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    assert BLOCK_HEADDIM % k_cnts == 0, f"BLOCK_HEADDIM(={BLOCK_HEADDIM}) must be divisible by k_cnts(={k_cnts})"
    BLOCK_M = 1
    BLOCK_N = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (1, batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        deb,
        k_scales, # 1D
        k_zero_points, # 1D
        v_scales, # 1D
        v_zero_points, # 1D
        softmax_scale,
        q.stride(0), # stride_b
        q.stride(2), # stride_h
        q.stride(1), # stride_s
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        deb.stride(0),
        deb.stride(2),
        deb.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        k_bits,
        k_cnts,
        v_bits,
        v_cnts,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale, deb  # softmax_scale could have been updated


def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)


# class FlashAttnQKVPackedFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, qkv, bias=None, causal=False, softmax_scale=None):
#         """
#         qkv: (batch, seqlen, 3, nheads, headdim)
#         bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
#             For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
#             ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
#         """
#         # Make sure that the last dimension is contiguous
#         if qkv.stride(-1) != 1:
#             qkv = qkv.contiguous()
#         o, lse, ctx.softmax_scale = _flash_attn_forward(
#             qkv[:, :, 0],
#             qkv[:, :, 1],
#             qkv[:, :, 2],
#             bias=bias,
#             causal=causal,
#             softmax_scale=softmax_scale,
#         )
#         ctx.save_for_backward(qkv, o, lse, bias)
#         ctx.causal = causal
#         return o

#     @staticmethod
#     def backward(ctx, do):
#         qkv, o, lse, bias = ctx.saved_tensors
#         assert not ctx.needs_input_grad[1], "FlashAttention does not support bias gradient yet"
#         # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
#         # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
#         with torch.inference_mode():
#             dqkv = torch.empty_like(qkv)
#             _flash_attn_backward(
#                 do,
#                 qkv[:, :, 0],
#                 qkv[:, :, 1],
#                 qkv[:, :, 2],
#                 o,
#                 lse,
#                 dqkv[:, :, 0],
#                 dqkv[:, :, 1],
#                 dqkv[:, :, 2],
#                 bias=bias,
#                 causal=ctx.causal,
#                 softmax_scale=ctx.softmax_scale,
#             )
#         return dqkv, None, None, None


# flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply


# class FlashAttnKVPackedFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
#         """
#         q: (batch, seqlen_q, nheads, headdim)
#         kv: (batch, seqlen_k, 2, nheads, headdim)
#         bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
#             For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
#             ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
#         """
#         # Make sure that the last dimension is contiguous
#         q, kv = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
#         o, lse, ctx.softmax_scale = _flash_attn_forward(
#             q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale
#         )
#         ctx.save_for_backward(q, kv, o, lse, bias)
#         ctx.causal = causal
#         return o

#     @staticmethod
#     def backward(ctx, do):
#         q, kv, o, lse, bias = ctx.saved_tensors
#         if len(ctx.needs_input_grad) >= 3:
#             assert not ctx.needs_input_grad[2], "FlashAttention does not support bias gradient yet"
#         # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
#         # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
#         with torch.inference_mode():
#             dq = torch.empty_like(q)
#             dkv = torch.empty_like(kv)
#             _flash_attn_backward(
#                 do,
#                 q,
#                 kv[:, :, 0],
#                 kv[:, :, 1],
#                 o,
#                 lse,
#                 dq,
#                 dkv[:, :, 0],
#                 dkv[:, :, 1],
#                 bias=bias,
#                 causal=ctx.causal,
#                 softmax_scale=ctx.softmax_scale,
#             )
#         return dq, dkv, None, None, None


# flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_bits, k_scales, k_zero_points, v_bits, v_scales, v_zero_points, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v, k_scales, k_zero_points, v_scales, v_zero_points = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v, k_scales, k_zero_points, v_scales, v_zero_points]]
        o, lse, ctx.softmax_scale, deb = _flash_attn_forward(
            q, k, v, k_bits, k_scales, k_zero_points, v_bits, v_scales, v_zero_points, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o, lse, deb

    @staticmethod
    def backward(ctx, do, dlse_use_needed=None):
        q, k, v, o, lse, bias = ctx.saved_tensors
        # assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        # with torch.inference_mode():
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _flash_attn_backward(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk,
            dv,
            bias=bias,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
        )
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply

# flash_attn_func_apply = FlashAttnFunc.apply
# def flash_attn_func(q, k, v, bias=None, causal=False, softmax_scale=None):
#     k_bits = 2
#     k_lut = torch.zeros(q.shape[0], 2**k_bits, device=q.device, dtype=q.dtype)
#     return flash_attn_func_apply(q, k, v, k_bits, k_lut, bias, causal, softmax_scale)

from attention.flash_attn2.quantization import quantize, dequantize, prepare_data, test_quantization2

def test_flash_attn_quantized(bits = 2, batch_size = 2, seq_len = 1024, hn = 32, dh = 256):
    # test_quantization2(bits, batch_size, seq_len, hn, dh)
    dtype = torch.float16
    device = "cuda"

    RTOL = 1e-4 if dtype == torch.float else 1e-2
    ATOL = 1e-4 if dtype == torch.float else 1e-2

    k_org, k_deq, k_ind, k_lut, k_min, k_max, _, _ = prepare_data(bits, batch_size, seq_len, hn, dh, dtype=dtype, device=device)

    compression_scale = 8 // bits
    # k_ind_v = k_ind.view(batch_size, -1)
    # assert k_ind_v.shape[-1] % compression_scale == 0, f"Unsupported shape {k_ind_v.shape}"
    # k_ind_v = k_ind.view(batch_size, -1, compression_scale)
    # k_encoded = quantize(k_ind_v, bits) # k_encoded dtype = uint8

    # k_encoded_v = k_encoded.view(batch_size, -1)
    # print(f"{k_encoded_v.shape=}, {k_encoded_v[0:2, 0:4]=}")
    
    # k_decoded = dequantize(k_encoded, k_lut, bits).reshape(k_deq.shape)
    # torch.testing.assert_close(k_deq.to(device=k_decoded.device), k_decoded, rtol=RTOL, atol=ATOL)

    
    q = torch.randn((batch_size, 1, hn, dh), dtype=dtype, device=device, requires_grad=False)
    v = torch.randn((batch_size, seq_len, hn, dh), dtype=dtype, device=device, requires_grad=False)
    print(f"{q.shape=}, {k_org.shape=}, {v.shape=}")

    flash_attn_func(q, k_org, v, bits, k_lut, None, False, None)

    print("Passed!")


if __name__ == "__main__":
    bits = 2
    batch_size = 128
    seq_len = 3072 
    hn = 48 # 32
    dh = 128

    test_flash_attn_quantized(bits, batch_size, seq_len, hn, dh)