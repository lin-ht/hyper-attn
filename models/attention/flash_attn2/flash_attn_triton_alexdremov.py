import logging
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError as e:
    flash_attn_func_cuda = None
    print(f"flash_attn importError: {e}")

MAX_TILE_SIZE = 256
MIN_TILE_SIZE = 32


logger = logging.getLogger(__name__)


# BLOCK_Q, BLOCK_K, num_warps, num_stages
_h100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (32, 64, 4, 3),
    (torch.float32, 256): (32, 32, 4, 3),
    (torch.bfloat16, 64): (128, 128, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (64, 32, 4, 3),
    (torch.float16, 64): (128, 128, 4, 3),
    (torch.float16, 128): (128, 128, 8, 3),
    (torch.float16, 256): (64, 32, 4, 3),
}

_a100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (128, 32, 4, 3),
    (torch.float32, 256): (64, 16, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (32, 64, 4, 3),
    (torch.float16, 64): (128, 64, 4, 3),
    (torch.float16, 128): (128, 64, 8, 3),
    (torch.float16, 256): (32, 64, 4, 3),
}


def _get_default_config_fwd(head_dim, dtype) -> tuple[int, int, int, int]:
    default_config = None

    if head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _h100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (8, 0):  # A100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _a100_default_config.get((dtype, head_dim), default_config)
    else:  # modest hardware or extremely large head_dim
        if dtype == torch.float32:
            default_config = (32, 16, 4, 3)
        else:
            default_config = (64, 32, 4, 3)

    return default_config


def strides(t):
    assert t is not None
    return [t.stride(i) for i in range(t.ndim)]


def fwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    min_size, max_size = 16, 256
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 32
        max_pipeline = 3
        max_warps = 4
    elif HEAD_DIM == 256:
        max_size = 128
        min_size = 32
        max_pipeline = 2
        max_warps = 4

    configs = [i for i in configs if min_size <= i.kwargs["TILE_K_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_Q_SIZE"] <= max_size]
    configs = [
        i for i in configs if min_pipeline <= i.kwargs["PIPELINING"] <= max_pipeline
    ]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]

    default_config = _get_default_config_fwd(HEAD_DIM, DTYPE)
    if default_config is not None:
        configs += [
            triton.Config(
                dict(
                    PIPELINING=default_config[3],
                    TILE_Q_SIZE=default_config[0],
                    TILE_K_SIZE=default_config[1],
                    V_PRELOAD=V_PRELOAD,
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
            for V_PRELOAD in (True, False)
        ]

    logger.warning(f"Start benchmarking forward streaming_attention {len(configs) = }")
    return configs


# fmt: off
@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
        V_PRELOAD=lambda _: True,
    )
)
@triton.jit
def _self_attn_fwd(
    Q: tl.tensor, Kt: tl.tensor, V: tl.tensor, 
    K_scales: tl.tensor,  # B
    K_zero_points: tl.tensor,  # B
    V_scales: tl.tensor,  # B
    V_zero_points: tl.tensor,  # B
    L: tl.tensor, # L is a list of batch sequence lengths
    O: tl.tensor,  #
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kk: int, stride_kt: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int, #
    lens_stride: int, # lens L is a 1D vector or None
    Tq: int,  #
    Tk: int,  #
    K_BITS: tl.constexpr,  # int
    K_CNTS: tl.constexpr,  # int
    V_BITS: tl.constexpr,  # int
    V_CNTS: tl.constexpr,  # int
    PRESCALE: tl.constexpr,  #
    TIME_BUCKET:  int,  #
    LEN_PRESENT: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    DTYPE:  tl.constexpr,  #
    TILE_Q_SIZE: tl.constexpr,  #
    TILE_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    V_PRELOAD: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    if LEN_PRESENT:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len, Tq)
        need_q_mask = q_token_idx + TILE_Q_SIZE >= seq_len
    else:
        seq_len = Tq
        need_q_mask = False

    if seq_len <= q_token_idx:
        return

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(Tq, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    HEAD_DIM_K: tl.constexpr = HEAD_DIM // K_CNTS # assert HEAD_DIM % V_CNTS == 0
    kt_tile_ptr = tl.make_block_ptr(
        base=Kt + kbatch_head_offset,
        shape=(HEAD_DIM_K, Tk),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_K, TILE_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    HEAD_DIM_V: tl.constexpr = HEAD_DIM // V_CNTS # assert HEAD_DIM % V_CNTS == 0
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(Tk, HEAD_DIM_V),
        strides=(stride_vt, stride_vk),
        offsets=(0, 0),
        block_shape=(TILE_K_SIZE, HEAD_DIM_V),
        order=(1, 0),
    )

    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)

    q_tile = tl.load(
        q_tile_ptr,
        boundary_check=(0,),
    )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    if PRESCALE:
        q_tile *= softmax_scale

    max_tile = tl.cdiv(Tk, TILE_K_SIZE)
    for kv_tile_idx in tl.range(
        0, max_tile, num_stages=PIPELINING
    ):
        last_iter = kv_tile_idx == max_tile - 1
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if last_iter:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
            )
        else:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )
        # # dequantize k
        # tl.debug_barrier()
        K_BITS_MASK: tl.constexpr = ((1 << K_BITS) - 1)
        K_BITS_BASE: tl.constexpr = (K_CNTS - 1) * K_BITS
        k_cnts_arange = tl.arange(0, K_CNTS)
        kt_ind = (kt_tile[:, None, :] >> (K_BITS_BASE - k_cnts_arange[None, :, None]* K_BITS)) & K_BITS_MASK
        kt_ind = tl.reshape(kt_ind, [HEAD_DIM, TILE_K_SIZE])

        k_scales_ptr = K_scales + batch
        k_zero_points_ptr = K_zero_points + batch
        k_scales_data = tl.load(k_scales_ptr)
        k_zero_points_data = tl.load(k_zero_points_ptr)
        kt_tile_decoded = (kt_ind - k_zero_points_data) / k_scales_data
        
        if V_PRELOAD:
            if last_iter:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                    boundary_check=(0,),
                )
            else:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                )

        if TILE_Q_SIZE == 1:
            qk = tl.reshape(tl.sum(q_tile * tl.trans(kt_tile_decoded), axis=-1), [TILE_Q_SIZE, TILE_K_SIZE])
        else:
            qk = tl.dot(
                q_tile, kt_tile_decoded, input_precision=INPUT_PRECISION, out_dtype=tl.float32
            )

        if not PRESCALE:
            qk *= softmax_scale

        if last_iter:
            kv_indices = kv_token_idx + tile_k_arange

            mask = (
                kv_indices[None, :] < Tk
            )

            qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        m_ij = tl.maximum(m_i, tl.max(qk, 1)) # (TILE_Q_SIZE,)
        p = tl.math.exp2(qk - m_ij[:, None]) # (TILE_Q_SIZE, TILE_K_SIZE)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] # (TILE_Q_SIZE, HEAD_DIM)

        if not V_PRELOAD:
            if last_iter:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                    boundary_check=(0,),
                )
            else:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                ) # (TILE_K_SIZE, HEAD_DIM_V)

        # # dequantize v
        # tl.debug_barrier()
        V_BITS_MASK: tl.constexpr = ((1 << V_BITS) - 1)
        V_BITS_BASE: tl.constexpr = (V_CNTS - 1) * V_BITS
        v_cnts_arange = tl.arange(0, V_CNTS)
        v_ind = (v_tile[:, :, None] >> (V_BITS_BASE - v_cnts_arange[None, None, :]* V_BITS)) & V_BITS_MASK
        v_ind = tl.reshape(v_ind, [TILE_K_SIZE, HEAD_DIM])

        v_scales_ptr = V_scales + batch
        v_zero_points_ptr = V_zero_points + batch
        v_scales_data = tl.load(v_scales_ptr)
        v_zero_points_data = tl.load(v_zero_points_ptr)
        v_tile_decoded = (v_ind - v_zero_points_data) / v_scales_data

        if TILE_Q_SIZE == 1:
            acc += tl.sum(p[:, :, None] * v_tile_decoded[None, :, :].to(p.dtype), axis=1) # (TILE_Q_SIZE, HEAD_DIM)
        else:
            acc = tl.dot(
                p,
                v_tile_decoded.to(p.dtype),
                acc,
                input_precision=INPUT_PRECISION,
                out_dtype=tl.float32,
            )
        m_i = m_ij

    acc = acc / l_i[:, None]
    if need_q_mask:
        q_lens_mask = (
            q_tile_indices[:, None] < seq_len
        )
        acc = tl.where(q_lens_mask, acc, 0.0)

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(Tq, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(
        o_tile_ptr,
        acc.to(o_tile_ptr.type.element_ty),
        boundary_check=(0,),
    )
# fmt: on


def autotune_prehook(kwargs, reset_only=False):
    if kwargs["L"] is not None:
        kwargs["L"].add_(kwargs["q"].size(2))  # L += time


def autotune_posthook(kwargs, exception=None):
    if kwargs["L"] is not None:
        kwargs["L"].add_(-kwargs["q"].size(2))  # L -= time


streaming_forward = triton.heuristics(
    dict(
        PIPELINING=lambda _: 1, # TODO: Tune PIPELINING? lambda args: min(args["Tk"], 4) if args["Tq"] == 1 else 1,
        # TILE_Q_SIZE=lambda args: min(args["Tq"], 64),
        TILE_K_SIZE=lambda _: 64,
    )
)(_self_attn_fwd)

streaming_forward_autotune = triton.autotune(
    configs=[
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_Q_SIZE=tile_q,
                TILE_K_SIZE=tile_k,
                V_PRELOAD=V_PRELOAD,
            ),
            num_warps=num_warps,
            num_stages=pipe,
        )
        for num_warps in [4, 8]
        for pipe in [1, 2]
        for tile_q in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for tile_k in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for V_PRELOAD in (True, False)
    ],
    key=["HEAD_DIM", "INPUT_PRECISION", "TIME_BUCKET", "DTYPE"],
    prune_configs_by=dict(early_config_prune=fwd_configs_pruner),
    pre_hook=autotune_prehook,
    post_hook=autotune_posthook,
)(_self_attn_fwd)


@torch.library.custom_op(
    "alexdremov_flash_attention::forward", mutates_args=(), device_types=("cuda",)
)
def attention_forward_adapter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scales: torch.Tensor,
    k_zero_points: torch.Tensor,
    v_scales: torch.Tensor,
    v_zero_points: torch.Tensor,
    lens: torch.Tensor,
    sm_scale: float,
    autotune: bool,
    prescale: bool,
) -> torch.Tensor:
    batch, heads, Tq, HEAD_DIM = q.shape
    Tk = k.shape[-2]
    HEAD_DIM_K = k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    assert HEAD_DIM % HEAD_DIM_K == 0
    assert HEAD_DIM % HEAD_DIM_V == 0
    K_CNTS = HEAD_DIM // HEAD_DIM_K
    V_CNTS = HEAD_DIM // HEAD_DIM_V
    assert 8 % K_CNTS == 0
    assert 8 % V_CNTS == 0
    K_BITS = 8 // K_CNTS
    V_BITS = 8 // V_CNTS
    # assert HEAD_DIM == k.shape[-1] and HEAD_DIM == v.shape[-1]
    assert Tk == v.shape[-2]
    # assert Tq == k.shape[-2] and Tq == v.shape[-2]
    assert sm_scale is not None
    assert lens is None or (
        lens.dtype == torch.int32 and batch == len(lens) and lens.ndim == 1
    )

    O = torch.zeros_like(q, memory_format=torch.contiguous_format)
    INPUT_PRECISION = (
        "tf32" if torch.get_float32_matmul_precision() != "highest" else "ieee"
    )

    grid = lambda args: (
        batch,
        heads,
        triton.cdiv(Tq, args["TILE_Q_SIZE"]),
    )
    tile_q = min(64, Tq)
    # print(f"tile_q: {tile_q}")

    kt = k.transpose(-1, -2)  # just stride tricks, same data
    fwd_fn = streaming_forward_autotune if autotune else streaming_forward
    fwd_fn[grid](
        q,
        kt,
        v,
        k_scales,
        k_zero_points,
        v_scales,
        v_zero_points,
        lens,
        O,
        *strides(q),
        *strides(kt),
        *strides(v),
        *strides(O),
        *(strides(lens) if lens is not None else [0]),
        Tq=Tq,
        Tk=Tk,
        K_BITS=K_BITS,
        K_CNTS=K_CNTS,
        V_BITS=V_BITS,
        V_CNTS=V_CNTS,
        PRESCALE=prescale,
        HEAD_DIM=HEAD_DIM,
        INPUT_PRECISION=INPUT_PRECISION,
        DTYPE=q.dtype,
        TIME_BUCKET=triton.next_power_of_2(Tq),
        TILE_Q_SIZE=tile_q,
        LEN_PRESENT=lens is not None,
        SM_SCALE=sm_scale,
    )
    return O


@torch.library.register_fake("alexdremov_flash_attention::forward")
def attention_forward_adapter_abstract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scales: torch.Tensor,
    k_zero_points: torch.Tensor,
    v_scales: torch.Tensor,
    v_zero_points: torch.Tensor,
    lens: torch.Tensor,
    sm_scale: float,
    autotune: bool,
    prescale: bool,
) -> torch.Tensor:
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def self_attention_fa2(q, k, v, layout: str = "bshd"):
    # flash_attn: (B, S, H, D)
    if layout == "bshd":
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = flash_attn_func_cuda(q, k, v, causal=False)
    else:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = flash_attn_func_cuda(q, k, v, causal=False)
        o = o.transpose(1, 2)
    return o

def self_attention_reference(q, k, v, lens, layout:str = "bshd"):
    if layout == "bshd":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    Tq = q.shape[-2]

    attn_mask = None
    if lens is not None:
        assert Tq == k.shape[-2], f"{(Tq, k.shape[-2])}"
        key_padding_mask = (
            torch.arange(Tq, device="cuda").unsqueeze(0) < lens.unsqueeze(-1)
        ).unsqueeze(-1)
        key_padding_mask_ref = key_padding_mask
        key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
        attn_mask = key_padding_mask.unsqueeze(1) # (1, 1, Tq, Tq)
        res_mask = key_padding_mask_ref.unsqueeze(1) # (1, 1, Tq, 1)
    else:
        res_mask = torch.tensor([True], device="cuda")

    o =F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask) * res_mask # (B, H, T:SEQLEN, HEAD_DIM)
    if layout == "bshd":
        o = o.transpose(1, 2)
        if res_mask.ndim == 4:
            res_mask = res_mask.transpose(1, 2)

    return (
        o, 
        res_mask,
    )


def self_attention_reference_naive(q, k, v, lens):
    Tq = q.shape[-2]
    D = q.shape[-1]

    attn_mask = None
    if lens is not None:
        key_padding_mask = (
            torch.arange(Tq, device="cuda").unsqueeze(0) < lens.unsqueeze(-1)
        ).unsqueeze(-1)
        key_padding_mask_ref = key_padding_mask
        key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
        attn_mask = key_padding_mask.unsqueeze(1)
        res_mask = key_padding_mask_ref.unsqueeze(1)
    else:
        res_mask = torch.tensor([True], device="cuda")

    qkt = (q / (D**0.5)) @ k.transpose(-1, -2)
    if attn_mask is not None:
        qkt = torch.where(attn_mask, qkt, -float("inf"))
    scores = F.softmax(qkt, dim=-1)
    result = scores @ v

    return (
        torch.where(res_mask, result, 0),
        res_mask,
    )


def self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scales: torch.Tensor,
    k_zero_points: torch.Tensor,
    v_scales: torch.Tensor,
    v_zero_points: torch.Tensor,
    lens: torch.Tensor | None,
    sm_scale: float | None = None,
    autotune=True,
    prescale=False,
):
    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM**-0.5
    return torch.ops.alexdremov_flash_attention.forward(
        q,
        k,
        v,
        k_scales,
        k_zero_points,
        v_scales,
        v_zero_points,
        lens,
        sm_scale,
        autotune,
        prescale,
    )

def self_attention_for_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scales: torch.Tensor,
    k_zero_points: torch.Tensor,
    v_scales: torch.Tensor,
    v_zero_points: torch.Tensor,
    lens: torch.Tensor | None = None,
    sm_scale: float | None = None,
    autotune=False,
    prescale=False,
    layout: str = "bshd",
):
    if layout == "bhsd":
        return self_attention(q, k, v, k_scales, k_zero_points, v_scales, v_zero_points, lens, sm_scale, autotune, prescale)
    
    # elif layout == "bshd":
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    o = self_attention(q, k, v, k_scales, k_zero_points, v_scales, v_zero_points, lens, sm_scale, autotune, prescale)
    return o.transpose(1, 2)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../../../")
    sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../../")
    sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")

    B, H, Tk, Tq, D = 7, 2, 32, 1, 128
    context, back = 10, 9

    from tests.test_self_attention import test_self_attention

    test_self_attention(
        B=B,
        H=H,
        Tq=Tq,
        Tk=Tk,
        HEAD_DIM=D,
        LAYOUT="bshd",
        dtype=torch.float32,
        lens="none",
        noncontiguous=False,
        autotune=False,
    )
    print("Passed!")