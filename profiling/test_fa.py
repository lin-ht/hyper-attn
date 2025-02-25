import os
import argparse
import torch
import triton
import triton.language as tl

from attention.flash_attn2.flash_attn_triton_amd import flash_attn_func as flash_attn_func_amd
from attention.flash_attn2.flash_attn_triton_alexdremov import self_attention_for_layout as flash_attn_func_alex
from attention.flash_attn2.flash_attn_triton_quantized import flash_attn_func
from attention.hyper_attn.hyper_attn import HyperAttention
from attention.flash_attn2.flash_attn_xformers import flash_attn_func as flash_attn_func_xformers

try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError as e:
    flash_attn_func_cuda = None
    print(f"flash_attn importError: {e}")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_causal", action="store_true")
    parser.add_argument("--mode", type=str, default="fwd", choices=['fwd', 'bwd', 'fwd+bwd'])
    parser.add_argument("--attn_method", type=str, default="flash", choices=['flash', 'hyper'])
    parser.add_argument("--impl", type=str, default="alex", choices=['cuda', 'triton', 'amd', 'alex', 'xformers'])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--head_size", type=int, default=48)
    parser.add_argument("--dim", type=int, default=128)
    return parser.parse_args()

@triton.heuristics(
    {
        "EVEN_N": lambda args: args["seqlen"] % args["BLOCK_SIZE"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def decode_kernel(
    x,  # b, s, hn, dh
    o,
    scales,  # b
    zero_points,  # b
    stride_b,
    stride_h,
    stride_s,
    stride_ob,
    stride_oh,
    stride_os,
    seqlen,
    nheads,
    headdim,  # output head dim
    BITS: tl.constexpr,
    CNTS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr, # Always EVEN HEAD
    BLOCK_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    start_m = 0  # seqlen // BLOCK_M
    off_hb = tl.program_id(1)  # nheads * batch
    off_b = off_hb // nheads  # b
    off_h = off_hb % nheads  # h
    # initialize offsets
    offs_s = tl.arange(0, BLOCK_SIZE)
    offs_d_o = tl.arange(0, BLOCK_HEADDIM)
    
    offs_d_i = tl.arange(0, BLOCK_HEADDIM//CNTS)
    offs_d_i_cnts = tl.arange(0, CNTS)

    # Initialize pointers to X, O
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    x_ptrs = (
        x + off_b * stride_b + off_h * stride_h + (offs_s[:, None] * stride_s + offs_d_i[None, :])
    ) # [BLOCK_N, BLOCK_HEADDIM]
    o_ptrs = (
        o + off_b * stride_ob + off_h * stride_oh + (offs_s[:, None] * stride_os + offs_d_o[None, :])
    ) # [BLOCK_N, BLOCK_HEADDIM]

    scales_ptr = scales + off_b
    zero_points_ptr = zero_points + off_b

    scales_data = tl.load(scales_ptr)
    zero_points_data = tl.load(zero_points_ptr)

    # loop over x along the sequence dimension
    for start_n in range(0, seqlen, BLOCK_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE)
        # -- load x data ----
        if EVEN_N:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                x_data = tl.load(x_ptrs + start_n * stride_s)
            else:
                x_data = tl.load(x_ptrs + start_n * stride_s, mask=offs_d_i[None, :] * CNTS < headdim, other=0)
        else:
            if EVEN_HEADDIM:
                x_data = tl.load(
                    x_ptrs + start_n * stride_s,
                    mask=(start_n + offs_s)[:, None] < seqlen,
                    other=0,
                )
            else:
                x_data = tl.load(
                    x_ptrs + start_n * stride_s,
                    mask=((start_n + offs_s)[:, None] < seqlen) & (offs_d_i[None, :] * CNTS< headdim),
                    other=0,
                )
        # dequantize k
        tl.debug_barrier()
        BITS_MASK: tl.constexpr = ((1 << BITS) - 1)
        BITS_BASE: tl.constexpr = (CNTS - 1) * BITS
        x_ind = (x_data[:, :, None] >> (BITS_BASE - offs_d_i_cnts[None, None, :]* BITS)) & BITS_MASK
        x_ind = tl.reshape(x_ind, [BLOCK_SIZE, BLOCK_HEADDIM])
        # print(f"[{off_b},{off_h},{start_n}]:\n{x_ind}")
        # tl.device_print("x_ind:", x_ind)
        x_decoded = (x_ind - zero_points_data) / scales_data
        # x_decoded = tl.cast(x_ind, tl.float16)
        if EVEN_N:
            if EVEN_HEADDIM:
                tl.store(o_ptrs + start_n * stride_os, x_decoded.to(tl.float16))
            else:
                tl.store(o_ptrs + start_n * stride_os, x_decoded.to(tl.float16), mask=offs_d_o[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                tl.store(
                    o_ptrs + start_n * stride_os,
                    x_decoded.to(tl.float16),
                    mask=(start_n + offs_s)[:, None] < seqlen,
                )
            else:
                tl.store(
                    o_ptrs + start_n * stride_os,
                    x_decoded.to(tl.float16),
                    mask=((start_n + offs_s)[:, None] < seqlen) & (offs_d_o[None, :] < headdim),
                )


def decode_triton(x, bits, scales, zero_points):
    cnts = 8 // bits
    batch, seqlen, nheads, d_x = x.shape
    d_o = d_x * cnts

    print(f"batch: {batch}, seqlen_q: {seqlen}, nheads: {nheads}, d_x: {d_x}, d_o: {d_o}")
    o = torch.empty((batch, seqlen, nheads, d_o), device=x.device, dtype=scales.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d_o), 16)
    assert BLOCK_HEADDIM % cnts == 0, f"BLOCK_HEADDIM(={BLOCK_HEADDIM}) must be divisible by cnts(={cnts})"
    
    BLOCK_SIZE = 128
    num_warps = 4 if d_o <= 64 else 8
    grid = lambda META: (1, batch * nheads)
    decode_kernel[grid](
        x,  # b, s, hn, dh
        o,
        scales,  # b
        zero_points,  # b
        x.stride(0), # stride_b
        x.stride(2), # stride_h
        x.stride(1), # stride_s
        o.stride(0), # stride_b
        o.stride(2), # stride_h
        o.stride(1), # stride_s
        seqlen,
        nheads,
        d_o,  # output head dim
        BITS=bits,
        CNTS=cnts,
        BLOCK_HEADDIM=BLOCK_HEADDIM, # Always EVEN HEAD
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return o


def encode_torch(x, bits):
    compression_scale = 8 // bits
    shp = list(x.shape)
    assert shp[-1] % compression_scale == 0, f"Last dimension of input tensor must be divisible by {compression_scale}"
    shp[-1] = shp[-1] // compression_scale
    
    x_view = x.view(*shp, compression_scale)
    x_encoded = torch.zeros(*shp, dtype=torch.uint8, device=x.device)
    for bit_pos in range(compression_scale):
        x_encoded |= (x_view[..., bit_pos] & ((1 << bits) - 1)) << ((compression_scale - 1 - bit_pos) * bits)
    return x_encoded


def decode_torch(x, bits, scales, zero_points):
    compression_scale = 8 // bits
    shp = list(x.shape)
    x_decoded = torch.zeros(*shp, compression_scale, dtype=torch.uint8, device=x.device)

    for bit_pos in range(compression_scale):
        x_decoded[..., bit_pos] = (x >> ((compression_scale - 1 - bit_pos) * bits)) & ((1 << bits) - 1)

    x_decoded = x_decoded.view(*shp[:-1], -1)
    shp_ = [-1] + (len(shp)-1) * [1]
    v = (x_decoded  - zero_points.view(shp_)) / scales.view(shp_)
    return x_decoded, v


def check_diff(rst, rst_expected, verbose=True):
    is_allclose = torch.allclose(rst, rst_expected)
    max_err = (rst - rst_expected).abs().max()
    mean_err = (rst - rst_expected).abs().mean()
    max_err_pos = (rst - rst_expected).abs().argmax()
    max_err_pos = torch.unravel_index(max_err_pos, rst.shape)
    if verbose:
        print(f"Position of maximum error: {max_err_pos}")
        print(f"Maximum error: {max_err}, Mean error: {mean_err}, Allclose: {is_allclose}")
    return is_allclose, max_err, mean_err


def test_encode_decode():
    torch.manual_seed(0)
    batch_size,seq_len, head_size, headdim = 100, 3201, 48, 128
    # batch_size,seq_len, head_size, headdim = 2, 128, 4, 32
    # batch_size,seq_len, head_size, headdim = 120, 127*3, 48, 128
    bits = 2
    p0 = 1

    x = torch.randint(0, 4, (batch_size, seq_len, head_size, headdim), device='cuda', dtype=torch.uint8)
    print(f"{x.shape=},\n{x[p0, 0, 0, :]=}")
    scales = torch.abs(torch.randn(batch_size, dtype=torch.float16, device='cuda'))
    zero_points = torch.randn(batch_size, dtype=torch.float16, device='cuda')

    shp_ = [-1] + (len(list(x.shape))-1) * [1]
    v = (x - zero_points.view(shp_)) / scales.view(shp_)
    print(f"{v.shape=},\n{v[p0, 0, 0, :]=}")

    x_encoded = encode_torch(x, bits)
    print(f"{x_encoded.shape=},\n{x_encoded[p0, 0, 0, :]=}")
    # x_decoded, v_decoded = decode_torch(x_encoded, bits, scales, zero_points)
    # print(f"{v_decoded.shape=},\n{v_decoded[p0, 0, 0, :]=}")
    # torch.testing.assert_close(x, x_decoded)
    # torch.testing.assert_close(v, v_decoded)
    o_decoded = decode_triton(x_encoded, bits, scales, zero_points)
    print(f"{o_decoded.shape=},\n{o_decoded[p0, 0, 0, :]=}")
    is_allclose, max_err, mean_err = check_diff(v, o_decoded)
    # torch.testing.assert_close(x, o_decoded.to(torch.uint8))
    torch.testing.assert_close(v, o_decoded)
    print("Passed!")


def run_quantization(val, bits = 2):
    # b,s,hn,dh
    shp = val.shape
    batch_size = shp[0]

    val_view = val.view(batch_size, -1)

    val_min = val_view.min(dim=-1).values.unsqueeze_(-1)
    val_max = val_view.max(dim=-1).values.unsqueeze_(-1)

    quant_levels = 2**bits - 1
    scale = quant_levels / (val_max - val_min).to(torch.float32)
    zero_point = -val_min * scale

    ind = torch.round(torch.clamp(val_view * scale + zero_point, 0, quant_levels)).to(torch.uint8)
    lut = val_min + torch.arange(0, 2**bits, device=val.device) / scale
    
    val_deq = (ind - zero_point) / scale
    val_deq = val_deq.reshape(shp)

    ind = ind.reshape(shp)
    lut = lut.reshape(batch_size, -1).to(val.dtype)

    return val_deq, ind, lut, val_min, val_max, scale, zero_point


def get_tensors(batch_size, seq_len, head_size, dim):
    q = torch.randn((batch_size, 1, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)

    return q, k, v


def do_bench(fn, warmup, rep, mode:str='fwd'):
    if mode == 'fwd':
        # with torch.no_grad():
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.2, 0.5, 0.8])
    elif mode == 'bwd':
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.2, 0.5, 0.8])
    else: # mode == 'fwd+bwd'
        q20_fwd, median_fwd, q80_fwd = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.2, 0.5, 0.8])
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        q20_bwd, median_bwd, q80_bwd = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.2, 0.5, 0.8])
        return q20_fwd+q20_bwd, median_fwd+median_bwd, q80_fwd+q80_bwd


def run_flash_attn(batch_size, head_size, seq_len, dim, causal, mode, impl="triton", warmup=20, rep=100):
    # torch.manual_seed(0)
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)
    k_bits = 1
    v_bits = 2
    k_deq, k_ind, k_lut, k_min, k_max, k_scales, k_zero_points = run_quantization(k, bits=k_bits)
    v_deq, v_ind, v_lut, v_min, v_max, v_scales, v_zero_points = run_quantization(v, bits=v_bits)

    k_ind_encoded = encode_torch(k_ind, k_bits)
    v_ind_encoded = encode_torch(v_ind, v_bits)

    k = k_deq.to(q.dtype)
    v = v_deq.to(q.dtype)
    
    try:
        if impl != "cuda":
            rst_expected = flash_attn_func_cuda(q, k, v, causal=causal) # flash_attn: (B, S, H, D)
        
            if impl == "triton":
                # # No encoding:
                rst, _, deb = flash_attn_func(q, k, v, k_bits, k_scales, k_zero_points, v_bits, v_scales, v_zero_points, None, causal, None)
                # # Encoded:
                # rst, _, deb = flash_attn_func(q, k_ind_encoded, v_ind_encoded, k_bits, k_scales, k_zero_points, v_bits, v_scales, v_zero_points, None, causal, None)
                # rst = flash_attn_func(q, k, v, None, causal, None)[0]
                print("flash attn output shape:", rst.shape)

                # check_diff(deb[:,0:seq_len,:,:], k.to(deb.dtype))
            elif impl == "amd":
                rst = flash_attn_func_amd(q, k, v, causal)[0]
            elif impl == "alex":
                rst = flash_attn_func_alex(q, k, v)

            is_allclose = torch.allclose(rst, rst_expected)
            max_err = (rst - rst_expected).abs().max()
            mean_err = (rst - rst_expected).abs().mean()
            print(f"[{impl}]Maximum error: {max_err}, Mean error: {mean_err}, Allclose: {is_allclose}")
    except Exception as e:
        print(f"Accuracy test exception: {e}")
    
    if impl == "cuda":
        if flash_attn_func_cuda is None:
            raise ImportError("Please install flash_attn (pip install flash-attn --no-build-isolation)")
        fn = lambda: flash_attn_func_cuda(q, k, v, causal=causal)
    elif impl == "triton":
        # # No encoding:
        fn = lambda: flash_attn_func(q, k, v, k_bits, k_scales, k_zero_points, v_bits, v_scales, v_zero_points, None, causal, None)[0]
        # # Encoded:
        # fn = lambda: flash_attn_func(q, k_ind_encoded, v_ind_encoded, k_bits, k_scales, k_zero_points, v_bits, v_scales, v_zero_points, None, causal, None)[0]
    elif impl == "amd":
        fn = lambda: flash_attn_func_amd(q, k, v, causal)[0]
    elif impl == "alex":
        fn = lambda: flash_attn_func_alex(q, k, v)
    else:  # impl == "xformers"
        fn = lambda: flash_attn_func_xformers(q, k, v, None, causal, None)[0]

    return do_bench(fn, warmup, rep, mode=mode)


def run_hyper_attn(batch_size, head_size, seq_len, dim, causal, mode, impl="triton", warmup=20, rep=100):
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)[:3]
    block_size = 256
    sample_size = 256

    attn = HyperAttention(
        input_dim=dim,
        block_size=block_size,
        sample_size=sample_size,
        min_seq_len=4096,
        impl=impl).to(device='cuda', dtype=q.dtype)

    fn = lambda: attn(q, k, v, causal=causal)

    return do_bench(fn, warmup, rep, mode=mode)


def main():
    args = get_arguments()
    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    # seq_lens = [2**i for i in range(16, 18)]
    seq_lens = [128, 3328]

    attn_method = args.attn_method # ['flash', 'hyper']
    attn_impl = args.impl # ['cuda', 'triton', 'xformers']
    mode = args.mode # ['fwd', 'bwd', 'fwd+bwd']
    batch_size, head_size, dim = args.batch_size, args.head_size, args.dim
    print(f"mode: {mode}, attn_method: {attn_method}-{attn_impl}, batch_size: {batch_size}, head_size: {head_size}, dim: {dim}")

    # causal = not args.no_causal
    causal = False

    for seq_len in seq_lens:
        if attn_method == 'flash':
            ms = run_flash_attn(batch_size, head_size, seq_len, dim, causal, mode=args.mode, impl=attn_impl)
        elif attn_method == 'hyper':
            ms = run_hyper_attn(batch_size, head_size, seq_len, dim, causal, mode=args.mode, impl=attn_impl)
        else:
            raise NotImplementedError

        print(f"[{mode:<8}], {attn_method}, seq_len: {seq_len:<8}, causal: {causal}, ms: {ms[0]:5.5f} ({ms[1]:5.5f}, {ms[2]:5.5f}) | ")


if __name__ == "__main__":
    # test_encode_decode()
    main()
