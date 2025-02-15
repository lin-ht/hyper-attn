import os
import argparse
import torch
import triton

from attention.flash_attn2.flash_attn_triton_amd import flash_attn_func as flash_attn_func_amd
from attention.flash_attn2.flash_attn_triton_q import flash_attn_func
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
    parser.add_argument("--impl", type=str, default="triton", choices=['cuda', 'triton', 'amd', 'xformers'])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--head_size", type=int, default=48)
    parser.add_argument("--dim", type=int, default=128)
    return parser.parse_args()


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


def decode_torch(x, bits, scales, offsets):
    compression_scale = 8 // bits
    shp = list(x.shape)
    x_decoded = torch.zeros(*shp, compression_scale, dtype=torch.uint8, device=x.device)

    for bit_pos in range(compression_scale):
        x_decoded[..., bit_pos] = (x >> ((compression_scale - 1 - bit_pos) * bits)) & ((1 << bits) - 1)

    x_decoded = x_decoded.view(*shp[:-1], -1)
    shp_ = [-1] + (len(shp)-1) * [1]
    v = x_decoded * scales.view(shp_) + offsets.view(shp_)
    return x_decoded, v


def test_encode_decode():
    # torch.manual_seed(0)
    x = torch.randint(0, 4, (2, 3, 4), device='cuda', dtype=torch.uint8)
    scales = torch.randn(2, dtype=torch.float16, device='cuda')
    offsets = torch.randn(2, dtype=torch.float16, device='cuda')

    shp_ = [-1] + (len(list(x.shape))-1) * [1]
    v = x * scales.view(shp_) + offsets.view(shp_)

    x_encoded = encode_torch(x, 2)
    x_decoded, v_decoded = decode_torch(x_encoded, 2, scales, offsets)
    torch.testing.assert_close(x, x_decoded)
    torch.testing.assert_close(v, v_decoded)


def run_quantization(val, bits = 2):
    # b,s,hn,dh
    shp = val.shape
    batch_size = shp[0]

    val_view = val.view(batch_size, -1)

    val_min = val_view.min(dim=-1).values.unsqueeze_(-1)
    val_max = val_view.max(dim=-1).values.unsqueeze_(-1)

    quant_levels = 2**bits - 1
    scale = quant_levels / (val_max - val_min)
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

    k_deq, k_ind, k_lut, k_min, k_max, k_scale, k_zero_point = run_quantization(k, bits=2)
    return q, k, v, k_deq, k_ind, k_lut, k_scale, k_zero_point


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
    q, k, v, k_deq, k_ind, k_lut, k_scale, k_zero_point = get_tensors(batch_size, seq_len, head_size, dim)

    k = k_ind.to(q.dtype)
    
    try:
        if impl != "cuda":
            rst_expected = flash_attn_func_cuda(q, k, v, causal=causal)
        
            if impl == "triton":
                rst = flash_attn_func(q, k, v, None, causal, None)[0]
                print("flash attn output shape:", rst.shape)
            elif impl == "amd":
                rst = flash_attn_func_amd(q, k, v, causal)[0]

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
        fn = lambda: flash_attn_func(q, k, v, None, causal, None)[0]
    elif impl == "amd":
        fn = lambda: flash_attn_func_amd(q, k, v, causal)[0]
    else:  # impl == "xformers"
        fn = lambda: flash_attn_func_xformers(q, k, v, None, causal, None)[0]

    return do_bench(fn, warmup, rep, mode=mode)


def run_hyper_attn(batch_size, head_size, seq_len, dim, causal, mode, impl="triton", warmup=20, rep=100):
    q, k, v = get_tensors(batch_size, head_size, seq_len, dim)[:3]
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
    seq_lens = [2500, 3328]

    attn_method = args.attn_method # ['flash', 'hyper']
    attn_impl = args.impl # ['cuda', 'triton', 'xformers']
    mode = args.mode # ['fwd', 'bwd', 'fwd+bwd']
    batch_size, head_size, dim = args.batch_size, args.head_size, args.dim
    print(f"mode: {mode}, attn_method: {attn_method}-{attn_impl}, batch_size: {batch_size}, head_size: {head_size}, dim: {dim}")

    causal = not args.no_causal

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
