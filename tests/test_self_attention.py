import os

import numpy as np
import pytest
import torch

from attention.flash_attn2.flash_attn_triton_alexdremov import (
    self_attention, 
    self_attention_for_layout,
    self_attention_reference,
    self_attention_fa2,
)

def check_diff(rst, rst_expected, prefix="", verbose=True):
    is_allclose = torch.allclose(rst, rst_expected)
    max_err = (rst - rst_expected).abs().max()
    mean_err = (rst - rst_expected).abs().mean()
    max_err_pos = (rst - rst_expected).abs().argmax()
    max_err_pos = torch.unravel_index(max_err_pos, rst.shape)
    if verbose:
        print(f"[{prefix}]Position of maximum error: {[i.item() for i in max_err_pos]}")
        print(f"[{prefix}]Maximum error: {max_err}, Mean error: {mean_err}, Allclose: {is_allclose}")
    return is_allclose, max_err, mean_err


def test_diff(rst, ref, lens, LAYOUT, atol = 0.05, prefix=""):
    diff = rst - ref
    if LAYOUT == "bshd":
        diff = diff.transpose(1, 2)
    errors = abs(diff) > atol
    b_mismatch = torch.argmax(errors.sum((1, 2, 3)).view(-1)).item()
    h_mismatch = torch.argmax(errors[b_mismatch].sum((1, 2)).view(-1)).item()

    torch.testing.assert_close(
        ref,
        rst,
        atol=atol,
        rtol=0,
        msg=lambda x: f"{x}\n\n[{prefix}]{(b_mismatch, h_mismatch)}:\n{(errors[b_mismatch, h_mismatch]).long()} \n\n {diff[errors].view(-1)}\n\nlens:\n{lens}\n{ref}\n{rst}",
    )


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=lambda x: f"{x}")
# @pytest.mark.parametrize(
#     "lens", ["none", "tricky", "random"], ids=lambda x: f"lens-{x}"
# )
# @pytest.mark.parametrize(
#     "noncontiguous", [False, True], ids=lambda x: f"noncontiguous-{x}"
# )
# @pytest.mark.parametrize("HEAD_DIM", [16, 128, 256], ids=lambda x: f"dim-{x}")
# @pytest.mark.parametrize("B", [1, 40, 64], ids=lambda x: f"batch-{x}")
# @pytest.mark.parametrize("H", [1, 6, 8], ids=lambda x: f"heads-{x}")
# @pytest.mark.parametrize("Tq", [-1], ids=lambda x: f"Tq-{x}")
# @pytest.mark.parametrize("Tk", [1, 10, 16, 800, 1025], ids=lambda x: f"Tk-{x}")
# @pytest.mark.parametrize("autotune", [False, True], ids=lambda x: f"autotune-{x}")

@pytest.mark.parametrize("dtype", [torch.float16], ids=lambda x: f"{x}")
@pytest.mark.parametrize(
    "lens", ["none"], ids=lambda x: f"lens-{x}"
)
@pytest.mark.parametrize(
    "noncontiguous", [False], ids=lambda x: f"noncontiguous-{x}"
)
@pytest.mark.parametrize("HEAD_DIM", [128], ids=lambda x: f"dim-{x}")
@pytest.mark.parametrize("B", [1, 40], ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("H", [48], ids=lambda x: f"heads-{x}")
@pytest.mark.parametrize("Tq", [16], ids=lambda x: f"Tq-{x}")
@pytest.mark.parametrize("Tk", [1025], ids=lambda x: f"Tk-{x}")
@pytest.mark.parametrize("autotune", [False], ids=lambda x: f"autotune-{x}")
@pytest.mark.parametrize("LAYOUT", ["bhsd", "bshd"], ids=lambda x: f"dim-{x}")
def test_self_attention(
    B,
    H,
    Tq,
    Tk,
    HEAD_DIM,
    LAYOUT,
    dtype,
    lens,
    noncontiguous,
    autotune,
):
    torch._dynamo.reset()

    torch.manual_seed(20)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()

    if Tq == -1:
        Tq = Tk

    assert Tk >= Tq, f"{(Tk, Tq)}"

    if os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16:
        pytest.skip("skipping bf16 in interpreter mode")

    if autotune and not (
        Tk in {16, 800} and H == 1 and B == 67 and noncontiguous and lens == "tricky"
    ):
        pytest.skip("reduced tests for autotune")

    shape_mul = 2 if noncontiguous else 1

    if LAYOUT == "bhsd":
        shape_tuple_q = (B * shape_mul, H * shape_mul, Tq * shape_mul, HEAD_DIM * shape_mul)
        shape_tuple_k = (B * shape_mul, H * shape_mul, Tk * shape_mul, HEAD_DIM * shape_mul)
    else:
        shape_tuple_q = (B * shape_mul, Tq * shape_mul, H * shape_mul, HEAD_DIM * shape_mul)
        shape_tuple_k = (B * shape_mul, Tk * shape_mul, H * shape_mul, HEAD_DIM * shape_mul)

    if noncontiguous:
        slice_obj = slice(shape_mul-1, None, shape_mul)
    else:
        slice_obj = slice(None)

    # (B, H, SEQLEN, HEAD_DIM)
    val_mag = 1.0
    requires_grad = True
    q = torch.testing.make_tensor(
            shape_tuple_q,
            dtype=dtype,
            device="cuda",
            requires_grad=requires_grad,
            noncontiguous=noncontiguous,
            low=-val_mag,
            high=val_mag,
        )[slice_obj, slice_obj, slice_obj, slice_obj].detach().clone()
    k, v = [
        torch.testing.make_tensor(
            shape_tuple_k,
            dtype=dtype,
            device="cuda",
            requires_grad=requires_grad,
            noncontiguous=noncontiguous,
            low=-val_mag,
            high=val_mag,
        )[slice_obj, slice_obj, slice_obj, slice_obj].detach().clone()
        for _ in range(2)
    ]
    
    if requires_grad:
        q = q.requires_grad_()
        k = k.requires_grad_()
        v = v.requires_grad_()
    
    if lens == "none":
        lens = None
    elif lens == "tricky":
        tricky_lens = [
            1,
            2,
            5,
            Tq + 1,
            Tq,
            max(Tq // 2, 1),
            max(Tq // 4, 1),
        ]
        lens = torch.tensor(
            np.random.choice(tricky_lens, B), dtype=torch.int32, device="cuda"
        )
    else:
        lens = torch.randint(1, Tq + 1, (B,), dtype=torch.int32, device="cuda")

    ref, res_mask = self_attention_reference(q, k, v, lens, layout=LAYOUT)
    tri_out = self_attention_for_layout(q, k, v, lens, autotune=autotune, layout=LAYOUT)
    ref_fa2 = self_attention_fa2(q, k, v, layout=LAYOUT)

    # torch.set_printoptions(linewidth=400, profile="full")
    ref_fa2 = ref_fa2 * res_mask.broadcast_to(ref_fa2.shape)
    test_diff(ref_fa2, ref, lens, LAYOUT, prefix="ref_fa2")

    tri_out = tri_out * res_mask.broadcast_to(tri_out.shape)
    test_diff(tri_out, ref, lens, LAYOUT, prefix="tri_out")
    print("Passed!")


def get_tensors(batch_size, seq_len, head_size, dim, dtype=torch.bfloat16):
    q = torch.randn((batch_size, 1, head_size, dim), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((batch_size, seq_len, head_size, dim), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((batch_size, seq_len, head_size, dim), dtype=dtype, device="cuda", requires_grad=True)
    return q, k, v


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


def run_flash_attn(batch_size, head_size, seq_len, dim, causal=False, mode="alex", impl="triton", warmup=20, rep=100):
    # torch.manual_seed(0)
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)

    def print_stats(t, prefix=""):
        print(f"{prefix}: min={t.min().item()}, max={t.max().item()}, mean={t.mean().item()}, std={t.std().item()}")

    # Statics of q, k and v
    print_stats(q, "q_org")
    print_stats(k, "k_org")
    print_stats(v, "v_org")

    k_bits = 1
    v_bits = 2
    k_deq, k_ind, k_lut, k_min, k_max, k_scales, k_zero_points = run_quantization(k, bits=k_bits)
    v_deq, v_ind, v_lut, v_min, v_max, v_scales, v_zero_points = run_quantization(v, bits=v_bits)

    k_ind_encoded = encode_torch(k_ind, k_bits)
    v_ind_encoded = encode_torch(v_ind, v_bits)

    k_deq = k_deq.to(q.dtype)
    v_deq = v_deq.to(q.dtype)

    k_deq_ref = (k_ind - k_zero_points[:, :, None, None])/k_scales[:, :, None, None]
    check_diff(k_deq, k_deq_ref.to(q.dtype), prefix="k_deq")

    print_stats(k_deq, "k_deq")
    print_stats(v_deq, "v_deq")

    k = k_deq
    v = v_deq

    print_stats(k, "k_final")
    print_stats(v, "v_final")

    lens = None
    LAYOUT = "bshd"

    ref, res_mask = self_attention_reference(q, k, v, lens, layout=LAYOUT)
    tri_out = self_attention_for_layout(q, k_ind_encoded, v_ind_encoded, k_scales, k_zero_points, v_scales, v_zero_points, lens, autotune=False, layout=LAYOUT)
    ref_fa2 = self_attention_fa2(q, k, v, layout=LAYOUT)

    # torch.set_printoptions(linewidth=400, profile="full")
    ref_fa2 = ref_fa2 * res_mask.broadcast_to(ref_fa2.shape)
    check_diff(ref_fa2, ref, prefix="ref_fa2")

    tri_out = tri_out * res_mask.broadcast_to(tri_out.shape)
    check_diff(tri_out, ref, prefix="tri_out")
    test_diff(tri_out, ref, lens, LAYOUT, prefix="tri_out")
    print("Passed!")
    return 


if __name__ == "__main__":
    B=50
    H=48
    Tq=1
    Tk=3200
    HEAD_DIM=128

    run_flash_attn(B, H, Tk, HEAD_DIM)

    # test_self_attention(
    #     B=B,
    #     H=H,
    #     Tq=1,
    #     Tk=Tk,
    #     HEAD_DIM=HEAD_DIM,
    #     LAYOUT="bhsd",
    #     dtype=torch.bfloat16,
    #     lens="none",
    #     noncontiguous=False,
    #     autotune=False,
    # )