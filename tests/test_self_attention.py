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

def check_diff(rst, ref, lens, LAYOUT, prefix=""):
    atol = 1e-3
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
# @pytest.mark.parametrize("H", [1, 6, 8, 48], ids=lambda x: f"heads-{x}")
# @pytest.mark.parametrize("T", [1, 10, 16, 800, 1025], ids=lambda x: f"time-{x}")
# @pytest.mark.parametrize("autotune", [False, True], ids=lambda x: f"autotune-{x}")

# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=lambda x: f"{x}")
@pytest.mark.parametrize("dtype", [torch.float16], ids=lambda x: f"{x}")
@pytest.mark.parametrize(
    "lens", ["none"], ids=lambda x: f"lens-{x}"
)
@pytest.mark.parametrize(
    "noncontiguous", [False], ids=lambda x: f"noncontiguous-{x}"
)
@pytest.mark.parametrize("HEAD_DIM", [128], ids=lambda x: f"dim-{x}")
@pytest.mark.parametrize("B", [1], ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("H", [48], ids=lambda x: f"heads-{x}")
@pytest.mark.parametrize("T", [16], ids=lambda x: f"time-{x}")
@pytest.mark.parametrize("autotune", [False], ids=lambda x: f"autotune-{x}")
@pytest.mark.parametrize("LAYOUT", ["bhsd", "bshd"], ids=lambda x: f"dim-{x}")
def test_self_attention(
    B,
    H,
    T,
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

    if os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16:
        pytest.skip("skipping bf16 in interpreter mode")

    if autotune and not (
        T in {16, 800} and H == 1 and B == 67 and noncontiguous and lens == "tricky"
    ):
        pytest.skip("reduced tests for autotune")

    shape_mul = 2 if noncontiguous else 1

    if LAYOUT == "bhsd":
        shape_tuple = (B * shape_mul, H * shape_mul, T * shape_mul, HEAD_DIM * shape_mul)    
    else:
        shape_tuple = (B * shape_mul, T * shape_mul, H * shape_mul, HEAD_DIM * shape_mul)

    # q, k, v = [ # (B, H, T:SEQLEN, HEAD_DIM)
    #     torch.testing.make_tensor(
    #         shape_tuple,
    #         dtype=dtype,
    #         device="cuda",
    #         requires_grad=True,
    #         noncontiguous=noncontiguous,
    #         low=-0.1,
    #         high=0.1,
    #     )
    #     for _ in range(3)
    # ]

    q = torch.randn(shape_tuple, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(shape_tuple, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(shape_tuple, dtype=dtype, device="cuda", requires_grad=True)

    if noncontiguous:
        slice_obj = slice(shape_mul-1, None, shape_mul)
        q = q[slice_obj, slice_obj, slice_obj, slice_obj].detach().clone().requires_grad_()
        k = k[slice_obj, slice_obj, slice_obj, slice_obj].detach().clone().requires_grad_()
        v = v[slice_obj, slice_obj, slice_obj, slice_obj].detach().clone().requires_grad_()

    # single seqlen_q
    q = q[:, :, :1, :] if LAYOUT == "bhsd" else q[:, :1, :, :]

    if lens == "none":
        lens = None
    elif lens == "tricky":
        tricky_lens = [
            1,
            2,
            5,
            T + 1,
            T,
            max(T // 2, 1),
            max(T // 4, 1),
        ]
        lens = torch.tensor(
            np.random.choice(tricky_lens, B), dtype=torch.int32, device="cuda"
        )
    else:
        lens = torch.randint(1, T + 1, (B,), dtype=torch.int32, device="cuda")

    ref, res_mask = self_attention_reference(q, k, v, lens, layout=LAYOUT)
    tri_out = self_attention_for_layout(q, k, v, lens, autotune=autotune, layout=LAYOUT)
    ref_fa2 = self_attention_fa2(q, k, v, layout=LAYOUT)
    # ref, res_mask = self_attention_reference(q.detach().clone(), k.detach().clone(), v.detach().clone(), lens, layout=LAYOUT)
    # ref_fa2 = self_attention_fa2(q.detach().clone(), k.detach().clone(), v.detach().clone(), layout=LAYOUT)
    # tri_out = self_attention_for_layout(q.detach().clone(), k.detach().clone(), v.detach().clone(), lens, autotune=autotune, layout=LAYOUT)

    # torch.set_printoptions(linewidth=400, profile="full")
    ref_fa2 = ref_fa2 * res_mask.broadcast_to(ref_fa2.shape)
    check_diff(ref_fa2, ref, lens, LAYOUT, prefix="ref_a2")

    tri_out = tri_out * res_mask.broadcast_to(tri_out.shape)
    check_diff(tri_out, ref, lens, LAYOUT, prefix="tri_out")
    print("Passed!")


if __name__ == "__main__":
    test_self_attention(
        B=40,
        H=48,
        T=1025,
        HEAD_DIM=128,
        LAYOUT="bhsd",
        dtype=torch.float16,
        lens="none",
        noncontiguous=True,
        autotune=False,
    )