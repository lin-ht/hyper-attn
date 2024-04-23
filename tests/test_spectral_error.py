import torch
import pytest

from attention.hyper_attn.hyper_attn_org import HyperAttention as HyperAttentionOrg
from attention.hyper_attn.hyper_attn import HyperAttention
from attention.hyper_attn.utils import (
    exact_attention_xformers,
    exact_attention_cuda,
    exact_attention,
)


def get_tensors(batch_size, seq_len, head_size, dim, requires_grad=False):
    q = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
    k = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
    v = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
    return q, k, v


class HyperAttentionConfig:
    def __init__(self, input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=4096, impl='xformers'):
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.impl = impl

TEST_HYPER_ATTN_CONFIGS = [
    # HyperAttentionConfig(input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=2048, impl='xformers'),
    HyperAttentionConfig(input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=2048, impl='cuda'),
    # HyperAttentionConfig(input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=2048, impl='triton'),
]

TEST_CASES = [
    (4, 8, 32768, 128, False),
    # (1, 32, 2**16, 64, False),
    # (1, 32, 2**16, 64, True),
]

@pytest.mark.parametrize("config", TEST_HYPER_ATTN_CONFIGS)
@pytest.mark.parametrize(
    ["batch_size", "head_size", "seq_len", "dim", "causal"], TEST_CASES
)
def test_spectral_error(config: HyperAttentionConfig, batch_size, head_size, seq_len, dim, causal):
    seed = 42 #1234
    mode = 'fwd'
    # ord_list = [2, 1, float('inf')]
    ord_list = ['fro']

    # set seed
    torch.manual_seed(seed)

    if config.input_dim != dim:
        print(f"Warning: config.input_dim({config.input_dim}) != dim({dim}), reassigning config.input_dim to dim")
        config.input_dim = dim

    q, k, v = get_tensors(batch_size, head_size, seq_len, dim)
    print(f"q[0, 0, 0, :] = \n{q[0, 0, 0, :]}")
    print(f"k[0, 0, 0, :] = \n{k[0, 0, 0, :]}")
    print(f"v[0, 0, 0, :] = \n{v[0, 0, 0, :]}")

    with torch.no_grad():
        softmax_scale = dim ** (-0.5)
        exact_attn, exact_lse = exact_attention_xformers(q, k, v, softmax_scale, causal=causal)
        exact_attn_f32 = exact_attn.to(torch.float32)
        exact_lse_f32 = exact_lse.to(torch.float32)

        exact_attn2, exact_lse2 = exact_attention_cuda(q, k, v, softmax_scale, causal=causal)
        # exact_attn, exact_lse = exact_attention(q, k, v, softmax_scale, causal=causal)
        exact_attn2_f32 = exact_attn2.to(torch.float32)
        exact_lse2_f32 = exact_lse2.to(torch.float32)

        attn_hyper0 = HyperAttentionOrg(
            input_dim=dim,  # config.input_dim == dim
            block_size=config.block_size,
            sample_size=config.sample_size,
            min_seq_len=config.min_seq_len,
            impl="cuda").to(device='cuda', dtype=q.dtype)

        attn_hyper = HyperAttention(
            input_dim=dim,  # config.input_dim == dim
            block_size=config.block_size,
            sample_size=config.sample_size,
            min_seq_len=config.min_seq_len,
            impl=config.impl).to(device='cuda', dtype=q.dtype)

        rst_attn0, rst_lse0 = attn_hyper0(q, k, v, causal=causal, return_lse=True)
        rst_attn0_f32 = rst_attn0.to(torch.float32)
        rst_lse0_f32 = rst_lse0.to(torch.float32)

        rst_attn, rst_lse = attn_hyper(q, k, v, causal=causal, return_lse=True)
        rst_attn_f32 = rst_attn.to(torch.float32)
        rst_lse_f32 = rst_lse.to(torch.float32)

        # Use restricter left side value ||Ax||_p <= ||A||_p * ||x||_p
        # to calculate the spectral error ratio.
        diff_attn_f32 = torch.abs(rst_attn_f32 - exact_attn_f32)
        diff_lse_f32 = torch.abs(rst_lse_f32 - exact_lse_f32)

        diff_attn0_f32 = torch.abs(rst_attn0_f32 - exact_attn_f32)
        diff_lse0_f32 = torch.abs(rst_lse0_f32 - exact_lse_f32)

        print(f"rst_attn0[0, 0, 0, :] = \n{rst_attn0[0, 0, 0, :]}")
        print(f"rst_attn[0, 0, 0, :] = \n{rst_attn[0, 0, 0, :]}")
        print(f"exact_attn[0, 0, 0, :] = \n{exact_attn[0, 0, 0, :]}")
        print(f"diff_attn_f32[0, 0, 0, :] = \n{diff_attn_f32[0, 0, 0, :]}")

        print(f"rst_lse0[0, 0, :10, :] = \n{torch.squeeze(rst_lse0[0, 0, :10, :])}")
        print(f"rst_lse[0, 0, :10, :] = \n{torch.squeeze(rst_lse[0, 0, :10, :])}")
        print(f"exact_lse[0, 0, :10, :] = \n{torch.squeeze(exact_lse[0, 0, :10, :])}")
        print(f"diff_lse_f32[0, 0, :10, :] = \n{torch.squeeze(diff_lse_f32[0, 0, :10, :])}")

        print(f"torch.norm reference: {torch.norm(exact_attn - rst_attn0, p='fro') / torch.norm(exact_attn, p='fro')}")
        for ord_ in ord_list:
            log_prefix = f"[{config.impl:<8}], seq_len: {seq_len:<8}, causal: {causal}, ord: {ord_}, "
            compute_error_ratio(diff_attn_f32, diff_lse_f32, exact_attn_f32, exact_lse_f32, ord=ord_, log_prefix=log_prefix + "new ")
            compute_error_ratio(diff_attn0_f32, diff_lse0_f32, exact_attn_f32, exact_lse_f32, ord=ord_, log_prefix=log_prefix + "old ")


def compute_error_ratio(diff_attn, diff_lse, exact_attn, exact_lse, ord='fro', log_prefix=""):
    diff_attn_norm = torch.linalg.matrix_norm(diff_attn, ord=ord)  # By default: dim = (-2, -1)
    exact_attn_norm = torch.linalg.matrix_norm(exact_attn, ord=ord)
    spectral_error_ratio = diff_attn_norm / exact_attn_norm
    max_spectral_error_ratio = spectral_error_ratio.max().item()

    lse_error_ratio = diff_lse/torch.abs(exact_lse)
    max_lse_error_ratio = lse_error_ratio.max().item()
    print(f"{log_prefix}max_spectral_error_ratio: {max_spectral_error_ratio:.5f}, max_lse_error_ratio: {max_lse_error_ratio:.5f} | ")
    return max_spectral_error_ratio, max_lse_error_ratio


if __name__ == "__main__":
    # pytest.main([__file__])
    test_spectral_error(TEST_HYPER_ATTN_CONFIGS[0], *(TEST_CASES[0]))
