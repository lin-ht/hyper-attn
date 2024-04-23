import time
import torch
import pytest

from attention.hyper_attn.hyper_attn_org import HyperAttention as HyperAttentionOrg
from attention.hyper_attn.hyper_attn import HyperAttention
from attention.hyper_attn.utils import (
    exact_attention_xformers,
    exact_attention_cuda,
    exact_attention,
)

class HyperAttentionConfig:
    def __init__(self, input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=4096, impl='xformers'):
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.impl = impl


def get_tensors(batch_size, seq_len, head_size, dim, requires_grad:bool=False, config:HyperAttentionConfig|None=None, noise_scale:float=0.01, permute:bool=True):
    if config is None:
        q = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        k = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        v = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
    else:
        start_time = time.time()
        n_bases = config.block_size
        n_blocks = (seq_len + n_bases - 1) // n_bases
        seq_len_sup = n_blocks * n_bases
        n_padding = seq_len_sup - seq_len
        k_sup = torch.randn((batch_size, seq_len_sup, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        if n_padding > 0:
            k_sup[:, seq_len:, :, :] = k_sup[:, seq_len-n_padding:seq_len, :, :]

        s = torch.rand((batch_size, seq_len, head_size, n_bases), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        s = s / s.sum(dim=-1, keepdim=True)

        q = noise_scale * torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        k_sup_view = k_sup.view(batch_size, n_blocks, n_bases, head_size, dim)
        for i in range(n_blocks):
            s_block = s[:, i*n_bases:i*n_bases+n_bases, :, :]
            k_sup_view_block = k_sup_view[:, i, :, :, :]
            q_block = torch.einsum("blhn,bnhd->blhd", s_block, k_sup_view_block)
            q[:, i*n_bases:i*n_bases+n_bases, :, :] += q_block

        v = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        # Apply random permutation:
        if permute:
            k = torch.zeros_like(v)
            for i in range(batch_size):
                for j in range(head_size):
                    q[i, :, j, :] = q[i, torch.randperm(seq_len), j, :]
                    k[i, :, j, :] = k_sup[i, torch.randperm(seq_len), j, :]
        else:
            k = k_sup[:, :seq_len, :, :]

        total_time = time.time() - start_time
        print(f"get_tensors run time: {total_time:.3f} seconds.")

    return q, k, v


def test_get_tensors():
    torch.manual_seed(42)

    batch_size = 4
    head_size = 32
    seq_len = 128
    dim = 64

    config = HyperAttentionConfig(input_dim=dim, lsh_num_projs=7, block_size=16, sample_size=16, min_seq_len=32, impl='xformers')

    t_0 = time.time()
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim, config=config, noise_scale=0, permute=False)

    # q.shape = (batch_size, seq_len, head_size, dim)
    q_ = q.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim)
    k_ = k.permute(0, 2, 3, 1).reshape(batch_size * head_size, dim, seq_len)
    w_ = torch.bmm(q_, k_) * (dim ** (-0.5))  # (batch_size * head_size, seq_len, seq_len)
    a = torch.nn.functional.softmax(w_, dim=2).reshape(batch_size, head_size, seq_len, seq_len)

    k_as_q = k.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim)
    w_k = torch.bmm(k_as_q, k_) * (dim ** (-0.5))
    a_k = torch.nn.functional.softmax(w_k, dim=2).reshape(batch_size, head_size, seq_len, seq_len)
    a_k_diag = a_k * torch.eye(seq_len, dtype=torch.bfloat16, device=a.device).unsqueeze_(0).unsqueeze_(0)

    ord = 'fro'
    diff_a_k = a_k - a_k_diag
    diff_a_k_norm = torch.linalg.matrix_norm(diff_a_k, ord=ord)  # By default: dim = (-2, -1)
    exact_a_k_norm = torch.linalg.matrix_norm(a_k, ord=ord)
    spectral_error_ratio_a_k = diff_a_k_norm / exact_a_k_norm
    max_spectral_error_ratio_ref = spectral_error_ratio_a_k.max().item()

    print(f"a_k[0, 0, 0:{config.block_size}, 0:{config.block_size}] = \n{a_k[0, 0, 0:config.block_size, 0:config.block_size]}")
    print(f"a_k_diag[0, 0, 0:{config.block_size}, 0:{config.block_size}] = \n{a_k_diag[0, 0, 0:config.block_size, 0:config.block_size]}")
    print(f"For maxtrix a {a.shape}: max_spectral_error_ratio_ref is {max_spectral_error_ratio_ref:.5f}")

    block2d = torch.ones((config.block_size, config.block_size), dtype=torch.bfloat16, device=a.device, requires_grad=False)
    block2d_list = [block2d for _ in range(seq_len // config.block_size)] + [block2d[:seq_len % config.block_size, :seq_len % config.block_size]]
    block_diag_2d = torch.block_diag(*block2d_list).unsqueeze_(0).unsqueeze_(0)
    a_blocks = a * block_diag_2d

    diff_a = a - a_blocks
    diff_norm = torch.linalg.matrix_norm(diff_a, ord=ord)  # By default: dim = (-2, -1)
    exact_norm = torch.linalg.matrix_norm(a, ord=ord)
    spectral_error_ratio = diff_norm / exact_norm
    max_spectral_error_ratio = spectral_error_ratio.max().item()

    print(f"seq_len: {seq_len:<8}, block_size: {config.block_size}, dim: {dim:<8}, ord: {ord}")
    print(f"a[0, 0, 0:2, 0:{3*config.block_size}] = \n{a[0, 0, 0:2, 0:3*config.block_size]}")
    print(f"a_blocks[0, 0, 0:2, 0:{3*config.block_size}] = \n{a_blocks[0, 0, 0:2, 0:3*config.block_size]}")
    print(f"For maxtrix a {a.shape}: max_spectral_error_ratio is {max_spectral_error_ratio:.5f}")
    print("End of test_get_tensors")


TEST_HYPER_ATTN_CONFIGS = [
    HyperAttentionConfig(input_dim=64, lsh_num_projs=7, block_size=1024, sample_size=1024, min_seq_len=2048, impl='xformers'),
    # HyperAttentionConfig(input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=2048, impl='cuda'),
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
    test_get_tensors()
    # test_spectral_error(TEST_HYPER_ATTN_CONFIGS[0], *(TEST_CASES[0]))
