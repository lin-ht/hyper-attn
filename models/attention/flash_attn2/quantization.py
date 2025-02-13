import torch
import os
import triton
import triton.language as tl
from triton.runtime import driver as drv

os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '10000'

def prepare_data(bits = 2, batch_size = 2, seq_len = 1024, hn = 32, dh = 256, device="cuda"):
    value_states = torch.randn(batch_size, hn, seq_len, dh, device=device)
    value_states_v = value_states.view(batch_size, -1)

    x_min = value_states_v.min(dim=-1).values.unsqueeze_(-1)
    x_max = value_states_v.max(dim=-1).values.unsqueeze_(-1)

    quant_levels = 2**bits - 1
    scale = quant_levels / (x_max - x_min)
    zero_point = -x_min * scale

    x_ind = torch.round(torch.clamp(value_states_v * scale + zero_point, 0, quant_levels)).to(torch.int32)
    x_lut = x_min + torch.arange(0, 2**bits, device=device) / scale
    
    value_states_deq = (x_ind - zero_point) / scale
    value_states_deq = value_states_deq.view(batch_size, hn, seq_len, dh)

    x_ind = x_ind.reshape(batch_size, hn, seq_len, dh)
    x_lut = x_lut.reshape(batch_size, -1)

    return value_states, value_states_deq, x_ind, x_lut, x_min, x_max, scale, zero_point


@triton.jit
def reduction_or(x, y):
    return x | y


@triton.jit
def quantize_kernel(
    x_ptr,  # *Pointer* to the contiguous input vector with type .
    y_ptr,  # *Pointer* to the coutiguous output vector with type uint8.
    n_elements,  # Size of the vector.
    bits: tl.constexpr, # Number of bits to represent each element.
    compression_scale: tl.constexpr, # its value = 8 // bits
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start_y = pid * BLOCK_SIZE
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE)

    block_start_x = block_start_y * compression_scale
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE * compression_scale)
    x = tl.load(x_ptr + offsets_x, mask=offsets_x < n_elements)
    x = tl.reshape(x, [BLOCK_SIZE, compression_scale])
    data_bits = ((1 << bits) - 1)
    data = tl.cast((x & data_bits) << ((compression_scale - 1 - tl.arange(0, compression_scale)) * bits) , tl.uint8)
    rst = tl.reduce(data, 1, reduction_or)
    # Write compression result back to DRAM.
    mask_y = offsets_y < tl.cdiv(n_elements, compression_scale)
    tl.store(y_ptr + offsets_y, rst, mask=mask_y)


def quantize(x: torch.Tensor, bits = 2) -> torch.Tensor:
    assert bits in [2, 4], f"Unsupported bits {bits}"

    compression_scale = 8 // bits
    n_elements = x.numel()
    # We need to preallocate the output.
    assert n_elements % compression_scale == 0, f"Unsupported shape {x.shape}"
    x = x.to(device="cuda")
    y = torch.zeros(int(n_elements // compression_scale), device=x.device, dtype=torch.uint8)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    quantize_kernel[grid](x, y, n_elements, bits, compression_scale, BLOCK_SIZE=1024)

    shape_y = list(x.shape)
    shape_y[-1] //= compression_scale
    return y.reshape(shape_y)


@triton.jit
def dequantize_kernel(
    x_ptr,  # *Pointer* to the contiguous input vector.
    lut_ptr, # *Pointer* to the coutiguous lut vector.
    y_ptr,  # *Pointer* to the coutiguous output vector with type uint8.
    n_batches,  # Number of batches.
    n_elements_per_batch,  # Size of the vector.
    bits: tl.constexpr, # Number of bits to represent each element.
    compression_scale: tl.constexpr, # its value = 8 // bits
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    bid = tl.program_id(axis=0)  # We use a 2D launch grid.
    pid = tl.program_id(axis=1)
    n_elements = n_elements_per_batch * n_batches

    block_start_x = bid * n_elements_per_batch + pid * BLOCK_SIZE
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets_x, mask=offsets_x < n_elements)
    u = tl.arange(0, compression_scale)

    indices = tl.cast((x[:, None] >> ((compression_scale - u[None, :] - 1) * bits)) & ((1 << bits) - 1), tl.uint8)
    indices = tl.reshape(indices, [BLOCK_SIZE*compression_scale])
    y = tl.cast(tl.load(lut_ptr + bid * compression_scale + indices), tl.float16)

    offsets_y = block_start_x * compression_scale + tl.arange(0, BLOCK_SIZE * compression_scale)
    n_elements_output = n_elements * compression_scale
    tl.store(y_ptr + offsets_y, y, mask=offsets_y < n_elements_output)


def dequantize(x: torch.Tensor, lut: torch.Tensor, bits = 2) -> torch.Tensor:
    assert bits in [2, 4], f"Unsupported bits {bits}"
    n_batches = x.shape[0]
    compression_scale = 8 // bits
    n_elements = x.numel()
    n_elements_per_batch = n_elements // n_batches
    # We need to preallocate the output.
    x = x.to(device="cuda")
    y = torch.zeros(n_elements * compression_scale, device=x.device, dtype=torch.float16)
    lut = lut.to(device=y.device)

    grid = lambda meta: (n_batches, triton.cdiv(n_elements_per_batch, meta['BLOCK_SIZE']))
    dequantize_kernel[grid](x, lut, y, n_batches, n_elements_per_batch, bits, compression_scale, BLOCK_SIZE=1024)

    shape_y = list(x.shape)
    shape_y[-1] *= compression_scale
    return y.reshape(shape_y)


def test_quantization(bits = 2, batch_size = 2, seq_len = 1024, hn = 32, dh = 256):
    value_states, value_states_deq, x_ind, x_lut, x_min, x_max, scale, zero_point = prepare_data(bits, batch_size, seq_len, hn, dh)

    compression_scale = 8 // bits
    x_ind_v = x_ind.view(batch_size, -1)
    assert x_ind_v.shape[-1] % compression_scale == 0, f"Unsupported shape {x_ind_v.shape}"
    x_ind_v = x_ind.view(batch_size, -1, compression_scale)

    # Encoding in torch
    encoded_tensor = torch.zeros(x_ind_v.shape[0], x_ind_v.shape[1], dtype=torch.uint8, device=x_ind_v.device)
    for bit_pos in range(compression_scale):
        encoded_tensor |= (x_ind_v[..., bit_pos].squeeze(dim=-1)) << ((compression_scale - 1 - bit_pos) * bits)

    print(f"{encoded_tensor.shape=}, {encoded_tensor[0, 0:4]=}")

    x_ind_compressed = quantize(x_ind_v, bits)
    x_ind_compressed_v = x_ind_compressed.view(batch_size, -1)
    print(f"{x_ind_compressed_v.shape=}, {x_ind_compressed_v[0, 0:4]=}")

    x_val_decompressed = dequantize(x_ind_compressed, x_lut, bits)
    assert torch.allclose(value_states_deq.to(device=x_val_decompressed.device).to(x_val_decompressed.dtype), x_val_decompressed.view(value_states_deq.shape))

    encoded_tensor = x_ind_compressed_v
    # Decoding in torch
    decoded_indices = torch.zeros_like(x_ind_v, dtype=torch.uint8, device=x_ind_compressed.device)
    for bit_pos in range(compression_scale):
        decoded_indices[..., bit_pos] = (encoded_tensor >> ((compression_scale - 1 - bit_pos) * bits)) & (2**bits - 1)
    decoded_indices = decoded_indices.view(batch_size, -1)

    decoded_tensor = torch.gather(x_lut, dim=-1, index=decoded_indices.long())
    decoded_tensor = decoded_tensor.view(batch_size, hn, seq_len, dh)

    assert torch.allclose(value_states_deq.to(device=decoded_tensor.device), decoded_tensor)
    print("Passed!")


if __name__ == "__main__":
    bits = 2
    batch_size = 2
    seq_len = 1024
    hn = 32
    dh = 128

    test_quantization(bits, batch_size, seq_len, hn, dh)

