import torch
import os
import triton
import triton.language as tl
from triton.runtime import driver as drv

os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '10000'

def prepare_data(bits = 2, batch_size = 2, seq_len = 1024, hn = 32, dh = 256, dtype=torch.float16, device="cuda"):
    # b,s,hn,dh
    shp = (batch_size, seq_len, hn, dh)
    value_states = torch.randn(*shp, dtype=dtype, device=device)

    value_states_v = value_states.view(batch_size, -1)

    x_min = value_states_v.min(dim=-1).values.unsqueeze_(-1)
    x_max = value_states_v.max(dim=-1).values.unsqueeze_(-1)

    quant_levels = 2**bits - 1
    scale = quant_levels / (x_max - x_min)
    zero_point = -x_min * scale

    x_ind = torch.round(torch.clamp(value_states_v * scale + zero_point, 0, quant_levels)).to(torch.int32)
    x_lut = x_min + torch.arange(0, 2**bits, device=device) / scale
    
    value_states_deq = (x_ind - zero_point) / scale
    value_states_deq = value_states_deq.reshape(*shp)

    x_ind = x_ind.reshape(*shp)
    x_lut = x_lut.reshape(batch_size, -1)

    return value_states, value_states_deq, x_ind, x_lut, x_min, x_max, scale, zero_point


@triton.jit
def op_or(x, y):
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
    rst = tl.reduce(data, 1, op_or)
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
    return y.reshape(shape_y).squeeze(-1)


@triton.jit
def dequantize_kernel(
    x_ptr,  # *Pointer* to the contiguous input vector.
    lut_ptr, # *Pointer* to the coutiguous lut vector.
    y_ptr,  # *Pointer* to the coutiguous output vector with type uint8.
    x_stride_b,  # Size of the vector.
    lut_stride_b,
    y_stride_b,  # Size of the vector.
    bits: tl.constexpr, # Number of bits to represent each element.
    compression_scale: tl.constexpr, # its value = 8 // bits
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    bid = tl.program_id(axis=0)  # We use a 2D launch grid.
    pid = tl.program_id(axis=1)
    offsets_x_b = bid * x_stride_b
    offsets_x =  pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets_x_b +  offsets_x, mask=offsets_x < x_stride_b)
    u = tl.arange(0, compression_scale)

    BLOCK_SIZE_Y:tl.constexpr = (BLOCK_SIZE * compression_scale)
    indices = tl.cast((x[:, None] >> ((compression_scale - u[None, :] - 1) * bits)) & ((1 << bits) - 1), tl.uint8)
    indices = tl.reshape(indices, [BLOCK_SIZE_Y])
    
    y = tl.load(lut_ptr + bid * lut_stride_b + indices) #tl.float16

    offsets_y = pid * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    tl.store(y_ptr + bid * y_stride_b + offsets_y, y, mask=offsets_y < y_stride_b)


def dequantize(x: torch.Tensor, lut: torch.Tensor, bits = 2, dtype=torch.float16) -> torch.Tensor:
    assert bits in [1, 2, 4], f"Unsupported bits {bits}"
    n_batches = x.shape[0]
    compression_scale = 8 // bits
    n_elements = x.numel()
    n_elements_per_batch = n_elements // n_batches
    # We need to preallocate the output.
    x = x.to(device="cuda")
    y = torch.zeros(n_batches, n_elements_per_batch*compression_scale, device=x.device, dtype=dtype)
    lut = lut.to(device=y.device)

    grid = lambda meta: (n_batches, triton.cdiv(n_elements_per_batch, meta['BLOCK_SIZE']))
    dequantize_kernel[grid](x, lut, y, x.stride(0), lut.stride(0), y.stride(0), bits, compression_scale, BLOCK_SIZE=1024)

    shape_y = list(x.shape)
    shape_y[-1] *= compression_scale
    return y.reshape(shape_y)


def test_quantization(bits = 2, batch_size = 2, seq_len = 1024, hn = 32, dh = 256):
    dtype = torch.float16
    RTOL = 1e-4 if dtype == torch.float else 1e-2
    ATOL = 1e-4 if dtype == torch.float else 1e-2

    value_states, value_states_deq, x_ind, x_lut, x_min, x_max, scale, zero_point = prepare_data(bits, batch_size, seq_len, hn, dh, dtype=dtype, device="cuda")

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
    torch.testing.assert_close(value_states_deq, x_val_decompressed.view(value_states_deq.shape), rtol=RTOL, atol=ATOL)

    encoded_tensor = x_ind_compressed_v
    # Decoding in torch
    decoded_indices = torch.zeros_like(x_ind_v, dtype=torch.uint8, device=x_ind_compressed.device)
    for bit_pos in range(compression_scale):
        decoded_indices[..., bit_pos] = (encoded_tensor >> ((compression_scale - 1 - bit_pos) * bits)) & (2**bits - 1)
    decoded_indices = decoded_indices.view(batch_size, -1)

    decoded_tensor = torch.gather(x_lut, dim=-1, index=decoded_indices.long())
    decoded_tensor = decoded_tensor.view(value_states_deq.shape)

    torch.testing.assert_close(value_states_deq.to(device=decoded_tensor.device), decoded_tensor, rtol=RTOL, atol=ATOL)
    print("Passed!")


def test_quantization2(bits, batch_size, seq_len, hn, dh):
    dtype = torch.float16
    device = "cuda"

    RTOL = 1e-4 if dtype == torch.float else 1e-2
    ATOL = 1e-4 if dtype == torch.float else 1e-2

    k_org, k_deq, k_ind, k_lut, k_min, k_max, _, _ = prepare_data(bits, batch_size, seq_len, hn, dh, dtype=dtype, device=device)

    compression_scale = 8 // bits
    k_ind_v = k_ind.view(batch_size, -1)
    assert k_ind_v.shape[-1] % compression_scale == 0, f"Unsupported shape {k_ind_v.shape}"
    k_ind_v = k_ind.view(batch_size, -1, compression_scale)
    k_encoded = quantize(k_ind_v, bits) # k_encoded dtype = uint8

     # Encoding in torch
    k_encoded_torch = torch.zeros(k_ind_v.shape[0], k_ind_v.shape[1], dtype=torch.uint8, device=k_ind_v.device)
    for bit_pos in range(compression_scale):
        k_encoded_torch |= (k_ind_v[..., bit_pos].squeeze(dim=-1)) << ((compression_scale - 1 - bit_pos) * bits)
    print(f"{k_encoded_torch.shape=}, {k_encoded_torch[0, 0:4]=}")
    print(f"{k_encoded.shape=},             {k_encoded[0, 0:4]=}")
    torch.testing.assert_close(k_encoded, k_encoded_torch)

    # Decoding in torch
    k_ind_decoded_torch = torch.zeros_like(k_ind_v, dtype=torch.uint8, device=k_encoded.device)
    for bit_pos in range(compression_scale):
        k_ind_decoded_torch[..., bit_pos] = (k_encoded >> ((compression_scale - 1 - bit_pos) * bits)) & (2**bits - 1)

    torch.testing.assert_close(k_ind_v, k_ind_decoded_torch.to(k_ind_v.dtype))
    
    k_decoded = dequantize(k_encoded, k_lut, bits).reshape(k_deq.shape)
    torch.testing.assert_close(k_deq.to(device=k_decoded.device), k_decoded, rtol=RTOL, atol=ATOL)
    print("Passed!")


if __name__ == "__main__":
    bits = 2
    batch_size = 128
    seq_len = 3072 
    hn = 48
    dh = 128

    test_quantization2(bits, batch_size, seq_len, hn, dh)

