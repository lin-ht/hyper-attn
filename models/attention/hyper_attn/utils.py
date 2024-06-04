import math

import torch

try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError:
    flash_attn_func_cuda = None

from attention.flash_attn2.flash_attn_triton_for_hyper import flash_attn_func
from attention.flash_attn2.flash_attn_xformers import (
    flash_attn_func as flash_attn_func_xformers,
)


def indexing(x, indices, chunk_size=-1):
    """
    inputs:
        - x: 4d-tensor with shape [b, h, n, d]
        - indices: 3d-tensor with shape [b, h, s] where each entry should be in [0, n-1]
    output:
        - out: 4d-tensor with shape [b, h, s, d] where out[i,j] = x[i,j][indices[i,j],:]
               out will be padded with 0 to make the second dimension multiples of chunk_size.

    A naive implementation:
        out = torch.zeros(b, h, s, d)
        for i in range(b):
            for j in range(h):
                out[i,j] = x[i,j][idx[i,j],:]
        return out
    """
    if chunk_size < 0 or (chunk_size > 0 and x.shape[-2] % chunk_size == 0):
        # gather along dim=2: out[i,j,s,t] = x[i,j,idx[i,j,s,t],t]
        # which requires idx having the same dimension as x.
        return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
    else:
        x = x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        new_n = math.ceil(x.shape[2] / chunk_size) * chunk_size
        if new_n <= 0 or new_n - x.shape[2] <= 0:
            import pdb

            pdb.set_trace()
        return torch.nn.functional.pad(
            x, (0, 0, 0, new_n - x.shape[2]), mode="constant", value=0.0
        )


def add_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
        = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
        - lse
        = log(exp(lse1) + exp(lse2))
        = log(exp(lse1) * (1 + exp(lse2 - lse1)))
        = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)
    """
    c = (1 / (1 + (lse2 - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1 - c) * attn2
    lse = lse1 - (c + torch.finfo(lse1.dtype).eps).log()
    return attn, lse


def subtract_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) - attn2 * exp(lse2)) / (exp(lse1) - exp(lse2))
        = (attn1 - attn2 * exp(lse2 - lse1)) / (1 - exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 - exp(lse2-lse1)),
        - lse
        = log(exp(lse1) - exp(lse2))
        = log(exp(lse1) * (1 - exp(lse2 - lse1)))
        = lse1 + log(1 - exp(lse2 - lse1)) = lse1 - log(c)
    """
    c = (1 / (1 - (lse2 - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1 - c) * attn2
    lse = lse1 - (c + torch.finfo(lse1.dtype).eps).log()
    return attn, lse


def add_scaled_self_attentions(attn1, lse1, s1, attn2, lse2, s2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
        - s1, s2: tensors of relative scales (>0) with shape [b, h, n] or [b, h]
    output:
        - same as add_self_attentions(attn1, lse1+log(s1), attn2, lse2+log(s2))
    """
    log_s = torch.log(s2 / s1)
    c = (1 / (1 + (lse2 + log_s - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1 - c) * attn2
    s = s1 / (c + torch.finfo(lse1.dtype).eps)
    return attn, lse1, s


def exact_attention(query, key, value, softmax_scale, causal=False, bias=None):
    # input query.shape = [batch_size, head_size, n_query, dim]
    # input   key.shape = [batch_size, head_size, n_key,   dim]
    if query.dtype not in [torch.bfloat16, torch.float16]:
        # qk.shape = [batch_size, head_size, n_query, n_key]
        qk = query @ key.transpose(-1, -2) * softmax_scale
        if causal:
            # wipe out the uper triangular part of qk by adding -inf
            qk += (
                (
                    torch.ones(query.shape[2], key.shape[2], device=query.device)
                    * torch.finfo(query.dtype).min
                )
                .triu(1)
                .reshape(1, 1, query.shape[2], key.shape[2])
            )
        # out.shape = [batch_size, head_size, n_query, dim]
        out = qk.softmax(dim=-1) @ value
        # lse.shape = [batch_size, head_size, n_query, 1]
        lse = torch.logsumexp(qk, dim=-1, keepdim=True)
        return out, lse

    out, lse = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        bias,
        causal,
        softmax_scale,
    )
    out = out.transpose(1, 2)

    lse = lse.detach()
    if lse.shape[2] != out.shape[2]:
        lse = lse[:, :, : out.shape[2]]
    lse = lse.unsqueeze(-1)
    return out, lse


def exact_attention_cuda(query, key, value, softmax_scale, causal=False, bias=None):
    if flash_attn_func_cuda is None:
        raise ImportError(
            "Please install flash_attn (pip install flash-attn --no-build-isolation)"
        )
    out, lse, _ = flash_attn_func_cuda(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    out = out.transpose(1, 2)
    lse = lse.unsqueeze(-1)
    return out, lse


def exact_attention_xformers(query, key, value, softmax_scale, causal=False, bias=None):
    # HyperAttn input format: BHMK
    # In contrast: xformers dim format: BMK, BMHK, BMGHK
    # B = batch size
    # M = sequence length
    # G = heads groups (in case of multiquery/grouped query attention)
    # H = heads per group
    # K = embedding size per head
    shape_size = len(query.shape)
    if shape_size > 3:
        q_ = query.transpose(1, -2)
        k_ = key.transpose(1, -2)
        v_ = value.transpose(1, -2)
    else:
        q_ = query.unsqueeze(1)
        k_ = key.unsqueeze(1)
        v_ = value.unsqueeze(1)

    out, lse = flash_attn_func_xformers(q_, k_, v_, bias, causal, softmax_scale)

    if shape_size > 3:
        out = out.transpose(1, -2)
        lse = lse.unsqueeze(-1)
    else:
        out = out.squeeze(1)
        lse = lse.squeeze(1).unsqueeze(-1)

    seq_len = query.shape[-2]
    lse = lse[..., :seq_len, :].to(out.dtype)
    return out, lse
