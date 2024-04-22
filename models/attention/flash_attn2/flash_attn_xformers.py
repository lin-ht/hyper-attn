import torch
from xformers.ops import (
    AttentionOpDispatch,
    memory_efficient_attention_backward,
    memory_efficient_attention_forward_requires_grad,
    LowerTriangularMask,
)


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, causal, bias, scale=None, p=0.0):
        """
        Input tensors query, key, value must be in format
            [Batch, n_seq, n_heads, dim] or [Batch, n_seq, dim],
            where n_seq (n_query or n_key) the sequence length,
            n_heads the number of heads, and dim the embeding size.
        Input attention bias tensor shape can be (Batch or 1, n_query, n_key)
        Input p is the dropout probability.
        """
        # Fixme: support both causal and bias.
        assert(causal == False or bias is None), f"(causal={causal}) for non-None bias is not supported."
        attn_mask = LowerTriangularMask if causal else bias

        # The most common values in an attention bias are zero and negative infinity.
        # So that bias works as a mask to make some queries only attend to some keys.
        # Zero means that the corresponding query can attend to the corresponding key.
        mem_eff_attn_op = AttentionOpDispatch.from_arguments(
            query=query, key=key, value=value, attn_bias=attn_mask, scale=scale, p=p
        ).op

        out, lse = memory_efficient_attention_forward_requires_grad(
            query, key, value, op=mem_eff_attn_op[0], attn_bias=attn_mask, scale=scale, p=p
        )
        # out, lse = memory_efficient_attention_forward_requires_grad(
        #     query, key, value, op=None, attn_bias=bias, scale=scale, p=p
        # )

        ctx.save_for_backward(query, key, value, out, lse, bias)
        ctx.attn_mask = attn_mask
        ctx.scale = scale
        return out, lse

    @staticmethod
    def backward(ctx, do, dlse_use_needed=None):
        query, key, value, out, lse, bias = ctx.saved_tensors
        attn_mask = ctx.attn_mask
        scale = ctx.scale
        # dropout is not supported on the non-autograd API
        dq, dk, dv = memory_efficient_attention_backward(
            do, out, lse, query, key, value, attn_bias=attn_mask, p=0.0, scale=scale
        )
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply
