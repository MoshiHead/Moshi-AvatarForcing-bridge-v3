# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings

__all__ = [
    'flash_attention',
    'attention',
]

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    fa_version=None,
):
    """
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    """
    b, lq, nq, c1 = q.shape
    _, lk, nk, c2 = v.shape

    q = q.transpose(1, 2).to(dtype)  # [B, Nq, Lq, C1]
    k = k.transpose(1, 2).to(dtype)  # [B, Nk, Lk, C1]
    v = v.transpose(1, 2).to(dtype)  # [B, Nk, Lk, C2]

    if q_scale is not None:
        q = q * q_scale

    attn_mask = None
    if k_lens is not None or q_lens is not None:
        attn_mask = torch.ones((b, 1, lq, lk), dtype=torch.bool, device=q.device)
        
        if k_lens is not None:
            for i, length in enumerate(k_lens):
                attn_mask[i, :, :, length:] = False
                
        if q_lens is not None:
            for i, length in enumerate(q_lens):
                attn_mask[i, :, length:, :] = False
                
    if causal and attn_mask is not None:
        # Combine causal mask with padding mask
        causal_mask = torch.tril(torch.ones((lq, lk), dtype=torch.bool, device=q.device))
        attn_mask = attn_mask & causal_mask.view(1, 1, lq, lk)
        causal = False  # Let SDPA use our combined mask

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=attn_mask, 
        is_causal=causal, 
        dropout_p=dropout_p,
        scale=softmax_scale
    )

    out = out.transpose(1, 2).contiguous()  # [B, Lq, Nq, C1]
    return out

# Alias flash_attention so existing code works without modification
flash_attention = attention
