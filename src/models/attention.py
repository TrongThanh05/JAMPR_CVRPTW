"""Attention modules: SingleHeadAttention and slice-based MultiHeadAttention.

Implements formulas §1 and §2 from docs/math_formulas.md.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SingleHeadAttention(nn.Module):
    """Single-Head Attention (SHA) as defined in the JAMPR paper.

    Formula §1:
        attn = softmax( (1/√d_key) * query^T * W_q^T * W_k * keys )
        output = Σ_j attn_j * W_v * Z_j

    Args:
        d_in: Input feature dimension.
        d_key: Key/query projection dimension.
        d_out: Output dimension (value projection).
    """

    def __init__(self, d_in: int, d_key: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_key = d_key
        self.d_out = d_out

        self.W_q = nn.Linear(d_in, d_key, bias=False)
        self.W_k = nn.Linear(d_in, d_key, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, query: Tensor, keys: Tensor, mask: Tensor = None) -> Tensor:
        """Compute single-head attention.

        Args:
            query: (B, d_in) or (B, 1, d_in) — single query vector per batch.
            keys:  (B, S, d_in) — sequence of key/value vectors.
            mask:  (B, S) bool — True means masked (infeasible), set to -inf.

        Returns:
            Output tensor, shape (B, d_out).
        """
        # Handle query dimensionality
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, d_in)

        B, S, _ = keys.shape
        assert query.shape[-1] == self.d_in, \
            f"Query dim {query.shape[-1]} != d_in {self.d_in}"

        # Project
        q = self.W_q(query)   # (B, 1, d_key)
        k = self.W_k(keys)    # (B, S, d_key)
        v = self.W_v(keys)    # (B, S, d_out)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_key)  # (B, 1, S)

        if mask is not None:
            # mask: (B, S) -> (B, 1, S)
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, S)

        # Handle all-masked case (softmax produces NaN when all -inf)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        output = torch.matmul(attn_weights, v)  # (B, 1, d_out)
        output = output.squeeze(1)  # (B, d_out)

        assert output.shape == (B, self.d_out), \
            f"SHA output shape {output.shape} != expected ({B}, {self.d_out})"
        return output


class MultiHeadAttention(nn.Module):
    """Slice-based Multi-Head Attention (MHA) as defined in JAMPR paper.

    Formula §2: Each head h receives slice [h*Δ : (h+1)*Δ] of the input.
    Output = Σ_h W_head_h * SHA(z_slice(h), Z_slice(h))

    This differs from standard PyTorch MHA where all heads share the full input.

    Args:
        d_in: Input feature dimension (must be divisible by n_heads).
        d_out: Output feature dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int):
        super().__init__()
        assert d_in % n_heads == 0, \
            f"d_in ({d_in}) must be divisible by n_heads ({n_heads})"

        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_slice = d_in // n_heads  # Δh

        # Each head: SHA on its slice, then projected by W_head
        d_key = self.d_slice  # use slice size as key dim
        self.heads = nn.ModuleList([
            SingleHeadAttention(self.d_slice, d_key, d_out)
            for _ in range(n_heads)
        ])
        # Projection weights for each head output
        self.W_heads = nn.ModuleList([
            nn.Linear(d_out, d_out, bias=False)
            for _ in range(n_heads)
        ])

    def forward(self, query: Tensor, context: Tensor, mask: Tensor = None) -> Tensor:
        """Compute slice-based multi-head attention.

        Args:
            query:   (B, d_in) or (B, 1, d_in) — query vector.
            context: (B, S, d_in) — key/value sequence.
            mask:    (B, S) bool — True = infeasible.

        Returns:
            Output tensor, shape (B, d_out).
        """
        if query.dim() == 3:
            query = query.squeeze(1)  # (B, d_in)

        B = query.shape[0]
        assert query.shape[-1] == self.d_in, \
            f"Query dim {query.shape[-1]} != d_in {self.d_in}"

        output = torch.zeros(B, self.d_out, device=query.device, dtype=query.dtype)

        for h in range(self.n_heads):
            start = h * self.d_slice
            end = (h + 1) * self.d_slice

            q_slice = query[:, start:end]           # (B, d_slice)
            ctx_slice = context[:, :, start:end]    # (B, S, d_slice)

            head_out = self.heads[h](q_slice, ctx_slice, mask)  # (B, d_out)
            output = output + self.W_heads[h](head_out)         # accumulate

        assert output.shape == (B, self.d_out), \
            f"MHA output shape {output.shape} != expected ({B}, {self.d_out})"
        return output


class MultiHeadAttentionSeq(nn.Module):
    """MHA variant that handles sequence-to-sequence attention (for SA blocks).

    Applies MHA independently to each position in the query sequence,
    operating on sliced inputs per head.

    Args:
        d_in: Feature dimension.
        d_out: Output dimension.
        n_heads: Number of heads.
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int):
        super().__init__()
        assert d_in % n_heads == 0
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_slice = d_in // n_heads
        d_key = self.d_slice

        # Per-head parameters: W_q, W_k, W_v
        self.W_q = nn.ModuleList([nn.Linear(self.d_slice, d_key, bias=False) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(self.d_slice, d_key, bias=False) for _ in range(n_heads)])
        self.W_v = nn.ModuleList([nn.Linear(self.d_slice, d_out, bias=False) for _ in range(n_heads)])
        self.W_heads = nn.ModuleList([nn.Linear(d_out, d_out, bias=False) for _ in range(n_heads)])

    def forward(self, query_seq: Tensor, context_seq: Tensor, mask: Tensor = None) -> Tensor:
        """Sequence-to-sequence MHA.

        Args:
            query_seq:   (B, N, d_in) — query sequence.
            context_seq: (B, S, d_in) — key/value sequence.
            mask: Optional, not typically used in self-attention encoder.

        Returns:
            (B, N, d_out)
        """
        B, N, _ = query_seq.shape
        S = context_seq.shape[1]
        output = torch.zeros(B, N, self.d_out, device=query_seq.device, dtype=query_seq.dtype)

        for h in range(self.n_heads):
            start = h * self.d_slice
            end = (h + 1) * self.d_slice

            q_slice = query_seq[:, :, start:end]    # (B, N, d_slice)
            ctx_slice = context_seq[:, :, start:end]  # (B, S, d_slice)

            q = self.W_q[h](q_slice)  # (B, N, d_key)
            k = self.W_k[h](ctx_slice)  # (B, S, d_key)
            v = self.W_v[h](ctx_slice)  # (B, S, d_out)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_slice)  # (B, N, S)
            attn = F.softmax(scores, dim=-1)  # (B, N, S)
            head_out = torch.matmul(attn, v)  # (B, N, d_out)
            output = output + self.W_heads[h](head_out)

        return output
