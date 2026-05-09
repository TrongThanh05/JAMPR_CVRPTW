"""JAMPR Decoder: MHA + clipped attention scoring.

Implements formula §8 from docs/math_formulas.md.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from src.models.attention import MultiHeadAttention


class JAMPRDecoder(nn.Module):
    """Decoder that produces action logits over the joint action space.

    Step 1: h = MHA(context, M).squeeze(1)
    Step 2: scores = clip_value * tanh(q^T · k / sqrt(d_M))
    Step 3: scores[mask] = -inf
    Step 4: return scores (caller applies softmax)

    Args:
        d_context: Context vector dimension (d_C = 640).
        d_M: Affinity embedding dimension.
        d_hidden: Hidden dimension for MHA (default 256).
        clip_value: Clipping value for tanh (default 10.0).
    """

    def __init__(self, d_context: int, d_M: int, d_hidden: int = 256, clip_value: float = 10.0):
        super().__init__()
        self.d_M = d_M
        self.clip_value = clip_value

        # Project context to d_M for MHA compatibility
        self.context_proj = nn.Linear(d_context, d_M)

        # MHA: attend context over M
        self.mha = MultiHeadAttention(d_M, d_M, n_heads=8)

        # Final attention scoring
        self.W_q = nn.Linear(d_M, d_M, bias=False)
        self.W_k = nn.Linear(d_M, d_M, bias=False)

    def forward(self, context: Tensor, M: Tensor, mask: Tensor) -> Tensor:
        """Compute action logits.

        Args:
            context: (B, d_context) — comprehensive context vector.
            M:       (B, m_con*(N+1), d_M) — joint action embeddings.
            mask:    (B, m_con*(N+1)) bool — True = infeasible.

        Returns:
            Logits: (B, m_con*(N+1)) — raw scores, NO softmax.
        """
        B, seq_len, _ = M.shape

        # Project context to d_M
        ctx = self.context_proj(context)  # (B, d_M)

        # Step 1: MHA(context, M)
        h = self.mha(ctx, M, mask)  # (B, d_M)

        # Step 2: Clipped attention scores
        q = self.W_q(h)       # (B, d_M)
        k = self.W_k(M)       # (B, seq_len, d_M)
        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)).squeeze(1)  # (B, seq_len)
        scores = scores / math.sqrt(self.d_M)
        scores = self.clip_value * torch.tanh(scores)

        # Step 3: Mask infeasible actions
        scores = scores.masked_fill(mask, float('-inf'))

        return scores
