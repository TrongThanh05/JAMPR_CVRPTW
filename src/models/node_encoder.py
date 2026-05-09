"""Node Encoder: linear projection + SA blocks.

Implements formula §3-§4 from docs/math_formulas.md.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.attention import MultiHeadAttentionSeq


class SABlock(nn.Module):
    """Self-Attention Block: MHAres → BNres → FFres → BNres.

    Formula §3:
        MHAres(z_i, Z) = z_i + MHA(z_i, Z)
        FFres(z_i)     = z_i + FF(z_i)
        FF(z_i)        = max(0, W·z_i + b)   (ReLU)
        BN             = BatchNorm1d

    Args:
        d_node: Node embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
    """

    def __init__(self, d_node: int, n_heads: int, d_ff: int):
        super().__init__()
        self.mha = MultiHeadAttentionSeq(d_node, d_node, n_heads)
        self.bn1 = nn.BatchNorm1d(d_node)
        self.ff = nn.Sequential(
            nn.Linear(d_node, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_node),
        )
        self.bn2 = nn.BatchNorm1d(d_node)

    def forward(self, z: Tensor, Z: Tensor) -> Tensor:
        """Apply SA block.

        Args:
            z: Query sequence, (B, N, d_node).
            Z: Key/value sequence, (B, N, d_node). Usually same as z for self-attention.

        Returns:
            Updated embeddings, (B, N, d_node).
        """
        B, N, d = z.shape

        # MHAres + BN
        h = z + self.mha(z, Z)                  # (B, N, d)
        h = self.bn1(h.reshape(B * N, d)).reshape(B, N, d)

        # FFres + BN
        h = h + self.ff(h)                       # (B, N, d)
        h = self.bn2(h.reshape(B * N, d)).reshape(B, N, d)

        return h


class NodeEncoder(nn.Module):
    """Encodes node features into embeddings via linear projection + SA blocks.

    Formula §4:
        z0_i = W_in · x_i + b_in    (linear projection)
        ω_node_i = SA_3(SA_2(SA_1(z0)))

    Args:
        input_dim: Dimension of raw node features (typically 3: x, y, demand).
        d_node: Node embedding dimension.
        n_layers: Number of SA blocks.
        n_heads: Number of attention heads per SA block.
    """

    def __init__(self, input_dim: int, d_node: int, n_layers: int, n_heads: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_node = d_node

        # Linear projection: (input_dim) -> (d_node)
        self.linear_proj = nn.Linear(input_dim, d_node)

        # SA blocks
        self.sa_blocks = nn.ModuleList([
            SABlock(d_node, n_heads, d_ff=d_node)
            for _ in range(n_layers)
        ])

    def forward(self, features: Tensor) -> Tensor:
        """Encode node features.

        Args:
            features: (B, N+1, input_dim) — raw node features.

        Returns:
            Node embeddings, (B, N+1, d_node).
        """
        B, N, _ = features.shape
        assert features.shape[-1] == self.input_dim, \
            f"Input dim {features.shape[-1]} != expected {self.input_dim}"

        # Step 1: Linear projection
        z = self.linear_proj(features)  # (B, N, d_node)

        # Step 2: Apply SA blocks (self-attention: query = keys = z)
        for sa_block in self.sa_blocks:
            z = sa_block(z, z)

        assert z.shape == (B, N, self.d_node), \
            f"NodeEncoder output {z.shape} != expected ({B}, {N}, {self.d_node})"
        return z
