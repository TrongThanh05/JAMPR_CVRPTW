"""Affinity Network g_a: computes joint vehicle-node action embeddings.

Implements formula §7 from docs/math_formulas.md.
"""

import torch
import torch.nn as nn
from torch import Tensor


class AffinityNetwork(nn.Module):
    """Affinity network g_a combining vehicle and node embeddings.

    g_a(ω_v, ω_n) = W1·ω_n + W2·ω_v + W3·[ω_v ⊙ ω_n ; dot(ω_v, ω_n)]

    Args:
        d_node: Node embedding dimension.
        d_vehicle: Vehicle embedding dimension.
        d_M: Output affinity dimension.
    """

    def __init__(self, d_node: int, d_vehicle: int, d_M: int):
        super().__init__()
        self.d_node = d_node
        self.d_vehicle = d_vehicle
        self.d_M = d_M
        self.W1 = nn.Linear(d_node, d_M, bias=False)
        self.W2 = nn.Linear(d_vehicle, d_M, bias=False)
        self.W3 = nn.Linear(d_vehicle + 1, d_M, bias=False)

    def forward(self, vehicle_emb: Tensor, node_emb: Tensor) -> Tensor:
        """Compute joint affinity embeddings.

        Args:
            vehicle_emb: (B, K_act, d_vehicle)
            node_emb:    (B, N+1, d_node)

        Returns:
            M: (B, K_act*(N+1), d_M)
        """
        B, K_act, d_v = vehicle_emb.shape
        N1 = node_emb.shape[1]

        v_exp = vehicle_emb.unsqueeze(2).expand(B, K_act, N1, d_v)
        n_exp = node_emb.unsqueeze(1).expand(B, K_act, N1, self.d_node)

        term1 = self.W1(n_exp)
        term2 = self.W2(v_exp)

        hadamard = v_exp * n_exp
        dot_prod = (v_exp * n_exp).sum(dim=-1, keepdim=True)
        combined = torch.cat([hadamard, dot_prod], dim=-1)
        term3 = self.W3(combined)

        M = (term1 + term2 + term3).reshape(B, K_act * N1, self.d_M)
        return M
