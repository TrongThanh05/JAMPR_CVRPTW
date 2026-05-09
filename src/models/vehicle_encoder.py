"""Vehicle and Tour Encoders: g_v and g_s networks.

Implements formula §5 from docs/math_formulas.md.
"""

import torch
import torch.nn as nn
from torch import Tensor


class TourEncoder(nn.Module):
    """Tour encoder g_s: maps node embeddings to tour-level features.

    Maps each node embedding to d_vehicle//2, then averaged across tour nodes.

    Args:
        d_node: Input node embedding dimension.
        d_out: Output dimension (typically d_vehicle // 2).
        n_layers: Number of hidden layers (default 2).
    """

    def __init__(self, d_node: int, d_out: int, n_layers: int = 2):
        super().__init__()
        layers = []
        d_hidden = 64  # As specified in PROJECT_SPEC §2.2
        in_dim = d_node
        for i in range(n_layers):
            out_dim = d_hidden if i < n_layers - 1 else d_out
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.network = nn.Sequential(*layers)

    def forward(self, node_emb: Tensor) -> Tensor:
        """Encode a single node embedding.

        Args:
            node_emb: (*, d_node) — node embedding(s).

        Returns:
            Tour feature, (*, d_out).
        """
        return self.network(node_emb)


class VehicleEncoder(nn.Module):
    """Vehicle encoder g_v: maps vehicle features to vehicle-level embeddings.

    Vehicle features v_k = (k/K, return_cost, pos_x, pos_y, current_time,
                            remaining_load).
    All values normalized to [0, 1].

    Args:
        d_vehicle: Full vehicle embedding dimension.
        n_layers: Number of hidden layers (1 for CVRP, 3 for CVRP-TW).
    """

    def __init__(self, d_vehicle: int, n_layers: int):
        super().__init__()
        d_out = d_vehicle // 2
        input_dim = 6  # (k_norm, return_cost, pos_x, pos_y, time, remaining_load)

        layers = []
        in_dim = input_dim
        d_hidden = d_out
        for i in range(n_layers):
            out_dim = d_hidden if i < n_layers - 1 else d_out
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.network = nn.Sequential(*layers)

    def forward(self, v_k: Tensor) -> Tensor:
        """Encode vehicle features.

        Args:
            v_k: (*, 6) — vehicle feature vector.

        Returns:
            Vehicle embedding, (*, d_vehicle//2).
        """
        return self.network(v_k)


def encode_all_vehicles(state, node_emb: Tensor,
                        tour_enc: TourEncoder,
                        vehicle_enc: VehicleEncoder) -> Tensor:
    """Encode all vehicles combining g_v and g_s outputs.

    Formula §5:
        ω_vehicle_k = [g_v(v_k) ; mean_{i∈s_k}(g_s(ω_node_i))]

    Args:
        state: VRPTWState with tour info and vehicle features.
        node_emb: (B, N+1, d_node) — cached node embeddings.
        tour_enc: TourEncoder (g_s) module.
        vehicle_enc: VehicleEncoder (g_v) module.

    Returns:
        Vehicle embeddings, (B, K, d_vehicle).
    """
    B = node_emb.shape[0]
    d_node = node_emb.shape[2]
    K = state.tours.shape[1]
    d_half = vehicle_enc.network[-2].out_features  # d_vehicle // 2
    device = node_emb.device

    # Get vehicle features and encode with g_v (detach from state graph)
    v_features = state.get_vehicle_features().detach()  # (B, K, 5)
    gv_out = vehicle_enc(v_features)  # (B, K, d_vehicle//2)

    # Encode tours with g_s
    gs_out = torch.zeros(B, K, d_half, device=device, dtype=node_emb.dtype)

    # Clone state tensors to avoid in-place modification breaking autograd
    tours_snap = state.tours.clone().detach()
    tl_snap = state.tour_lengths.clone().detach()

    for b in range(B):
        for k in range(K):
            tour_len = tl_snap[b, k].item()
            if tour_len > 0:
                # Get node indices in this tour
                tour_nodes = tours_snap[b, k, :int(tour_len)].long()
                # Get node embeddings for these nodes
                tour_node_embs = node_emb[b, tour_nodes]  # (tour_len, d_node)
                # Apply g_s and average
                gs_encoded = tour_enc(tour_node_embs)  # (tour_len, d_half)
                gs_out[b, k] = gs_encoded.mean(dim=0)
            # else: zero vector (already initialized)

    # Concatenate: [g_v(v_k) ; mean(g_s(...))]
    vehicle_emb = torch.cat([gv_out, gs_out], dim=-1)  # (B, K, d_vehicle)
    return vehicle_emb
