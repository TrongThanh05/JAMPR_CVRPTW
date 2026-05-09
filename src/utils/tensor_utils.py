"""Tensor utilities for index mapping, distance computation, and travel times."""

import torch
from torch import Tensor


def index_to_vehicle_node(idx: Tensor, n_nodes: int) -> tuple[Tensor, Tensor]:
    """Map flat joint-action index to (vehicle_index, node_index).

    The joint action space flattens vehicle-node pairs as:
        flat_idx = vehicle * n_nodes + node

    Args:
        idx: Flat indices, shape (B,) or scalar.
        n_nodes: Number of nodes (N+1, including depot).

    Returns:
        Tuple of (vehicle_idx, node_idx), same shape as idx.
    """
    vehicle = idx // n_nodes
    node = idx % n_nodes
    return vehicle, node


def compute_distance_matrix(coords: Tensor) -> Tensor:
    """Compute pairwise L2 distance matrix.

    Args:
        coords: Node coordinates, shape (B, N, 2).

    Returns:
        Distance matrix, shape (B, N, N).
    """
    # (B, N, 1, 2) - (B, 1, N, 2) -> (B, N, N, 2) -> norm -> (B, N, N)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = torch.norm(diff, p=2, dim=-1)
    return dist


def compute_travel_times(coords: Tensor, service_times: Tensor) -> Tensor:
    """Compute travel time matrix including service duration at departure node.

    travel_time(i -> j) = ||r_i - r_j||_2 + service_times[i]

    Args:
        coords: Node coordinates, shape (B, N, 2).
        service_times: Service duration per node, shape (B, N).

    Returns:
        Travel time matrix, shape (B, N, N).
    """
    dist = compute_distance_matrix(coords)
    # Add service time of departure node i to each edge (i, j)
    travel = dist + service_times.unsqueeze(2)  # (B, N, 1) broadcast
    return travel
