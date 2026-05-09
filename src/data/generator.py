"""VRPTW data generator for CVRP and CVRP-TW problem instances."""

import math
import logging
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

# Default config path
_DEFAULT_CONFIG = "configs/data_config.yaml"


def _load_data_config(config_path: str = _DEFAULT_CONFIG) -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]


class VRPTWDataGenerator:
    """Generator for CVRP and CVRP-TW problem instances.

    Generates batches of problem instances with normalized features ready for
    model consumption.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        """Initialize with data config.

        Args:
            config_path: Path to data_config.yaml.
        """
        self.config = _load_data_config(config_path)

    def generate_cvrp_batch(self, n: int, batch_size: int,
                            seed: Optional[int] = None) -> dict:
        """Generate a batch of CVRP instances.

        Args:
            n: Number of customers (excluding depot).
            batch_size: Number of instances in batch.
            seed: Optional random seed.

        Returns:
            Dict with keys: coords, demands, time_windows, service_times.
            All tensors are float32 and normalized.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        cfg = self.config["cvrp"]
        capacity = cfg["capacity"][n]
        n_vehicles = cfg.get("n_vehicles", {}).get(n, n + 1)  # K xe, default N+1

        # Coords: Uniform [0,1]^2, shape (B, N+1, 2), depot at index 0
        coords = torch.rand(batch_size, n + 1, 2)

        # Demands: Uniform integer [1,9], normalized by capacity
        demands_raw = torch.randint(1, 10, (batch_size, n + 1)).float()
        demands_raw[:, 0] = 0.0  # Depot has zero demand
        demands = demands_raw / capacity

        # No time windows or service times for pure CVRP
        time_windows = None
        service_times = torch.zeros(batch_size, n + 1)

        return {
            "coords": coords,
            "demands": demands,
            "time_windows": time_windows,
            "service_times": service_times,
            "capacity": capacity,        # Q (raw, không normalize)
            "n_vehicles": n_vehicles,    # K xe trong đội
            "tw_mode": "none",           # CVRP thuần — không có TW
        }

    def generate_cvrptw_batch(self, n: int, batch_size: int,
                              seed: Optional[int] = None) -> dict:
        """Generate a batch of CVRP-TW instances (R201 Solomon style).

        Args:
            n: Number of customers (excluding depot).
            batch_size: Number of instances in batch.
            seed: Optional random seed.

        Returns:
            Dict with keys: coords, demands, time_windows, service_times.
            All tensors are float32 and normalized.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        cfg = self.config["cvrptw"]
        capacity = cfg["capacity"][n]
        n_vehicles = cfg.get("n_vehicles", {}).get(n, n + 1)  # K xe, default N+1
        time_horizon = cfg["time_horizon"]           # 1000
        service_dur = cfg["service_duration"]         # 10
        tw_width_factor = cfg["tw_width_factor"]      # 300
        coord_norm = cfg["coord_normalize"]           # 100.0

        # Coords: Uniform [0,100]^2, then normalize by 100
        coords_raw = torch.rand(batch_size, n + 1, 2) * 100.0
        coords = coords_raw / coord_norm  # [0, 1]

        # Demands: |N(15,10)| clipped [1,42], integer, normalized by capacity
        demands_raw = torch.randn(batch_size, n + 1) * 10.0 + 15.0
        demands_raw = demands_raw.abs()
        demands_raw = demands_raw.clamp(1.0, 42.0).floor()
        demands_raw[:, 0] = 0.0  # Depot demand = 0
        demands = demands_raw / capacity

        # Time windows generation (PROJECT_SPEC §4.2)
        # Work on un-normalized coords for distance computation
        depot_coords = coords_raw[:, 0:1, :]  # (B, 1, 2)
        customer_coords = coords_raw[:, 1:, :]  # (B, N, 2)

        # L2 distance from depot to each customer
        dist_to_depot = torch.norm(customer_coords - depot_coords, p=2, dim=-1)  # (B, N)

        # h_hat_i = ceil(dist) + 1
        h_hat = torch.ceil(dist_to_depot) + 1.0  # (B, N)

        # b_sample = 1000 - h_hat
        b_sample = time_horizon - h_hat  # (B, N)

        # a_i ~ Uniform(h_hat, b_sample)
        u = torch.rand(batch_size, n)  # uniform [0,1)
        a_i = h_hat + u * (b_sample - h_hat)  # (B, N)

        # eps = max(|N(0,1)|, 0.01)
        eps = torch.randn(batch_size, n).abs().clamp(min=0.01)

        # b_i = min(floor(a_i + eps * 300), b_sample)
        b_i = torch.min(torch.floor(a_i + eps * tw_width_factor), b_sample)

        # Ensure a_i <= b_i (should already be true by construction)
        b_i = torch.max(b_i, a_i)

        # Build time_windows tensor (B, N+1, 2)
        tw = torch.zeros(batch_size, n + 1, 2)
        tw[:, 0, 0] = 0.0              # Depot TW start
        tw[:, 0, 1] = 1.0              # Depot TW end (normalized)
        tw[:, 1:, 0] = a_i / time_horizon
        tw[:, 1:, 1] = b_i / time_horizon

        # Service times: 10/1000 = 0.01 for customers, 0.0 for depot
        service_times = torch.full((batch_size, n + 1), service_dur / time_horizon)
        service_times[:, 0] = 0.0

        return {
            "coords": coords,
            "demands": demands,
            "time_windows": tw,
            "service_times": service_times,
            "capacity": capacity,        # Q (raw, không normalize)
            "n_vehicles": n_vehicles,    # K xe trong đội
            "tw_mode": "tw1",            # placeholder — được ghi đè bởi generate_batch()
        }

    def generate_batch(self, problem: str, n: int, batch_size: int,
                       seed: Optional[int] = None) -> dict:
        """Generate a batch for any supported problem type.

        Args:
            problem: One of 'cvrp', 'cvrptw_tw1', 'cvrptw_tw2', 'cvrptw_tw3'.
            n: Number of customers.
            batch_size: Batch size.
            seed: Optional seed.

        Returns:
            Batch dict (bao gồm key 'tw_mode').
        """
        if problem == "cvrp":
            return self.generate_cvrp_batch(n, batch_size, seed)
        elif problem == "cvrptw_tw1":
            batch = self.generate_cvrptw_batch(n, batch_size, seed)
            batch["tw_mode"] = "tw1"
            return batch
        elif problem == "cvrptw_tw2":
            batch = self.generate_cvrptw_batch(n, batch_size, seed)
            batch["tw_mode"] = "tw2"
            return batch
        elif problem == "cvrptw_tw3":
            batch = self.generate_cvrptw_batch(n, batch_size, seed)
            batch["tw_mode"] = "tw3"
            return batch
        else:
            raise ValueError(f"Unknown problem type: {problem}")
