"""Reader for fixed VRPTW benchmark text files.

Reads the tab-separated .txt files produced by export_to_txt.py and converts
them into the tensor dict format used by the JAMPR training pipeline.

Usage:
    from src.data.fixed_data_reader import load_fixed_txt

    # Load all instances with TW1 mode
    data = load_fixed_txt("data/vrptw_n20_fixed.txt", tw_mode="tw1")

    # Load first 100 instances only
    data = load_fixed_txt("data/vrptw_n20_fixed.txt", tw_mode="tw1", max_instances=100)

    # Save as .pt for VRPTWDataset
    import torch
    torch.save(data, "outputs/data/cvrptw_tw1_n20_test.pt")
"""

import logging
import re
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_fixed_txt(txt_path: str, tw_mode: str = "tw1",
                   max_instances: Optional[int] = None) -> dict:
    """Load a fixed benchmark .txt file and return a JAMPR-compatible dict.

    Args:
        txt_path: Path to the .txt data file.
        tw_mode: Which time window to use ('tw1', 'tw2', or 'tw3').
        max_instances: If set, only load this many instances.

    Returns:
        Dict with keys: coords, demands, time_windows, service_times,
        capacity, n_vehicles, tw_mode.
    """
    assert tw_mode in ("tw1", "tw2", "tw3"), \
        f"tw_mode must be 'tw1', 'tw2', or 'tw3', got '{tw_mode}'"

    # Parse header to get metadata
    metadata = _parse_header(txt_path)
    N = metadata["N"]
    capacity = metadata["Q"]
    n_vehicles = metadata["K"]
    T = metadata["T"]
    total_instances = metadata["INSTANCES"]

    if max_instances is not None:
        total_instances = min(total_instances, max_instances)

    n_nodes = N + 1  # depot + customers

    # TW column index mapping:
    # Columns: INST NODE X Y DEMAND TW1_A TW1_B TW2_A TW2_B TW3_A TW3_B SERVICE
    tw_col_map = {"tw1": (5, 6), "tw2": (7, 8), "tw3": (9, 10)}
    tw_a_col, tw_b_col = tw_col_map[tw_mode]

    # Pre-allocate tensors
    coords = torch.zeros(total_instances, n_nodes, 2)
    demands = torch.zeros(total_instances, n_nodes)
    time_windows = torch.zeros(total_instances, n_nodes, 2)
    service_times = torch.zeros(total_instances, n_nodes)

    logger.info("Loading %s (N=%d, %d instances, %s)...",
                txt_path, N, total_instances, tw_mode)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            inst_idx = int(parts[0])
            if inst_idx >= total_instances:
                break

            node_idx = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            demand = float(parts[4])
            tw_a = float(parts[tw_a_col])
            tw_b = float(parts[tw_b_col])
            service = float(parts[11])

            # Store raw coordinates (divide by 100 for normalization)
            coords[inst_idx, node_idx, 0] = x / 100.0
            coords[inst_idx, node_idx, 1] = y / 100.0

            # Normalize demand by capacity
            demands[inst_idx, node_idx] = demand / capacity

            # Normalize time windows and service times by time horizon
            time_windows[inst_idx, node_idx, 0] = tw_a / T
            time_windows[inst_idx, node_idx, 1] = tw_b / T
            service_times[inst_idx, node_idx] = service / T

    logger.info("Loaded %d instances successfully.", total_instances)

    return {
        "coords": coords,
        "demands": demands,
        "time_windows": time_windows,
        "service_times": service_times,
        "capacity": capacity,
        "n_vehicles": n_vehicles,
        "tw_mode": tw_mode,
    }


def _parse_header(txt_path: str) -> dict:
    """Parse metadata from the header comments of a .txt data file."""
    metadata = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break

            # Parse: # N=20  INSTANCES=10000  Q=500  K=10  T=1000  SERVICE=10
            for match in re.finditer(r"(\w+)=(\d+)", line):
                key, value = match.group(1), int(match.group(2))
                metadata[key] = value

    return metadata
