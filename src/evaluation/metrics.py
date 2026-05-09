"""Evaluation metrics: cost computation, feasibility checking, arrival times.

Implements cost formula from PROJECT_SPEC §1.3.
"""

import math
import torch
from torch import Tensor


def compute_cost(tours: list, batch: dict, problem_type: str = "cvrptw_tw1",
                 alpha: float = 1.0, beta: float = float('inf')) -> Tensor:
    """Compute total cost for a batch of solutions.

    Formula §1.3:
        cost_k = Σ_{(i,j)∈tour} travel_time(i,j) + α*λ(δ_ai) + β*λ(δ_bi)
        λ(x) = x (linear penalty)
        δ_ai = max(a_i - arrival_time, 0) (early)
        δ_bi = max(arrival_time - b_i, 0) (late)

    Args:
        tours: List of B solutions, each is list of tours (list of node indices).
        batch: Dict with coords, demands, time_windows, service_times.
        problem_type: 'cvrp', 'cvrptw_tw1', 'cvrptw_tw2', 'cvrptw_tw3'.
        alpha: Early arrival penalty weight.
        beta: Late arrival penalty weight.

    Returns:
        (B,) cost tensor.
    """
    # Set alpha/beta based on problem type if not explicitly overridden
    if problem_type == "cvrptw_tw1":
        alpha, beta = 1.0, float('inf')
    elif problem_type == "cvrptw_tw2":
        alpha, beta = 0.0, 0.5
    elif problem_type == "cvrptw_tw3":
        alpha, beta = 0.1, 0.5

    coords = batch["coords"]
    tw = batch.get("time_windows")
    st = batch.get("service_times")
    B = coords.shape[0]
    device = coords.device

    costs = torch.zeros(B, device=device)

    for b in range(B):
        total_cost = 0.0
        for tour in tours[b]:
            if len(tour) == 0:
                continue
            prev = 0  # depot
            current_time = 0.0

            for node in tour:
                # Travel distance
                travel_d = torch.norm(coords[b, prev] - coords[b, node]).item()
                # Add service time of departure node
                if st is not None and prev > 0:
                    travel_d += st[b, prev].item()

                total_cost += travel_d  # travel cost component
                current_time += travel_d

                # Time window penalties
                if tw is not None and node > 0:
                    a_n = tw[b, node, 0].item()
                    b_n = tw[b, node, 1].item()

                    # Early arrival: wait
                    if current_time < a_n:
                        early_dev = a_n - current_time
                        total_cost += alpha * early_dev
                        current_time = a_n  # wait until window opens

                    # Late arrival: penalty
                    if current_time > b_n:
                        late_dev = current_time - b_n
                        if beta == float('inf'):
                            total_cost = float('inf')
                            break
                        else:
                            total_cost += beta * late_dev

                prev = node

            if total_cost == float('inf'):
                break

            # Return to depot
            if len(tour) > 0:
                return_d = torch.norm(coords[b, prev] - coords[b, 0]).item()
                if st is not None and prev > 0:
                    return_d += st[b, prev].item()
                total_cost += return_d

        costs[b] = total_cost

    return costs


def check_feasibility(tours: list, batch: dict) -> tuple[bool, list[str]]:
    """Check all constraints for a single solution.

    Checks:
        (1) All customers visited
        (2) No duplicate visits
        (3) Capacity constraint (normalized demand per tour <= 1.0)
        (4) Fleet size constraint (number of tours <= K)

    Args:
        tours: List of tours for one instance.
        batch: Batch dict (for single instance, tensors without batch dim).

    Returns:
        (feasible, violations): True if feasible, list of violation descriptions.
    """
    violations = []
    all_nodes = []
    n_customers = batch["demands"].shape[0] - 1  # exclude depot

    for tour in tours:
        all_nodes.extend(tour)

    # Check all customers visited
    visited = set(all_nodes)
    expected = set(range(1, n_customers + 1))
    missing = expected - visited
    if missing:
        violations.append(f"Missing customers: {missing}")

    # Check duplicates
    if len(all_nodes) != len(set(all_nodes)):
        from collections import Counter
        counts = Counter(all_nodes)
        dups = {n: c for n, c in counts.items() if c > 1}
        violations.append(f"Duplicate visits: {dups}")

    # Check capacity (demands are normalized, so capacity in normalized space = 1.0)
    norm_capacity = 1.0
    raw_capacity = batch.get("capacity", None)  # e.g. 500, 750
    demands = batch["demands"]
    for i, tour in enumerate(tours):
        tour_demand = sum(demands[n].item() for n in tour)
        if tour_demand > norm_capacity + 1e-6:
            if raw_capacity is not None:
                violations.append(
                    f"Tour {i}: demand {tour_demand:.4f} > capacity 1.0 "
                    f"(raw: {tour_demand * raw_capacity:.0f}/{raw_capacity})")
            else:
                violations.append(f"Tour {i}: demand {tour_demand:.4f} > capacity 1.0")

    # Check fleet size
    n_vehicles = batch.get("n_vehicles", None)
    if n_vehicles is not None and len(tours) > int(n_vehicles):
        violations.append(
            f"Fleet size exceeded: {len(tours)} tours > {n_vehicles} vehicles")

    feasible = len(violations) == 0
    return feasible, violations


def compute_arrival_times(tour: list, coords: Tensor,
                          service_times: Tensor,
                          time_windows: Tensor = None) -> list[float]:
    """Compute arrival times at each stop in a single tour.

    arrival[0] = 0 (depart from depot at time 0)
    arrival[i] = max(arrival[i-1] + travel_time(prev, curr), a_curr)

    Args:
        tour: List of node indices.
        coords: (N+1, 2) coordinates.
        service_times: (N+1,) service durations.
        time_windows: (N+1, 2) optional time windows.

    Returns:
        List of arrival times at each stop.
    """
    arrivals = []
    current_time = 0.0
    prev = 0  # depot

    for node in tour:
        travel_d = torch.norm(coords[prev] - coords[node]).item()
        if prev > 0:
            travel_d += service_times[prev].item()

        arrival = current_time + travel_d

        # Wait if arrive early
        if time_windows is not None and node > 0:
            a_n = time_windows[node, 0].item()
            arrival = max(arrival, a_n)

        arrivals.append(arrival)
        current_time = arrival

        prev = node

    return arrivals
