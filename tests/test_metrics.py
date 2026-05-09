"""Tests for src/evaluation/metrics.py."""

import torch
import pytest

from src.evaluation.metrics import compute_cost, check_feasibility, compute_arrival_times


class TestComputeCost:
    def _make_batch(self):
        """Create a simple test batch."""
        coords = torch.tensor([
            [0.0, 0.0],  # depot
            [1.0, 0.0],  # node 1
            [0.0, 1.0],  # node 2
        ]).unsqueeze(0)  # (1, 3, 2)
        demands = torch.tensor([[0.0, 0.1, 0.1]])
        tw = torch.tensor([
            [[0.0, 1.0],    # depot
             [0.0, 0.5],    # node 1: window [0, 0.5]
             [0.0, 0.5]],   # node 2: window [0, 0.5]
        ])
        st = torch.tensor([[0.0, 0.01, 0.01]])
        return {"coords": coords, "demands": demands,
                "time_windows": tw, "service_times": st}

    def test_tw1_late_arrival_infinite_cost(self):
        """TW1: arrival after b_i should give infinite cost."""
        batch = self._make_batch()
        # Set very tight TW for node 1 so it's impossible
        batch["time_windows"][0, 1] = torch.tensor([0.0, 0.001])
        tours = [[[1, 2]]]  # Visit both from depot; travel to node 1 = 1.0 > 0.001
        cost = compute_cost(tours, batch, "cvrptw_tw1")
        assert cost[0].item() == float('inf'), \
            f"Expected inf cost for TW1 late arrival, got {cost[0].item()}"

    def test_tw2_late_penalty(self):
        """TW2: late arrival should incur finite penalty."""
        batch = self._make_batch()
        batch["time_windows"][0, 1] = torch.tensor([0.0, 0.5])
        # Visit node 1 (dist=1.0 > b=0.5), late by 0.5
        tours = [[[1]]]
        cost = compute_cost(tours, batch, "cvrptw_tw2")
        assert cost[0].item() < float('inf'), "TW2 should have finite cost"
        assert cost[0].item() > 0, "Cost should be positive"

    def test_tw3_early_penalty(self):
        """TW3: early arrival should incur penalty."""
        batch = self._make_batch()
        # Set TW for node 1: [0.5, 1.0] — vehicle arrives at time 1.0 (dist=1.0)
        # That's within the window, so no early penalty
        # Set TW for node 1: [2.0, 3.0] — vehicle arrives at 1.0, early by 1.0
        batch["time_windows"][0, 1] = torch.tensor([2.0, 3.0])
        tours = [[[1]]]
        cost_tw3 = compute_cost(tours, batch, "cvrptw_tw3")
        # alpha=0.1, early_dev=1.0 -> penalty=0.1
        assert cost_tw3[0].item() > 0, "Should have positive cost with early penalty"

    def test_depot_only_finite_cost(self):
        """Single vehicle visiting all nodes with no TW violation."""
        batch = self._make_batch()
        # Wide windows
        batch["time_windows"][0, 1] = torch.tensor([0.0, 100.0])
        batch["time_windows"][0, 2] = torch.tensor([0.0, 100.0])
        tours = [[[1, 2]]]
        cost = compute_cost(tours, batch, "cvrptw_tw1")
        assert cost[0].item() < float('inf'), "Should have finite cost"
        assert cost[0].item() > 0, "Should have positive cost"


class TestFeasibility:
    def test_feasibility_check_duplicate(self):
        """Tour with repeated node should be infeasible."""
        batch = {
            "demands": torch.tensor([0.0, 0.1, 0.1, 0.1]),
        }
        tours = [[1, 2, 1]]  # node 1 visited twice
        feasible, violations = check_feasibility(tours, batch)
        assert not feasible, "Should be infeasible with duplicate"
        assert any("Duplicate" in v for v in violations)

    def test_feasibility_missing_nodes(self):
        """Missing customers should be reported."""
        batch = {
            "demands": torch.tensor([0.0, 0.1, 0.1, 0.1]),
        }
        tours = [[1, 2]]  # node 3 missing
        feasible, violations = check_feasibility(tours, batch)
        assert not feasible
        assert any("Missing" in v for v in violations)


class TestArrivalTimes:
    def test_arrival_time_waiting(self):
        """Vehicle arrives before a_i, should wait until a_i."""
        coords = torch.tensor([
            [0.0, 0.0],   # depot
            [0.1, 0.0],   # node 1 (very close)
        ])
        st = torch.tensor([0.0, 0.01])
        tw = torch.tensor([
            [0.0, 1.0],   # depot
            [0.5, 1.0],   # node 1: opens at 0.5
        ])
        tour = [1]
        arrivals = compute_arrival_times(tour, coords, st, tw)
        # Travel to node 1: 0.1, but TW starts at 0.5 -> arrival = 0.5
        assert arrivals[0] >= 0.5 - 1e-6, \
            f"Should wait until a_i=0.5, got arrival={arrivals[0]}"
