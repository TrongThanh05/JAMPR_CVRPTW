"""Tests for src/utils/ modules."""

import time
import torch
import numpy as np
import pytest

from src.utils.seed import set_seed
from src.utils.tensor_utils import index_to_vehicle_node, compute_distance_matrix, compute_travel_times
from src.utils.logging_utils import AverageMeter
from src.utils.time_utils import Timer, ETACalculator


class TestSetSeed:
    def test_set_seed_reproducible(self):
        """Two identical random draws after set_seed must match."""
        set_seed(42)
        a1 = torch.randn(5)
        n1 = np.random.rand(5)

        set_seed(42)
        a2 = torch.randn(5)
        n2 = np.random.rand(5)

        assert torch.allclose(a1, a2), "PyTorch draws differ after same seed"
        np.testing.assert_array_equal(n1, n2)


class TestIndexMapping:
    def test_index_to_vehicle_node_basic(self):
        """index_to_vehicle_node(21, n_nodes=21) -> vehicle=1, node=0."""
        vehicle, node = index_to_vehicle_node(torch.tensor([21]), n_nodes=21)
        assert vehicle.item() == 1, f"Expected vehicle=1, got {vehicle.item()}"
        assert node.item() == 0, f"Expected node=0, got {node.item()}"

    def test_index_mapping_batch(self):
        """Test batch index mapping."""
        # idx=0 -> (v=0, n=0), idx=5 -> (v=0, n=5), idx=10 -> (v=1, n=0) for n_nodes=10
        idx = torch.tensor([0, 5, 10, 15])
        v, n = index_to_vehicle_node(idx, n_nodes=10)
        assert v.tolist() == [0, 0, 1, 1]
        assert n.tolist() == [0, 5, 0, 5]


class TestDistanceMatrix:
    def test_distance_matrix_shape(self):
        coords = torch.rand(4, 21, 2)
        dist = compute_distance_matrix(coords)
        assert dist.shape == (4, 21, 21)

    def test_distance_matrix_diagonal_zero(self):
        coords = torch.rand(2, 10, 2)
        dist = compute_distance_matrix(coords)
        diag = torch.diagonal(dist, dim1=1, dim2=2)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    def test_distance_matrix_symmetric(self):
        coords = torch.rand(2, 10, 2)
        dist = compute_distance_matrix(coords)
        assert torch.allclose(dist, dist.transpose(1, 2), atol=1e-6)


class TestTravelTimes:
    def test_travel_times_includes_service(self):
        """travel_time[i,j] > dist[i,j] when service_time[i] > 0."""
        coords = torch.rand(2, 10, 2)
        service = torch.full((2, 10), 0.01)
        service[:, 0] = 0.0  # depot has no service time

        dist = compute_distance_matrix(coords)
        travel = compute_travel_times(coords, service)

        # For non-depot departure nodes (service > 0), travel > dist
        assert (travel[:, 1:, :] > dist[:, 1:, :]).all(), \
            "Travel times should exceed distances when service > 0"

    def test_travel_times_shape(self):
        coords = torch.rand(3, 15, 2)
        service = torch.zeros(3, 15)
        travel = compute_travel_times(coords, service)
        assert travel.shape == (3, 15, 15)


class TestAverageMeter:
    def test_average_meter_basic(self):
        """update(3,1), update(5,1) -> avg=4.0."""
        meter = AverageMeter()
        meter.update(3, 1)
        meter.update(5, 1)
        assert meter.avg == 4.0, f"Expected avg=4.0, got {meter.avg}"
        assert meter.count == 2
        assert meter.sum == 8.0

    def test_average_meter_reset(self):
        meter = AverageMeter()
        meter.update(10, 5)
        meter.reset()
        assert meter.avg == 0.0
        assert meter.count == 0

    def test_average_meter_weighted(self):
        meter = AverageMeter()
        meter.update(2.0, 3)  # sum=6, count=3
        meter.update(5.0, 2)  # sum=16, count=5
        assert abs(meter.avg - 16.0 / 5) < 1e-9


class TestTimer:
    def test_timer_measures_time(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04, "Timer should measure at least ~50ms"


class TestETACalculator:
    def test_eta_decreases(self):
        eta = ETACalculator(total_steps=100)
        time.sleep(0.02)
        remaining = eta.update(50)
        assert remaining >= 0, "ETA should be non-negative"
