"""Tests for src/data/ modules."""

import torch
import pytest

from src.data.generator import VRPTWDataGenerator


@pytest.fixture
def generator():
    return VRPTWDataGenerator()


class TestCVRPTWGeneration:
    def test_cvrptw_tw_feasibility(self, generator):
        """All a_i <= b_i, and depot TW = [0, 1] (normalized)."""
        data = generator.generate_cvrptw_batch(n=20, batch_size=32, seed=42)
        tw = data["time_windows"]
        # a_i <= b_i for all nodes
        assert (tw[:, :, 0] <= tw[:, :, 1] + 1e-6).all(), \
            "Time window lower bound exceeds upper bound"
        # Depot TW = [0, 1]
        assert torch.allclose(tw[:, 0, 0], torch.zeros(32)), "Depot TW start != 0"
        assert torch.allclose(tw[:, 0, 1], torch.ones(32)), "Depot TW end != 1"

    def test_demand_normalization(self, generator):
        """Max demand < 1.0 for all instances (demands are normalized by capacity)."""
        data = generator.generate_cvrptw_batch(n=20, batch_size=64, seed=123)
        assert (data["demands"] < 1.0).all(), \
            f"Demands should be < 1.0 after normalization, max={data['demands'].max()}"

    def test_batch_shapes(self, generator):
        """coords (B,N+1,2), demands (B,N+1), tw (B,N+1,2)."""
        B, N = 16, 20
        data = generator.generate_cvrptw_batch(n=N, batch_size=B, seed=1)
        assert data["coords"].shape == (B, N + 1, 2)
        assert data["demands"].shape == (B, N + 1)
        assert data["time_windows"].shape == (B, N + 1, 2)
        assert data["service_times"].shape == (B, N + 1)

    def test_depot_demand_zero(self, generator):
        """Depot demand must be zero for all instances."""
        data = generator.generate_cvrptw_batch(n=20, batch_size=32, seed=7)
        assert (data["demands"][:, 0] == 0).all(), "Depot demand should be 0"

    def test_service_time_depot_zero(self, generator):
        """Depot service time must be zero."""
        data = generator.generate_cvrptw_batch(n=20, batch_size=32, seed=7)
        assert (data["service_times"][:, 0] == 0).all(), \
            "Depot service time should be 0"


class TestCVRPGeneration:
    def test_cvrp_batch_shapes(self, generator):
        B, N = 8, 50
        data = generator.generate_cvrp_batch(n=N, batch_size=B, seed=99)
        assert data["coords"].shape == (B, N + 1, 2)
        assert data["demands"].shape == (B, N + 1)
        assert data["time_windows"] is None

    def test_cvrp_depot_demand_zero(self, generator):
        data = generator.generate_cvrp_batch(n=20, batch_size=16, seed=42)
        assert (data["demands"][:, 0] == 0).all()


class TestGenerateBatch:
    def test_generate_batch_cvrptw(self, generator):
        data = generator.generate_batch("cvrptw_tw1", n=20, batch_size=4, seed=1)
        assert "coords" in data
        assert data["time_windows"] is not None

    def test_generate_batch_cvrp(self, generator):
        data = generator.generate_batch("cvrp", n=20, batch_size=4, seed=1)
        assert data["time_windows"] is None
