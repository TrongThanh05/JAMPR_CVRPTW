"""Tests for src/models/ modules."""

import torch
import pytest


class TestNodeEncoder:
    def test_output_shape(self):
        """NodeEncoder: (4, 21, 3) -> (4, 21, 128)."""
        from src.models.node_encoder import NodeEncoder
        enc = NodeEncoder(input_dim=3, d_node=128, n_layers=3, n_heads=8)
        x = torch.rand(4, 21, 3)
        out = enc(x)
        assert out.shape == (4, 21, 128), f"Expected (4,21,128), got {out.shape}"


class TestAffinityNetwork:
    def test_output_shape(self):
        """AffinityNetwork: vehicle(2,3,128), node(2,21,128) -> (2,63,128)."""
        from src.models.affinity import AffinityNetwork
        net = AffinityNetwork(d_node=128, d_vehicle=128, d_M=128)
        v = torch.rand(2, 3, 128)
        n = torch.rand(2, 21, 128)
        out = net(v, n)
        assert out.shape == (2, 63, 128), f"Expected (2,63,128), got {out.shape}"


class TestMasking:
    def _make_state(self):
        from src.models.jampr import VRPTWState
        B, N = 2, 5  # 5 customers + 1 depot = 6 nodes
        batch = {
            "coords": torch.rand(B, N + 1, 2),
            "demands": torch.rand(B, N + 1) * 0.1,
            "time_windows": torch.zeros(B, N + 1, 2),
            "service_times": torch.full((B, N + 1), 0.01),
        }
        batch["demands"][:, 0] = 0.0
        batch["service_times"][:, 0] = 0.0
        # Set feasible time windows
        batch["time_windows"][:, 0] = torch.tensor([0.0, 1.0])
        for i in range(1, N + 1):
            batch["time_windows"][:, i, 0] = 0.0
            batch["time_windows"][:, i, 1] = 1.0
        state = VRPTWState.from_batch(batch, m_con=2, m_pre=6)
        return state

    def test_visited_nodes_masked(self):
        """Visited nodes should be masked for all vehicles."""
        from src.models.jampr import JAMPRModel
        state = self._make_state()
        # Mark node 1 as visited
        state.visited[:, 1] = True

        # Build a minimal model just for compute_mask
        config = {"model": {"d_node": 128, "d_vehicle": 128, "d_M": 128,
                            "n_heads": 8, "n_sa_layers": 3, "clip_value": 10.0,
                            "m_con": 2, "d_decoder_hidden": 256,
                            "vehicle_encoder_layers": 3, "tour_encoder_layers": 2}}
        model = JAMPRModel(config)
        mask = model.compute_mask(state)
        N1 = 6
        # Reshape to (B, m_con, N+1) to check node 1
        mask_r = mask.reshape(2, 2, N1)
        assert mask_r[:, :, 1].all(), "Node 1 should be masked for all vehicles"

    def test_capacity_constraint(self):
        """Vehicle with insufficient capacity should have that node masked."""
        state = self._make_state()
        # Set vehicle 0 load very low
        state.vehicle_load[:, 0] = 0.01
        # Set node 2 demand high
        state.demands[:, 2] = 0.5

        config = {"model": {"d_node": 128, "d_vehicle": 128, "d_M": 128,
                            "n_heads": 8, "n_sa_layers": 3, "clip_value": 10.0,
                            "m_con": 2, "d_decoder_hidden": 256,
                            "vehicle_encoder_layers": 3, "tour_encoder_layers": 2}}
        from src.models.jampr import JAMPRModel
        model = JAMPRModel(config)
        mask = model.compute_mask(state)
        N1 = 6
        mask_r = mask.reshape(2, 2, N1)
        # Vehicle 0 (active_ids[b,0]=0) should have node 2 masked
        assert mask_r[:, 0, 2].all(), "Node 2 should be masked for vehicle 0 (low capacity)"

    def test_depot_always_feasible(self):
        """Depot (node 0) should never be masked for active vehicles."""
        state = self._make_state()
        config = {"model": {"d_node": 128, "d_vehicle": 128, "d_M": 128,
                            "n_heads": 8, "n_sa_layers": 3, "clip_value": 10.0,
                            "m_con": 2, "d_decoder_hidden": 256,
                            "vehicle_encoder_layers": 3, "tour_encoder_layers": 2}}
        from src.models.jampr import JAMPRModel
        model = JAMPRModel(config)
        mask = model.compute_mask(state)
        N1 = 6
        mask_r = mask.reshape(2, 2, N1)
        assert not mask_r[:, :, 0].any(), "Depot should never be masked"


class TestFullForwardPass:
    def _get_config(self):
        return {
            "model": {
                "d_node": 32, "d_vehicle": 32, "d_M": 32,
                "n_heads": 4, "n_sa_layers": 1, "clip_value": 10.0,
                "m_con": 2, "d_decoder_hidden": 64,
                "vehicle_encoder_layers": 1, "tour_encoder_layers": 1,
            },
            "training": {"m_pre": 6},
        }

    def _get_batch(self, B=2, N=5):
        batch = {
            "coords": torch.rand(B, N + 1, 2),
            "demands": torch.rand(B, N + 1) * 0.05,
            "time_windows": torch.zeros(B, N + 1, 2),
            "service_times": torch.full((B, N + 1), 0.01),
        }
        batch["demands"][:, 0] = 0.0
        batch["service_times"][:, 0] = 0.0
        batch["time_windows"][:, 0] = torch.tensor([0.0, 1.0])
        for i in range(1, N + 1):
            batch["time_windows"][:, i, 0] = 0.0
            batch["time_windows"][:, i, 1] = 1.0
        return batch

    def test_no_nan_in_output(self):
        """Forward pass should not produce NaN in log_probs."""
        from src.models.jampr import JAMPRModel
        config = self._get_config()
        model = JAMPRModel(config)
        model.eval()
        batch = self._get_batch()
        with torch.no_grad():
            solutions, log_probs = model(batch, greedy=True)
        # Filter out -inf (valid for zero-prob actions)
        finite_lp = log_probs[log_probs != float('-inf')]
        assert not torch.isnan(finite_lp).any(), "NaN in log_probs!"

    def test_solution_feasibility(self):
        """Forward pass should produce structurally valid solutions."""
        from src.models.jampr import JAMPRModel
        config = self._get_config()
        model = JAMPRModel(config)
        model.eval()
        batch = self._get_batch(B=2, N=5)
        with torch.no_grad():
            solutions, log_probs = model(batch, greedy=True)
        # Check structure: solutions is list of B items, each is list of tours
        assert len(solutions) == 2, f"Expected 2 solutions, got {len(solutions)}"
        for b, sol in enumerate(solutions):
            assert isinstance(sol, list), "Each solution should be a list of tours"
            # Each tour should contain valid node indices
            for tour in sol:
                for node in tour:
                    assert 1 <= node <= 5, f"Invalid node {node} in tour"
            # No duplicate nodes across tours
            all_nodes = []
            for tour in sol:
                all_nodes.extend(tour)
            assert len(all_nodes) == len(set(all_nodes)), \
                f"Batch {b}: duplicate nodes in solution"
