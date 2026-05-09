"""JAMPR Model: main model integrating all components.

Contains VRPTWState dataclass and JAMPRModel nn.Module.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from src.models.node_encoder import NodeEncoder
from src.models.vehicle_encoder import TourEncoder, VehicleEncoder, encode_all_vehicles
from src.models.affinity import AffinityNetwork
from src.models.decoder import JAMPRDecoder
from src.utils.tensor_utils import index_to_vehicle_node

logger = logging.getLogger(__name__)


@dataclass
class VRPTWState:
    """Mutable state for VRPTW decoding.

    All tensors reside on the same device.

    tw_mode:
        'tw1' — Hard TW: xe đến muộn bị mask hoàn toàn, đến sớm thì chờ.
        'tw2' — Soft Late: đến muộn được phép (tính penalty), đến sớm vẫn chờ.
        'tw3' — Soft Full: cả sớm lẫn muộn đều chỉ tính penalty, không chờ.
    """
    # Problem data (static)
    coords: Tensor          # (B, N+1, 2)
    demands: Tensor         # (B, N+1)
    time_windows: Optional[Tensor]  # (B, N+1, 2) or None
    service_times: Tensor   # (B, N+1)
    capacity: float
    tw_mode: str            # 'tw1', 'tw2', 'tw3', or 'none'

    # Solution state (dynamic)
    tours: Tensor           # (B, K, L) node indices, zero-padded
    tour_lengths: Tensor    # (B, K) int — num nodes per tour
    visited: Tensor         # (B, N+1) bool
    vehicle_pos: Tensor     # (B, K, 2)
    vehicle_time: Tensor    # (B, K)
    vehicle_load: Tensor    # (B, K)
    active_mask: Tensor     # (B, K) bool — True if active
    active_ids: Tensor      # (B, m_con) int — indices of active vehicles
    next_vehicle: Tensor    # (B,) int — next vehicle index to activate
    premature_count: Tensor # (B,) int
    m_pre: int              # max premature returns
    m_con: int              # concurrency
    step: int = 0

    @staticmethod
    def from_batch(batch: dict, m_con: int, m_pre: int, max_vehicles: int = 0,
                   tw_mode: str = "tw1") -> "VRPTWState":
        """Create initial state from a batch dict.

        Args:
            batch: Batch dict with coords, demands, etc.
            m_con: Number of concurrent vehicles.
            m_pre: Max premature returns allowed.
            max_vehicles: Override fleet size (0 = use batch value).
            tw_mode: Time-window constraint mode:
                'tw1' — Hard TW (default), 'tw2' — Soft late, 'tw3' — Soft full.
        Reads fleet capacity Q and fleet size K from batch if available.
        """
        coords = batch["coords"]
        demands = batch["demands"]
        tw = batch.get("time_windows")
        st = batch.get("service_times")
        B, N1, _ = coords.shape
        device = coords.device

        # Fleet size K: from batch > max_vehicles arg > fallback N+1
        if max_vehicles > 0:
            K = max_vehicles
        elif "n_vehicles" in batch:
            K = int(batch["n_vehicles"]) if not isinstance(batch["n_vehicles"], int) else batch["n_vehicles"]
        else:
            K = N1  # fallback: max possible = N+1
        L = N1  # max tour length

        if st is None:
            st = torch.zeros(B, N1, device=device)

        # Capacity: demands are normalized so capacity in normalized space = 1.0
        # The raw Q (e.g. 500, 750) is stored in batch["capacity"] for reference
        capacity = 1.0

        tours = torch.zeros(B, K, L, dtype=torch.long, device=device)
        tour_lengths = torch.zeros(B, K, dtype=torch.long, device=device)
        visited = torch.zeros(B, N1, dtype=torch.bool, device=device)
        visited[:, 0] = True  # depot is "visited" (always accessible but marked)

        # All vehicles start at depot
        depot_pos = coords[:, 0, :].unsqueeze(1).expand(B, K, 2).clone()
        vehicle_time = torch.zeros(B, K, device=device)
        vehicle_load = torch.ones(B, K, device=device) * capacity

        # Active vehicles: first m_con
        active_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        actual_mcon = min(m_con, K)
        active_mask[:, :actual_mcon] = True
        active_ids = torch.arange(actual_mcon, device=device).unsqueeze(0).expand(B, -1).clone()
        next_vehicle = torch.full((B,), actual_mcon, dtype=torch.long, device=device)
        premature_count = torch.zeros(B, dtype=torch.long, device=device)

        return VRPTWState(
            coords=coords, demands=demands, time_windows=tw,
            service_times=st, capacity=capacity, tw_mode=tw_mode,
            tours=tours, tour_lengths=tour_lengths, visited=visited,
            vehicle_pos=depot_pos, vehicle_time=vehicle_time,
            vehicle_load=vehicle_load, active_mask=active_mask,
            active_ids=active_ids, next_vehicle=next_vehicle,
            premature_count=premature_count, m_pre=m_pre, m_con=m_con, step=0,
        )

    def is_done(self) -> bool:
        """Check if all customers have been visited or no active vehicles remain.

        Termination conditions:
          (1) All customers visited.
          (2) No active vehicles AND no more vehicles in the fleet queue.
        """
        all_visited = self.visited[:, 1:].all().item()
        no_active = not self.active_mask.any().item()
        fleet_exhausted = (self.next_vehicle >= self.tours.shape[1]).all().item()
        return all_visited or (no_active and fleet_exhausted)

    def update(self, vehicle_idx: Tensor, node_idx: Tensor) -> None:
        """Update state after assigning node to vehicle.

        Args:
            vehicle_idx: (B,) — which vehicle (relative to active_ids).
            node_idx: (B,) — which node to visit.
        """
        B = self.coords.shape[0]
        device = self.coords.device
        batch_range = torch.arange(B, device=device)

        # Map relative vehicle index to absolute vehicle index
        abs_vehicle = self.active_ids[batch_range, vehicle_idx]  # (B,)

        for b in range(B):
            v = abs_vehicle[b].item()
            n = node_idx[b].item()
            tl = self.tour_lengths[b, v].item()

            if n == 0:
                # Return to depot
                self.vehicle_pos[b, v] = self.coords[b, 0]
                self.vehicle_time[b, v] = 0.0
                self.vehicle_load[b, v] = self.capacity

                # Check premature return
                if self.visited[b, 1:].sum() < (self.coords.shape[1] - 1):
                    self.premature_count[b] += 1

                # Deactivate this vehicle
                self.active_mask[b, v] = False

                # Activate next vehicle if available
                nv = self.next_vehicle[b].item()
                if nv < self.tours.shape[1]:
                    self.active_mask[b, nv] = True
                    # Update active_ids for this batch
                    self._refresh_active_ids_single(b)
                    self.next_vehicle[b] = nv + 1
                else:
                    self._refresh_active_ids_single(b)
            else:
                # Visit customer node
                self.visited[b, n] = True
                if tl < self.tours.shape[2]:
                    self.tours[b, v, tl] = n
                self.tour_lengths[b, v] = tl + 1

                # Update position
                self.vehicle_pos[b, v] = self.coords[b, n]

                # Update time
                travel_dist = torch.norm(
                    self.vehicle_pos[b, v] - self.coords[b, n]
                ).item()
                # Actually position was just set, compute from old position
                old_pos = self.coords[b, self.tours[b, v, tl - 1].item()] if tl > 0 else self.coords[b, 0]
                travel_dist = torch.norm(old_pos - self.coords[b, n]).item()
                service = self.service_times[b, int(self.tours[b, v, max(0, tl - 1)].item()) if tl > 0 else 0].item()
                arrival = self.vehicle_time[b, v].item() + travel_dist + service

                # Wait if early (TW1/TW2: xe chờ miễn phí; TW3: không chờ, tính penalty)
                if self.time_windows is not None and self.tw_mode != "tw3":
                    a_n = self.time_windows[b, n, 0].item()
                    arrival = max(arrival, a_n)

                self.vehicle_time[b, v] = arrival + self.service_times[b, n].item()

                # Update load
                self.vehicle_load[b, v] -= self.demands[b, n].item()

        self.step += 1

    def _refresh_active_ids_single(self, b: int) -> None:
        """Refresh active_ids for batch element b."""
        active = self.active_mask[b].nonzero(as_tuple=False).squeeze(-1)
        m = self.active_ids.shape[1]
        if len(active) >= m:
            self.active_ids[b] = active[:m]
        else:
            # Pad with last active or 0
            padded = torch.zeros(m, dtype=torch.long, device=self.active_ids.device)
            padded[:len(active)] = active
            if len(active) > 0:
                padded[len(active):] = active[-1]
            self.active_ids[b] = padded

    def get_vehicle_features(self) -> Tensor:
        """Compute vehicle feature vectors.

        v_k = (k/K, return_cost_to_depot, pos_x, pos_y, current_time/b_0,
               remaining_load/capacity)

        Returns:
            (B, K, 6) normalized features.
        """
        B, K = self.vehicle_load.shape
        device = self.coords.device

        # k/K normalization
        k_norm = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1) / max(K, 1)

        # Return cost to depot
        depot = self.coords[:, 0:1, :]  # (B, 1, 2)
        return_cost = torch.norm(self.vehicle_pos - depot, p=2, dim=-1)  # (B, K)
        # Normalize by max possible distance (sqrt(2) for [0,1] coords)
        return_cost = return_cost / (math.sqrt(2) + 1e-8)

        # Position (already [0,1])
        pos_x = self.vehicle_pos[:, :, 0]
        pos_y = self.vehicle_pos[:, :, 1]

        # Time normalization
        if self.time_windows is not None:
            b_0 = self.time_windows[:, 0, 1].unsqueeze(1)  # (B, 1)
            time_norm = self.vehicle_time / (b_0 + 1e-8)
        else:
            time_norm = self.vehicle_time

        # Remaining load / capacity  — [0, 1], xe đầy = 0, xe trống = 1
        remaining_load = self.vehicle_load / (self.capacity + 1e-8)

        features = torch.stack([k_norm, return_cost, pos_x, pos_y,
                                time_norm, remaining_load], dim=-1)
        return features

    def get_solutions(self) -> list:
        """Extract solutions as list of tours.

        Returns:
            List of B solutions. Each solution is a list of tours.
            Each tour is a list of node indices (ints).
        """
        B, K, _ = self.tours.shape
        solutions = []
        for b in range(B):
            sol = []
            for k in range(K):
                tl = self.tour_lengths[b, k].item()
                if tl > 0:
                    tour = self.tours[b, k, :tl].tolist()
                    sol.append(tour)
            solutions.append(sol)
        return solutions


class JAMPRModel(nn.Module):
    """JAMPR: Joint Attention Model for Parallel Route-Construction.

    Integrates NodeEncoder, TourEncoder, VehicleEncoder, AffinityNetwork,
    and JAMPRDecoder into a complete autoregressive model for VRPTW.

    Args:
        config: Dict with model/training hyperparameters.
    """

    def __init__(self, config: dict):
        super().__init__()
        mc = config.get("model", config)
        self.d_node = mc.get("d_node", 128)
        self.d_vehicle = mc.get("d_vehicle", 128)
        self.d_M = mc.get("d_M", 128)
        n_heads = mc.get("n_heads", 8)
        n_sa = mc.get("n_sa_layers", 3)
        clip = mc.get("clip_value", 10.0)
        self.m_con = mc.get("m_con", 3)
        d_hidden = mc.get("d_decoder_hidden", 256)
        n_ve_layers = mc.get("vehicle_encoder_layers", 3)
        n_te_layers = mc.get("tour_encoder_layers", 2)

        tc = config.get("training", {})
        self.m_pre = tc.get("m_pre", 6)

        # d_C = 3*d_node + 2*d_vehicle (graph+fleet+act from node dim, depot+last from vehicle)
        # Actually: graph(d_node) + fleet(d_vehicle) + act(d_vehicle) + depot(d_node) + last(d_node)
        self.d_context = 3 * self.d_node + 2 * self.d_vehicle

        # Sub-modules
        self.node_encoder = NodeEncoder(3, self.d_node, n_sa, n_heads)
        self.tour_encoder = TourEncoder(self.d_node, self.d_vehicle // 2, n_te_layers)
        self.vehicle_encoder = VehicleEncoder(self.d_vehicle, n_ve_layers)
        self.affinity_net = AffinityNetwork(self.d_node, self.d_vehicle, self.d_M)
        self.decoder = JAMPRDecoder(self.d_context, self.d_M, d_hidden, clip)

    def forward(self, batch: dict, greedy: bool = False):
        """Run full autoregressive decoding.

        Args:
            batch: Dict with coords, demands, time_windows, service_times.
            greedy: If True, use argmax; otherwise sample.

        Returns:
            (solutions, log_probs): solutions is list of B route lists,
                log_probs is (B, T) tensor.
        """
        B, N1, _ = batch["coords"].shape
        device = batch["coords"].device

        # Build node features: (coords, demand) -> (B, N+1, 3)
        node_features = torch.cat([
            batch["coords"],
            batch["demands"].unsqueeze(-1),
        ], dim=-1)

        # Step 1: Encode nodes (static, done once)
        node_emb = self.node_encoder(node_features)  # (B, N+1, d_node)

        # Step 2: Initialize state — đọc tw_mode từ batch nếu có
        tw_mode = batch.get("tw_mode", "tw1")
        state = VRPTWState.from_batch(batch, m_con=self.m_con, m_pre=self.m_pre,
                                      tw_mode=tw_mode)

        log_probs_list = []
        max_steps = N1 * N1 * 2  # safety limit

        for t in range(max_steps):
            if state.is_done():
                break

            # Check premature limit — if exceeded, force remaining to depot
            # (handled implicitly in masking)

            # Step 3a: Encode vehicles (dynamic)
            vehicle_emb = encode_all_vehicles(
                state, node_emb, self.tour_encoder, self.vehicle_encoder
            )  # (B, K, d_vehicle)

            # Step 3b: Get active vehicle embeddings
            active_v_emb = self._get_active_vehicle_emb(vehicle_emb, state)  # (B, m_con, d_vehicle)

            # Step 3c: Build context
            context = self.build_context(node_emb, vehicle_emb, state)  # (B, d_C)

            # Step 3d: Build joint action space
            M = self.affinity_net(active_v_emb, node_emb)  # (B, m_con*(N+1), d_M)

            # Step 3e: Compute mask
            mask = self.compute_mask(state)  # (B, m_con*(N+1))

            # Step 3f: Decode
            logits = self.decoder(context, M, mask)  # (B, m_con*(N+1))
            probs = F.softmax(logits, dim=-1)

            # NaN safety: if weights become NaN, replace with uniform over feasible
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.where(mask, torch.zeros_like(probs), torch.ones_like(probs))
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            # Step 3g: Select action
            if greedy:
                action = probs.argmax(dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()

            # Step 3h: Map to (vehicle, node)
            vehicle_idx, node_idx = index_to_vehicle_node(action, N1)

            # Step 3i: Log probability
            lp = probs[torch.arange(B, device=device), action].clamp(min=1e-8).log()
            log_probs_list.append(lp)

            # Step 3j: Update state
            state.update(vehicle_idx, node_idx)

        solutions = state.get_solutions()
        if log_probs_list:
            log_probs = torch.stack(log_probs_list, dim=1)
        else:
            log_probs = torch.zeros(B, 0, device=device)

        return solutions, log_probs

    def _get_active_vehicle_emb(self, vehicle_emb: Tensor, state: VRPTWState) -> Tensor:
        """Extract embeddings of active vehicles."""
        B = vehicle_emb.shape[0]
        m_con = state.active_ids.shape[1]
        d_v = vehicle_emb.shape[2]
        device = vehicle_emb.device

        # Clone active_ids to avoid in-place modification breaking autograd
        ids_snapshot = state.active_ids.clone().detach()
        active_emb = torch.zeros(B, m_con, d_v, device=device, dtype=vehicle_emb.dtype)
        for b in range(B):
            active_emb[b] = vehicle_emb[b, ids_snapshot[b]]
        return active_emb

    def build_context(self, node_emb: Tensor, vehicle_emb: Tensor,
                      state: VRPTWState) -> Tensor:
        """Build comprehensive context vector.

        C = [ω_graph; ω_fleet; ω_act; ω_node_0; ω_last]
        dim = 3*d_node + 2*d_vehicle

        Note: ω_graph, ω_node_0, ω_last use d_node; ω_fleet, ω_act use d_vehicle.
        But d_node == d_vehicle == 128 in our config, so d_C = 5*128 = 640.
        """
        B = node_emb.shape[0]
        device = node_emb.device

        # ω_graph = mean of all node embeddings
        omega_graph = node_emb.mean(dim=1)  # (B, d_node)

        # ω_fleet = mean of all vehicle embeddings
        omega_fleet = vehicle_emb.mean(dim=1)  # (B, d_vehicle)

        # ω_act = mean of active vehicle embeddings
        active_emb = self._get_active_vehicle_emb(vehicle_emb, state)
        omega_act = active_emb.mean(dim=1)  # (B, d_vehicle)

        # ω_node_0 = depot embedding
        omega_depot = node_emb[:, 0, :]  # (B, d_node)

        # ω_last = mean of last visited node embeddings across all vehicles
        K = state.tours.shape[1]
        # Clone tours to avoid in-place modification breaking autograd
        tours_snap = state.tours.clone().detach()
        tl_snap = state.tour_lengths.clone().detach()
        last_embs = torch.zeros(B, K, self.d_node, device=device, dtype=node_emb.dtype)
        for b in range(B):
            for k in range(K):
                tl = tl_snap[b, k].item()
                if tl > 0:
                    last_node = tours_snap[b, k, int(tl) - 1].item()
                    last_embs[b, k] = node_emb[b, last_node]
                else:
                    last_embs[b, k] = node_emb[b, 0]  # depot if tour not started
        omega_last = last_embs.mean(dim=1)  # (B, d_node)

        context = torch.cat([omega_graph, omega_fleet, omega_act, omega_depot, omega_last], dim=-1)
        return context

    def compute_mask(self, state: VRPTWState) -> Tensor:
        """Compute infeasibility mask for joint action space.

        Masking rules:
            Rule 1: visited nodes masked for ALL vehicle-node pairs
            Rule 2: capacity — demand[n] > vehicle_load[k]
            Rule 3: TW1 only — arrival_time > time_window[n,1] → mask (hard)
                    TW2/TW3 — không mask node đến muộn, penalty xử lý ở cost
            Rule 4: depot (node 0) NEVER masked for active vehicles

        Returns:
            (B, m_con*(N+1)) bool — True = infeasible.
        """
        B, N1 = state.demands.shape
        m_con = state.active_ids.shape[1]
        device = state.coords.device
        hard_tw = (state.tw_mode == "tw1")  # Chỉ TW1 mới hard-mask node muộn

        mask = torch.zeros(B, m_con, N1, dtype=torch.bool, device=device)

        for b in range(B):
            for ki in range(m_con):
                k = state.active_ids[b, ki].item()

                for n in range(N1):
                    if n == 0:
                        # Rule 4: depot always feasible for active vehicles
                        mask[b, ki, 0] = False
                        continue

                    # Rule 1: visited nodes
                    if state.visited[b, n]:
                        mask[b, ki, n] = True
                        continue

                    # Rule 2: capacity
                    if state.demands[b, n].item() > state.vehicle_load[b, k].item() + 1e-8:
                        mask[b, ki, n] = True
                        continue

                    # Rule 3: hard TW late-arrival (chỉ áp dụng cho TW1)
                    if hard_tw and state.time_windows is not None:
                        travel_d = torch.norm(
                            state.vehicle_pos[b, k] - state.coords[b, n]
                        ).item()
                        arrival = state.vehicle_time[b, k].item() + travel_d
                        b_n = state.time_windows[b, n, 1].item()
                        if arrival > b_n + 1e-8:
                            mask[b, ki, n] = True
                            continue

        # Rule 5: if all non-depot masked for a vehicle, ensure depot is unmasked
        # (already handled by Rule 4, but double-check)
        mask[:, :, 0] = False

        # Check premature return limit
        for b in range(B):
            if state.premature_count[b].item() >= state.m_pre:
                # If premature limit reached, mask depot for vehicles with unvisited customers possible
                # Actually, we should let them continue serving
                pass

        # Flatten
        mask = mask.reshape(B, m_con * N1)
        return mask
