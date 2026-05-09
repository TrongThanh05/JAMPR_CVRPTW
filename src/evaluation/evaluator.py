"""Evaluator: greedy and sampling evaluation with benchmark table output."""

import logging
import torch
from src.evaluation.metrics import compute_cost
from src.utils.time_utils import Timer

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates JAMPR model on test/validation datasets.

    Args:
        model: JAMPRModel instance.
        config: Config dict.
        device: Torch device.
    """

    def __init__(self, model, config: dict, device=None):
        self.model = model
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        tc = config.get("training", config)
        self.problem = tc.get("problem", "cvrptw_tw1")

    def evaluate_greedy(self, dataset) -> dict:
        """Evaluate using greedy decoding.

        Args:
            dataset: VRPTWDataset.

        Returns:
            Dict with cost_mean, cost_std, k_mean, k_std, time_per_instance_s.
        """
        costs_list = []
        k_list = []
        total_time = 0.0
        n = len(dataset) if hasattr(dataset, '__len__') else 0

        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                item = dataset[i]
                batch = {}
                for key, val in item.items():
                    if isinstance(val, torch.Tensor):
                        batch[key] = val.unsqueeze(0).to(self.device)
                    else:
                        batch[key] = val

                with Timer() as t:
                    solutions, _ = self.model(batch, greedy=True)
                total_time += t.elapsed

                cost = compute_cost(solutions, batch, self.problem)
                costs_list.append(cost.item())
                k_list.append(len(solutions[0]))

        costs_t = torch.tensor(costs_list)
        k_t = torch.tensor(k_list, dtype=torch.float32)

        return {
            "cost_mean": costs_t.mean().item() if len(costs_t) > 0 else 0.0,
            "cost_std": costs_t.std().item() if len(costs_t) > 1 else 0.0,
            "k_mean": k_t.mean().item() if len(k_t) > 0 else 0.0,
            "k_std": k_t.std().item() if len(k_t) > 1 else 0.0,
            "time_per_instance_s": total_time / max(n, 1),
        }

    def evaluate_sampling(self, dataset, n_samples: int = 1280) -> dict:
        """Evaluate using sampling (take best of n_samples).

        Args:
            dataset: VRPTWDataset.
            n_samples: Number of samples per instance.

        Returns:
            Same dict structure as evaluate_greedy.
        """
        costs_list = []
        k_list = []
        total_time = 0.0
        n = len(dataset) if hasattr(dataset, '__len__') else 0

        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                item = dataset[i]
                # Repeat instance n_samples times
                batch = {}
                for key, val in item.items():
                    if isinstance(val, torch.Tensor):
                        batch[key] = val.unsqueeze(0).repeat(n_samples, *([1] * val.dim())).to(self.device)
                    else:
                        batch[key] = val

                with Timer() as t:
                    solutions, _ = self.model(batch, greedy=False)

                total_time += t.elapsed

                # Compute cost for all samples, take best
                all_costs = compute_cost(solutions, batch, self.problem)
                best_idx = all_costs.argmin().item()
                costs_list.append(all_costs[best_idx].item())
                k_list.append(len(solutions[best_idx]))

        costs_t = torch.tensor(costs_list)
        k_t = torch.tensor(k_list, dtype=torch.float32)

        return {
            "cost_mean": costs_t.mean().item() if len(costs_t) > 0 else 0.0,
            "cost_std": costs_t.std().item() if len(costs_t) > 1 else 0.0,
            "k_mean": k_t.mean().item() if len(k_t) > 0 else 0.0,
            "k_std": k_t.std().item() if len(k_t) > 1 else 0.0,
            "time_per_instance_s": total_time / max(n, 1),
        }

    def benchmark(self, dataset) -> str:
        """Produce formatted benchmark table.

        Args:
            dataset: VRPTWDataset.

        Returns:
            Formatted string table.
        """
        greedy = self.evaluate_greedy(dataset)
        lines = [
            "=" * 60,
            f"{'Model':<15} {'N':<5} {'Cost':>10} {'k':>6} {'t_inf':>10}",
            "-" * 60,
            f"{'JAMPR(greedy)':<15} {'?':<5} {greedy['cost_mean']:>10.2f} "
            f"{greedy['k_mean']:>6.2f} {greedy['time_per_instance_s']:>10.4f}s",
            "=" * 60,
        ]
        return "\n".join(lines)
