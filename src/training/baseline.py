"""Rollout Baseline for REINFORCE.

Uses greedy rollout of best model checkpoint as baseline.
"""

import logging
from copy import deepcopy

import torch
from torch import Tensor
from scipy.stats import ttest_rel

logger = logging.getLogger(__name__)


class RolloutBaseline:
    """Rollout baseline that uses greedy decoding of a copy of the best model.

    Args:
        model: JAMPRModel to copy as baseline.
        config: Training config dict.
    """

    def __init__(self, model, config: dict):
        tc = config.get("training", config)
        self.baseline_beta = tc.get("baseline_beta", 0.8)
        self.ttest_alpha = tc.get("ttest_alpha", 0.05)
        self.warmup_epochs = tc.get("baseline_warmup_epochs", 1)
        self.epoch = 0
        self.ema_cost = None

        # Deep copy of model for baseline (eval mode, no grad)
        self.baseline_model = deepcopy(model)
        self.baseline_model.eval()
        for p in self.baseline_model.parameters():
            p.requires_grad_(False)

    def eval(self, batch: dict) -> Tensor:
        """Run baseline model in greedy mode, return costs.

        Args:
            batch: Input batch dict.

        Returns:
            (B,) cost tensor (detached).
        """
        with torch.no_grad():
            solutions, _ = self.baseline_model(batch, greedy=True)
        # Compute costs using simple distance metric
        costs = self._compute_simple_costs(solutions, batch)

        # During warmup, use EMA
        if self.epoch < self.warmup_epochs:
            mean_cost = costs.mean().item()
            if self.ema_cost is None:
                self.ema_cost = mean_cost
            else:
                self.ema_cost = self.baseline_beta * self.ema_cost + \
                               (1 - self.baseline_beta) * mean_cost
            return torch.full_like(costs, self.ema_cost)

        return costs

    def _compute_simple_costs(self, solutions: list, batch: dict) -> Tensor:
        """Compute total route distance for baseline cost."""
        coords = batch["coords"]
        B = coords.shape[0]
        costs = torch.zeros(B, device=coords.device)
        for b in range(B):
            total = 0.0
            for tour in solutions[b]:
                prev = 0  # start at depot
                for node in tour:
                    total += torch.norm(coords[b, prev] - coords[b, node]).item()
                    prev = node
                total += torch.norm(coords[b, prev] - coords[b, 0]).item()  # return
            costs[b] = total
        return costs

    def update(self, model, val_dataset) -> bool:
        """Update baseline if model is significantly better.

        Uses paired t-test to check if improvement is significant.

        Args:
            model: Current model to compare against baseline.
            val_dataset: Validation dataset for evaluation.

        Returns:
            True if baseline was updated, False otherwise.
        """
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            # During warmup, always update
            self.baseline_model.load_state_dict(deepcopy(model.state_dict()))
            self.baseline_model.eval()
            logger.info("Baseline updated (warmup epoch %d)", self.epoch)
            return True

        # Sample a small validation batch for comparison
        n_eval = min(len(val_dataset) if hasattr(val_dataset, '__len__') else 100, 100)

        model.eval()
        model_costs = []
        baseline_costs = []

        with torch.no_grad():
            for i in range(min(n_eval, 32)):
                if hasattr(val_dataset, '__getitem__'):
                    item = val_dataset[i]
                    # Make batch of 1
                    batch = {k: v.unsqueeze(0) if isinstance(v, Tensor) else v
                             for k, v in item.items() if v is not None}
                else:
                    break

                m_sol, _ = model(batch, greedy=True)
                b_sol, _ = self.baseline_model(batch, greedy=True)

                m_cost = self._compute_simple_costs(m_sol, batch)
                b_cost = self._compute_simple_costs(b_sol, batch)

                model_costs.append(m_cost.item())
                baseline_costs.append(b_cost.item())

        model.train()

        if len(model_costs) < 2:
            return False

        # Paired t-test
        _, p_value = ttest_rel(model_costs, baseline_costs)

        model_mean = sum(model_costs) / len(model_costs)
        baseline_mean = sum(baseline_costs) / len(baseline_costs)

        if p_value < self.ttest_alpha and model_mean < baseline_mean:
            self.baseline_model.load_state_dict(deepcopy(model.state_dict()))
            self.baseline_model.eval()
            logger.info("Baseline updated: model_cost=%.4f < baseline_cost=%.4f (p=%.4f)",
                       model_mean, baseline_mean, p_value)
            return True

        logger.info("Baseline NOT updated: model=%.4f, baseline=%.4f, p=%.4f",
                    model_mean, baseline_mean, p_value)
        return False
