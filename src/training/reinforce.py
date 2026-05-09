"""REINFORCE loss computation.

Implements formula §11 from docs/math_formulas.md.
"""

import torch
from torch import Tensor


def compute_reinforce_loss(log_probs: Tensor, costs: Tensor,
                           baseline_costs: Tensor) -> Tensor:
    """Compute REINFORCE policy gradient loss with baseline.

    Formula §11:
        advantage = costs - baseline_costs.detach()
        log_prob_sum = log_probs.sum(dim=1)
        loss = (advantage * log_prob_sum).mean()

    Minimizing this loss minimizes expected cost.

    Args:
        log_probs: (B, T) — log probabilities at each decoding step.
        costs: (B,) — total cost of sampled solutions.
        baseline_costs: (B,) — cost of baseline solutions (detached).

    Returns:
        Scalar loss tensor.
    """
    advantage = costs - baseline_costs.detach()  # (B,)
    log_prob_sum = log_probs.sum(dim=1)           # (B,)
    loss = (advantage * log_prob_sum).mean()       # scalar
    return loss
