"""Inverse LR Scheduler.

Formula from PROJECT_SPEC §3.1:
    η_t = (1 / (1 + γ * t)) * η_{t-1}
"""

import logging

logger = logging.getLogger(__name__)


class InverseLRScheduler:
    """Custom inverse learning rate scheduler.

    At each epoch t:
        η_t = η_{t-1} / (1 + γ * t)

    Args:
        optimizer: PyTorch optimizer.
        gamma: Decay rate (default 0.001).
    """

    def __init__(self, optimizer, gamma: float = 0.001):
        self.optimizer = optimizer
        self.gamma = gamma
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_lr = self.base_lr
        self._epoch = 0

    def step(self, epoch: int) -> None:
        """Update learning rate for the given epoch.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        self._epoch = epoch
        # η_t = η_{t-1} / (1 + γ * t)
        # Recurrence: lr_t = base_lr * product_{i=1}^{t} 1/(1+gamma*i)
        lr = self.base_lr
        for i in range(1, epoch + 1):
            lr = lr / (1.0 + self.gamma * i)
        self.current_lr = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.current_lr
        logger.info("LR updated to %.6e at epoch %d", self.current_lr, epoch)

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.current_lr

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "base_lr": self.base_lr,
            "gamma": self.gamma,
            "current_lr": self.current_lr,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.base_lr = state["base_lr"]
        self.gamma = state["gamma"]
        self.current_lr = state["current_lr"]
        self._epoch = state["epoch"]
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.current_lr
