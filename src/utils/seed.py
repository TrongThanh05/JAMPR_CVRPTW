"""Seed utilities for reproducibility across Python, NumPy, and PyTorch."""

import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all frameworks.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs),
    and enables deterministic cuDNN behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)
