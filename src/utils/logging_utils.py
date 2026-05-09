"""Logging utilities: console/file logger, TensorBoard wrapper, and AverageMeter."""

import os
import logging
from torch.utils.tensorboard import SummaryWriter


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Configure a logger that writes to both console and a log file.

    Args:
        log_dir: Directory for log files.
        experiment_name: Name used for the log file and logger.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{experiment_name}.log")

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(ch)

    return logger


class TBWriter:
    """Thin wrapper around TensorBoard SummaryWriter.

    Provides simplified methods for the metrics we care about.
    """

    def __init__(self, log_dir: str):
        """Initialize TensorBoard writer.

        Args:
            log_dir: Directory for TensorBoard event files.
        """
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Data identifier (e.g. 'train/cost').
            value: Scalar value.
            step: Global step.
        """
        self._writer.add_scalar(tag, value, step)

    def histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram of values.

        Args:
            tag: Data identifier.
            values: Tensor or ndarray of values.
            step: Global step.
        """
        self._writer.add_histogram(tag, values, step)

    def close(self) -> None:
        """Flush and close the writer."""
        self._writer.flush()
        self._writer.close()


class AverageMeter:
    """Tracks running average of a metric.

    Attributes:
        val: Last recorded value.
        sum: Accumulated sum.
        count: Number of observations.
        avg: Running average (sum / count).
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.avg: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Record a new observation.

        Args:
            val: Observed value.
            n: Number of observations this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
