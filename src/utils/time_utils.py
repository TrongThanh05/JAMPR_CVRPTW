"""Timing utilities: context-manager Timer and ETACalculator."""

import time


class Timer:
    """Context manager that measures elapsed wall-clock time.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed)
    """

    def __init__(self):
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self._start


class ETACalculator:
    """Estimates time of arrival (remaining seconds) for a multi-step process.

    Args:
        total_steps: Total number of steps expected.
    """

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self._start_time = time.perf_counter()

    def update(self, step: int) -> float:
        """Return estimated seconds remaining given current step.

        Args:
            step: Current step (1-indexed for best results).

        Returns:
            Estimated seconds remaining. Returns 0.0 if step >= total_steps.
        """
        if step <= 0:
            return float("inf")
        elapsed = time.perf_counter() - self._start_time
        rate = elapsed / step
        remaining = self.total_steps - step
        return max(rate * remaining, 0.0)
