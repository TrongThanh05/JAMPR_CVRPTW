"""Dataset classes for VRPTW: file-based and online (on-the-fly) generation."""

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class VRPTWDataset(Dataset):
    """Dataset that loads pre-generated VRPTW instances from a .pt file.

    Args:
        data_path: Path to a .pt file saved by generate_data.py.
    """

    def __init__(self, data_path: str):
        logger.info("Loading dataset from %s", data_path)
        self.data = torch.load(data_path, weights_only=False)
        self.size = self.data["coords"].shape[0]
        logger.info("Loaded %d instances", self.size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Return one instance with all tensors of shape (N+1, *)."""
        item = {
            "coords": self.data["coords"][idx],       # (N+1, 2)
            "demands": self.data["demands"][idx],      # (N+1,)
        }
        if self.data.get("time_windows") is not None:
            item["time_windows"] = self.data["time_windows"][idx]  # (N+1, 2)
        else:
            item["time_windows"] = None
        if self.data.get("service_times") is not None:
            item["service_times"] = self.data["service_times"][idx]  # (N+1,)
        else:
            item["service_times"] = torch.zeros(self.data["coords"].shape[1])
        return item


class OnlineVRPTWDataset(IterableDataset):
    """Dataset that generates VRPTW instances on-the-fly during training.

    No disk I/O — instances are created fresh by the generator each iteration.

    Args:
        generator: VRPTWDataGenerator instance.
        problem: Problem type string (e.g. 'cvrptw_tw1').
        n: Number of customers.
        n_instances: Total number of instances to generate per epoch.
    """

    def __init__(self, generator, problem: str, n: int, n_instances: int):
        self.generator = generator
        self.problem = problem
        self.n = n
        self.n_instances = n_instances

    def __iter__(self):
        """Yield individual instances from generated batches."""
        generated = 0
        while generated < self.n_instances:
            batch_size = min(256, self.n_instances - generated)
            batch = self.generator.generate_batch(
                self.problem, self.n, batch_size
            )
            for i in range(batch_size):
                item = {
                    "coords": batch["coords"][i],
                    "demands": batch["demands"][i],
                }
                if batch.get("time_windows") is not None:
                    item["time_windows"] = batch["time_windows"][i]
                else:
                    item["time_windows"] = None
                if batch.get("service_times") is not None:
                    item["service_times"] = batch["service_times"][i]
                else:
                    item["service_times"] = torch.zeros(self.n + 1)
                yield item
                generated += 1
                if generated >= self.n_instances:
                    break
