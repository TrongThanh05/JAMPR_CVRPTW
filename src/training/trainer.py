"""Trainer: orchestrates REINFORCE training loop with baseline and scheduling."""

import os
import logging
import yaml

import torch
from torch.nn.utils import clip_grad_norm_

from src.training.reinforce import compute_reinforce_loss
from src.training.baseline import RolloutBaseline
from src.training.scheduler import InverseLRScheduler
from src.evaluation.metrics import compute_cost
from src.utils.logging_utils import TBWriter, AverageMeter
from src.utils.time_utils import Timer

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for JAMPR model.

    Args:
        model: JAMPRModel instance.
        config: Full config dict (model + training).
        train_generator: VRPTWDataGenerator for on-the-fly data.
        val_dataset: Validation dataset.
    """

    def __init__(self, model, config: dict, train_generator, val_dataset):
        self.model = model
        self.config = config
        self.train_generator = train_generator
        self.val_dataset = val_dataset

        tc = config.get("training", config)
        self.lr_init = tc.get("lr_init", 1e-4)
        self.grad_clip = tc.get("grad_clip", 1.0)
        self.n_epochs = tc.get("n_epochs", 50)
        self.instances_per_epoch = tc.get("instances_per_epoch", 1024000)
        self.save_every = tc.get("save_every", 5)
        self.log_interval = tc.get("log_interval", 50)
        self.checkpoint_dir = tc.get("checkpoint_dir", "outputs/checkpoints")
        self.log_dir = tc.get("log_dir", "outputs/logs")
        self.problem = tc.get("problem", "cvrptw_tw1")
        self.n_customers = tc.get("n_customers", 20)
        self.seed = tc.get("seed", 1234)

        # Batch size depends on N
        if self.n_customers <= 20:
            self.batch_size = tc.get("batch_size_n20", 512)
        else:
            self.batch_size = tc.get("batch_size_n50", 128)

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)

        # Scheduler
        gamma = tc.get("lr_decay_gamma", 0.001)
        self.scheduler = InverseLRScheduler(self.optimizer, gamma=gamma)

        # Baseline
        self.baseline = RolloutBaseline(self.model, config)

        # TensorBoard
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_writer = TBWriter(self.log_dir)

        # Checkpoint dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.global_step = 0

    def train(self) -> None:
        """Run full training loop."""
        logger.info("Starting training: %d epochs, %d instances/epoch, batch=%d",
                    self.n_epochs, self.instances_per_epoch, self.batch_size)

        for epoch in range(1, self.n_epochs + 1):
            metrics = self.train_epoch(epoch)
            val_cost = self.validate()

            # Update baseline
            if self.val_dataset is not None:
                self.baseline.update(self.model, self.val_dataset)

            # LR schedule
            self.scheduler.step(epoch)

            # Log
            self.tb_writer.scalar("train/cost", metrics["cost_mean"], epoch)
            self.tb_writer.scalar("val/cost", val_cost, epoch)
            self.tb_writer.scalar("train/lr", self.scheduler.get_lr(), epoch)

            logger.info("Epoch %d/%d | cost=%.4f | val=%.4f | lr=%.3e",
                       epoch, self.n_epochs, metrics["cost_mean"], val_cost,
                       self.scheduler.get_lr())

            # Checkpoint
            if epoch % self.save_every == 0 or epoch == self.n_epochs:
                self.save_checkpoint(epoch, metrics)

        self.tb_writer.close()
        logger.info("Training complete.")

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict with cost_mean, cost_std, loss.
        """
        self.model.train()
        cost_meter = AverageMeter()
        loss_meter = AverageMeter()

        n_batches = max(1, self.instances_per_epoch // self.batch_size)

        for batch_idx in range(n_batches):
            # Generate batch on-the-fly
            batch = self.train_generator.generate_batch(
                self.problem, self.n_customers, self.batch_size
            )
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass (sampling)
            solutions, log_probs = self.model(batch, greedy=False)

            # Compute costs
            costs = compute_cost(solutions, batch, self.problem)
            costs = costs.to(self.device)

            # Clamp inf/NaN costs to a large finite value to prevent gradient corruption
            max_finite = 100.0  # reasonable upper bound for normalized VRPTW cost
            costs = torch.where(torch.isfinite(costs), costs,
                               torch.full_like(costs, max_finite))

            # Baseline costs
            baseline_costs = self.baseline.eval(batch)
            baseline_costs = baseline_costs.to(self.device)
            baseline_costs = torch.where(torch.isfinite(baseline_costs), baseline_costs,
                                        torch.full_like(baseline_costs, max_finite))

            # REINFORCE loss
            loss = compute_reinforce_loss(log_probs, costs, baseline_costs)

            # Skip step if loss is NaN/inf (protect weights)
            if not torch.isfinite(loss):
                logger.warning("  Batch %d: loss is %s, skipping update", batch_idx + 1, loss.item())
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # Track metrics
            cost_meter.update(costs.mean().item(), self.batch_size)
            loss_meter.update(loss.item(), 1)
            self.global_step += 1

            if (batch_idx + 1) % self.log_interval == 0:
                logger.info("  Batch %d/%d | cost=%.4f | loss=%.4f",
                           batch_idx + 1, n_batches, cost_meter.avg, loss_meter.avg)

        return {
            "cost_mean": cost_meter.avg,
            "cost_std": 0.0,  # simplified
            "loss": loss_meter.avg,
        }

    def validate(self) -> float:
        """Run validation and return mean cost.

        Returns:
            Mean validation cost.
        """
        if self.val_dataset is None:
            return 0.0

        self.model.eval()
        costs_list = []

        with torch.no_grad():
            n_eval = min(len(self.val_dataset) if hasattr(self.val_dataset, '__len__') else 10, 10)
            for i in range(n_eval):
                if hasattr(self.val_dataset, '__getitem__'):
                    item = self.val_dataset[i]
                    batch = {}
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.unsqueeze(0).to(self.device)
                        else:
                            batch[k] = v
                    solutions, _ = self.model(batch, greedy=True)
                    cost = compute_cost(solutions, batch, self.problem)
                    costs_list.append(cost.item())

        self.model.train()
        if costs_list:
            return sum(costs_list) / len(costs_list)
        return 0.0

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save training checkpoint.

        Filename: {problem}_{n}_mcon{mcon}_epoch{epoch:03d}.pt
        """
        mc = self.config.get("model", self.config)
        m_con = mc.get("m_con", 3)
        filename = f"{self.problem}_{self.n_customers}_mcon{m_con}_epoch{epoch:03d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return epoch number.

        Args:
            path: Path to checkpoint file.

        Returns:
            Epoch number from checkpoint.
        """
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        logger.info("Loaded checkpoint from %s (epoch %d)", path, epoch)
        return epoch
