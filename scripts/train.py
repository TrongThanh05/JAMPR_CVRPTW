"""Training script for JAMPR VRPTW model.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --debug
    python scripts/train.py --config configs/training_config.yaml --resume outputs/checkpoints/best.pt
"""

import argparse
import logging
import os
import sys

import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logging
from src.models.jampr import JAMPRModel
from src.data.generator import VRPTWDataGenerator
from src.data.dataset import VRPTWDataset
from src.training.trainer import Trainer


def run_training(args):
    """Main training function."""
    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config_path = args.model_config or "configs/model_config.yaml"
    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Merge configs
    config.update(model_cfg)

    # Debug mode overrides
    if args.debug:
        debug_cfg = config.get("training", {}).get("debug", {})
        for key, val in debug_cfg.items():
            config["training"][key] = val
        config["training"]["checkpoint_dir"] = "outputs/checkpoints/debug"

    # Set problem and n_customers
    config["training"]["problem"] = args.problem
    config["training"]["n_customers"] = args.n

    # Set m_con from model config
    mc = config.get("model", {})
    m_con_key = f"{args.problem}_n{args.n}"
    m_con_settings = mc.get("m_con_settings", {})
    if m_con_key in m_con_settings:
        mc["m_con"] = m_con_settings[m_con_key]

    # Setup root logger so all module loggers output to console
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    seed = config["training"].get("seed", 1234)
    set_seed(seed)

    log_dir = config["training"].get("log_dir", "outputs/logs")
    experiment = f"{args.problem}_n{args.n}"
    logger = setup_logging(log_dir, experiment)
    logger.info("Config: %s", config)

    # Model
    model = JAMPRModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model created with %d parameters", n_params)

    # Data
    train_gen = VRPTWDataGenerator()

    # Validation dataset
    val_path = os.path.join("outputs/data", f"{args.problem}_n{args.n}_val.pt")
    val_dataset = None
    if os.path.exists(val_path):
        val_dataset = VRPTWDataset(val_path)
        logger.info("Loaded validation dataset: %s", val_path)

    # Trainer
    trainer = Trainer(model, config, train_gen, val_dataset)

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train JAMPR VRPTW model")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--problem", type=str, default="cvrptw_tw1",
                        choices=["cvrp", "cvrptw_tw1", "cvrptw_tw2", "cvrptw_tw3"])
    parser.add_argument("--n", type=int, default=20, choices=[20, 50])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
