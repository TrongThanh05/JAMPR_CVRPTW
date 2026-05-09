"""Evaluation script for JAMPR VRPTW model.

Usage:
    python scripts/evaluate.py --model outputs/checkpoints/best.pt
    python scripts/evaluate.py --model outputs/checkpoints/best.pt --problem cvrptw_tw1 --n 20
"""

import argparse
import logging
import os
import sys

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logging
from src.models.jampr import JAMPRModel
from src.data.dataset import VRPTWDataset
from src.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate JAMPR model")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--problem", type=str, default="cvrptw_tw1")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--data", type=str, default=None, help="Path to test data .pt file")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--n_samples", type=int, default=1280)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, weights_only=False, map_location=device)

    config = checkpoint.get("config", {})
    if not config:
        # Fallback: load from files
        with open("configs/model_config.yaml") as f:
            config = yaml.safe_load(f)
        with open("configs/training_config.yaml") as f:
            config.update(yaml.safe_load(f))

    config.setdefault("training", {})["problem"] = args.problem

    # Create model
    model = JAMPRModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded model from %s (epoch %d)", args.model, checkpoint.get("epoch", "?"))

    # Load test data
    if args.data:
        data_path = args.data
    else:
        data_path = f"outputs/data/{args.problem}_n{args.n}_test.pt"

    if not os.path.exists(data_path):
        logger.warning("Test data not found at %s, generating small dataset...", data_path)
        from src.data.generator import VRPTWDataGenerator
        gen = VRPTWDataGenerator()
        data = gen.generate_batch(args.problem, args.n, 100, seed=9999)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save(data, data_path)

    dataset = VRPTWDataset(data_path)

    # Evaluate
    evaluator = Evaluator(model, config, device)

    if args.mode == "greedy":
        results = evaluator.evaluate_greedy(dataset)
    else:
        results = evaluator.evaluate_sampling(dataset, n_samples=args.n_samples)

    logger.info("Results: %s", results)

    # Print benchmark table
    table = evaluator.benchmark(dataset)
    print(table)


if __name__ == "__main__":
    main()
