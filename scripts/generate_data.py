"""Script to generate and save validation/test datasets to disk."""

import os
import sys
import argparse
import logging

# Add project root to path so `src` is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.generator import VRPTWDataGenerator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate VRPTW data")
    parser.add_argument("--problem", type=str, required=True,
                        choices=["cvrp", "cvrptw_tw1", "cvrptw_tw2", "cvrptw_tw3"],
                        help="Problem type")
    parser.add_argument("--n", type=int, required=True, choices=[20, 50],
                        help="Number of customers")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Data split")
    parser.add_argument("--size", type=int, default=10000,
                        help="Number of instances")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="outputs/data/",
                        help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    generator = VRPTWDataGenerator()
    seed = args.seed

    logger.info("Generating %d %s instances (n=%d, split=%s)",
                args.size, args.problem, args.n, args.split)

    data = generator.generate_batch(args.problem, args.n, args.size, seed=seed)

    os.makedirs(args.output, exist_ok=True)
    filename = f"{args.problem}_n{args.n}_{args.split}.pt"
    save_path = os.path.join(args.output, filename)
    torch.save(data, save_path)
    logger.info("Saved to %s", save_path)


if __name__ == "__main__":
    main()
