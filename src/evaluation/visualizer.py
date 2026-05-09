"""Visualization utilities: route plotting and learning curves."""

import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Color palette for routes
COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
          '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990',
          '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3']


def plot_routes(instance: dict, solution: list, title: str = "VRPTW Routes",
                save_path: str = "routes.png") -> None:
    """Plot vehicle routes with depot as star and customers as dots.

    Args:
        instance: Dict with 'coords' (N+1, 2) tensor.
        solution: List of tours (list of node indices).
        title: Plot title.
        save_path: Path to save the figure.
    """
    coords = instance["coords"]
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot depot
    ax.scatter(coords[0, 0], coords[0, 1], c='red', s=200, marker='*',
              zorder=5, label='Depot')

    # Plot customers
    ax.scatter(coords[1:, 0], coords[1:, 1], c='gray', s=50, alpha=0.7, zorder=3)

    # Label nodes
    for i in range(len(coords)):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]),
                   fontsize=7, ha='center', va='bottom')

    # Plot tours
    for k, tour in enumerate(solution):
        color = COLORS[k % len(COLORS)]
        route_coords = [coords[0]]  # start at depot
        for node in tour:
            route_coords.append(coords[node])
        route_coords.append(coords[0])  # return to depot
        route_coords = np.array(route_coords)

        ax.plot(route_coords[:, 0], route_coords[:, 1], '-o',
               color=color, markersize=4, linewidth=1.5, alpha=0.8,
               label=f'Tour {k+1} ({len(tour)} nodes)')

    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Route plot saved to %s", save_path)


def plot_learning_curves(log_dir: str, save_path: str = "learning_curves.png") -> None:
    """Plot training cost vs steps from TensorBoard log directory.

    Args:
        log_dir: Path to TensorBoard log directory.
        save_path: Path to save the figure.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training cost
        if 'train/cost' in ea.Tags()['scalars']:
            events = ea.Scalars('train/cost')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            axes[0].plot(steps, values, 'b-', alpha=0.7)
            axes[0].set_title('Training Cost')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Cost')
            axes[0].grid(True, alpha=0.3)

        # Learning rate
        if 'train/lr' in ea.Tags()['scalars']:
            events = ea.Scalars('train/lr')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            axes[1].plot(steps, values, 'r-', alpha=0.7)
            axes[1].set_title('Learning Rate')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('LR')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Learning curves saved to %s", save_path)

    except Exception as e:
        logger.warning("Could not plot learning curves: %s", e)
