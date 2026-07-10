"""
Entry point: evaluate a trained surrogate against its dataset's test split
(Hydra-configured).

Usage:
    python evaluate.py
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from evaluation.evaluator import Evaluator
from evaluation.visualizer import plot_profiles, plot_scatter


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = np.load(cfg.dataset_path)
    X, Y, x_grid = data["X"], data["Y"], data["x_grid"]

    evaluator = Evaluator(cfg.checkpoint_path, device=cfg.training.device)
    y_true, y_pred, metrics_phys, metrics_log = evaluator.evaluate(X, Y)
    print("Test metrics (physical units):", metrics_phys)
    print("Test metrics (log-space):     ", metrics_log)

    figures_dir = Path(cfg.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    test_idx = evaluator.ckpt["test_idx"]

    try:
        path1 = plot_profiles(x_grid, X[test_idx], y_true, y_pred, figures_dir / "profiles_true_vs_pred.png")
        print(f"Saved {path1}")
        path2 = plot_scatter(y_true, y_pred, metrics_phys["R2"], figures_dir / "scatter_true_vs_pred.png")
        print(f"Saved {path2}")
    except ImportError:
        print("matplotlib not installed - skipping plots (metrics above are still valid).")


if __name__ == "__main__":
    main()
