"""Plotting helpers: true-vs-predicted profiles and a global scatter plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_profiles(x_grid, X, y_true, y_pred, out_path: Path, n_examples: int = 6, seed: int = 0) -> Path:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    example_idx = rng.choice(len(y_true), size=min(n_examples, len(y_true)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, i in zip(axes.ravel(), example_idx):
        log_c0, log_D, log_r, t = X[i]
        ax.plot(x_grid, y_true[i], label="true (analytical)", lw=2)
        ax.plot(x_grid, y_pred[i], "--", label="NN prediction", lw=2)
        ax.set_title(f"c0=10^{log_c0:.1f}, D=10^{log_D:.1f}, r=10^{log_r:.1f}\nt={t:.2e} s")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("C(x,t)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_scatter(y_true, y_pred, r2: float, out_path: Path, seed: int = 0, max_points: int = 20000) -> Path:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    sample = rng.choice(y_true.size, size=min(max_points, y_true.size), replace=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true.ravel()[sample], y_pred.ravel()[sample], s=2, alpha=0.3)
    lims = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("true C(x,t)")
    ax.set_ylabel("predicted C(x,t)")
    ax.set_title(f"R2 = {r2:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path
