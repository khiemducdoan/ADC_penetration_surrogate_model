"""
Evaluate a trained surrogate: reload the exact test split saved in the
checkpoint, recompute MAE/RMSE/R2, and plot true-vs-predicted profiles.

Usage:
    python evaluate.py --dataset ../../outputs/dataset.npz \
        --checkpoint ../../outputs/surrogate_model.pt --out_dir ../../outputs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import MLPSurrogate
from train import compute_metrics


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLPSurrogate(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dims=ckpt["hidden_dims"],
        activation=ckpt["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(model, ckpt, X_raw: np.ndarray, device: torch.device) -> np.ndarray:
    Xn = (X_raw - ckpt["x_mean"]) / ckpt["x_std"]
    with torch.no_grad():
        pred_n = model(torch.tensor(Xn, dtype=torch.float32).to(device)).cpu().numpy()
    pred_log = pred_n * ckpt["y_std"] + ckpt["y_mean"]
    return np.exp(pred_log) - ckpt["eps"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="../../outputs/dataset.npz")
    parser.add_argument("--checkpoint", type=str, default="../../outputs/surrogate_model.pt")
    parser.add_argument("--out_dir", type=str, default="../../outputs/figures")
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data = np.load(args.dataset)
    X, Y, x_grid = data["X"], data["Y"], data["x_grid"]

    model, ckpt = load_checkpoint(args.checkpoint, device)
    test_idx = ckpt["test_idx"]

    pred_test = predict(model, ckpt, X[test_idx], device)
    y_true, y_pred = Y[test_idx], pred_test

    metrics_phys = compute_metrics(y_true, y_pred)
    metrics_log = compute_metrics(np.log(y_true + ckpt["eps"]), np.log(y_pred + ckpt["eps"]))
    print("Test metrics (physical units):", metrics_phys)
    print("Test metrics (log-space):     ", metrics_log)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots (metrics above are still valid).")
        return

    rng = np.random.default_rng(0)
    example_idx = rng.choice(len(test_idx), size=min(args.n_examples, len(test_idx)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, i in zip(axes.ravel(), example_idx):
        log_c0, log_D, log_r, t = X[test_idx][i]
        ax.plot(x_grid, y_true[i], label="true (analytical)", lw=2)
        ax.plot(x_grid, y_pred[i], "--", label="NN prediction", lw=2)
        ax.set_title(f"c0=10^{log_c0:.1f}, D=10^{log_D:.1f}, r=10^{log_r:.1f}\nt={t:.2e} s")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("C(x,t)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "profiles_true_vs_pred.png", dpi=150)
    print(f"Saved {out_dir / 'profiles_true_vs_pred.png'}")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sample = rng.choice(y_true.size, size=min(20000, y_true.size), replace=False)
    ax2.scatter(y_true.ravel()[sample], y_pred.ravel()[sample], s=2, alpha=0.3)
    lims = [0, max(y_true.max(), y_pred.max())]
    ax2.plot(lims, lims, "r--", lw=1)
    ax2.set_xlabel("true C(x,t)")
    ax2.set_ylabel("predicted C(x,t)")
    ax2.set_title(f"R2 = {metrics_phys['R2']:.4f}")
    fig2.tight_layout()
    fig2.savefig(out_dir / "scatter_true_vs_pred.png", dpi=150)
    print(f"Saved {out_dir / 'scatter_true_vs_pred.png'}")


if __name__ == "__main__":
    main()
