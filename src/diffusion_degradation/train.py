"""
Train the MLP surrogate on the synthetic dataset produced by generate_data.py.

Split: 70% train / 15% validation / 15% test (fixed seed, split by condition
sample index — every row already corresponds to one (params, t) pair, so a
plain random split at the row level is fine here since rows are i.i.d. given
the sampling scheme in generate_data.py).

Targets are trained in log-space: log(C + eps), then standardized. Metrics
are reported both in log-space and after inverting back to physical
concentration units.

Usage:
    python train.py --dataset ../../outputs/dataset.npz --epochs 200 \
        --hidden_dims 256,256,256 --out_dir ../../outputs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MLPSurrogate

EPS = 1e-8


def split_indices(n: int, seed: int, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="../../outputs/dataset.npz")
    parser.add_argument("--hidden_dims", type=str, default="256,256,256")
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="../../outputs")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    X, Y = data["X"], data["Y"]
    n = X.shape[0]
    train_idx, val_idx, test_idx = split_indices(n, args.seed)

    x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-12
    Xn = (X - x_mean) / x_std

    Ylog = np.log(Y + EPS)
    y_mean, y_std = Ylog[train_idx].mean(), Ylog[train_idx].std() + 1e-12
    Yn = (Ylog - y_mean) / y_std

    device = torch.device(args.device)

    def to_loader(idx, batch_size, shuffle):
        ds = TensorDataset(
            torch.tensor(Xn[idx], dtype=torch.float32),
            torch.tensor(Yn[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_idx, args.batch_size, True)
    val_loader = to_loader(val_idx, args.batch_size, False)

    hidden_dims = tuple(int(h) for h in args.hidden_dims.split(","))
    model = MLPSurrogate(input_dim=4, output_dim=Y.shape[1],
                          hidden_dims=hidden_dims, activation=args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"Training took {time.time() - t0:.1f} s, best val_loss={best_val:.5f}")
    model.load_state_dict(best_state)

    # ---- test-set evaluation ----
    model.eval()
    with torch.no_grad():
        pred_test_n = model(torch.tensor(Xn[test_idx], dtype=torch.float32).to(device)).cpu().numpy()
    pred_log = pred_test_n * y_std + y_mean
    pred_phys = np.exp(pred_log) - EPS

    metrics_log = compute_metrics(Ylog[test_idx], pred_log)
    metrics_phys = compute_metrics(Y[test_idx], pred_phys)
    print("Test metrics (log-space):   ", metrics_log)
    print("Test metrics (physical units):", metrics_phys)

    checkpoint = {
        "model_state": best_state,
        "hidden_dims": hidden_dims,
        "activation": args.activation,
        "input_dim": 4,
        "output_dim": Y.shape[1],
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "eps": EPS,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }
    torch.save(checkpoint, out_dir / "surrogate_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"log_space": metrics_log, "physical_units": metrics_phys}, f, indent=2)

    print(f"Saved checkpoint to {out_dir / 'surrogate_model.pt'}")


if __name__ == "__main__":
    main()
