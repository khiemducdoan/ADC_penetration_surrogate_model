"""
Entry point: train the MLP surrogate (Hydra-configured), logging to Weights
& Biases when cfg.wandb.enabled is true.

Usage:
    python train.py
    python train.py training.epochs=500 training.hidden_dims=[256,256,256,256]
    python train.py wandb.enabled=false
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from models.losses import get_loss
from models.metrics import compute_metrics
from models.mlp import MLPSurrogate
from training.trainer import Trainer
from training.utils import EPS, make_loader, normalize_inputs, normalize_targets, split_indices
from utils.seed import set_seed

try:
    import wandb
except ImportError:
    wandb = None


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(cfg.dataset_path)
    X, Y = data["X"], data["Y"]
    n = X.shape[0]

    train_idx, val_idx, test_idx = split_indices(n, cfg.training.seed, tuple(cfg.training.split))
    Xn, x_mean, x_std = normalize_inputs(X, train_idx)
    Yn, Ylog, y_mean, y_std = normalize_targets(Y, train_idx)

    train_loader = make_loader(Xn, Yn, train_idx, cfg.training.batch_size, shuffle=True)
    val_loader = make_loader(Xn, Yn, val_idx, cfg.training.batch_size, shuffle=False)

    model = MLPSurrogate(
        input_dim=4,
        output_dim=Y.shape[1],
        hidden_dims=tuple(cfg.training.hidden_dims),
        activation=cfg.training.activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = get_loss(cfg.training.loss)

    use_wandb = bool(cfg.wandb.enabled) and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = Trainer(
        model, optimizer, loss_fn, train_loader, val_loader,
        device=cfg.training.device, patience=cfg.training.patience, use_wandb=use_wandb,
    )
    history = trainer.fit(cfg.training.epochs)

    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(Xn[test_idx], dtype=torch.float32).to(trainer.device)
        pred_test_n = model(X_test).cpu().numpy()
    pred_log = pred_test_n * y_std + y_mean
    pred_phys = np.exp(pred_log) - EPS

    metrics_log = compute_metrics(Ylog[test_idx], pred_log)
    metrics_phys = compute_metrics(Y[test_idx], pred_phys)
    print("Test metrics (log-space):   ", metrics_log)
    print("Test metrics (physical units):", metrics_phys)

    if use_wandb:
        wandb.log({f"test/log_{k}": v for k, v in metrics_log.items()})
        wandb.log({f"test/phys_{k}": v for k, v in metrics_phys.items()})
        wandb.finish()

    checkpoint = {
        "model_state": model.state_dict(),
        "hidden_dims": tuple(cfg.training.hidden_dims),
        "activation": cfg.training.activation,
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
