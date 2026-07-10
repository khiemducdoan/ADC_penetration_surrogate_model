"""Trainer: runs the surrogate's training loop with early stopping and
optional Weights & Biases logging.
"""
from __future__ import annotations

import time

import torch

from training.callbacks import EarlyStopping

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        device: str = "cpu",
        patience: int = 20,
        use_wandb: bool = False,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.early_stopping = EarlyStopping(patience=patience)
        self.use_wandb = use_wandb and wandb is not None
        self.history: list[dict] = []

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total_loss, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            with torch.set_grad_enabled(train):
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        return total_loss / n

    def fit(self, epochs: int) -> list[dict]:
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if epoch % 10 == 0 or epoch == 1:
                print(f"epoch {epoch:4d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

            self.early_stopping.step(val_loss, self.model)
            if self.early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {self.early_stopping.patience} epochs)")
                break

        print(f"Training took {time.time() - t0:.1f} s, best val_loss={self.early_stopping.best:.5f}")
        self.model.load_state_dict(self.early_stopping.best_state)
        return self.history
