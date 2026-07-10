"""Training callbacks."""
from __future__ import annotations


class EarlyStopping:
    """Tracks the best validation loss and the model weights at that point,
    and signals when training should stop."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_state: dict | None = None
        self.epochs_no_improve = 0
        self.should_stop = False

    def step(self, val_loss: float, model) -> None:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
