"""Evaluator: reloads a trained checkpoint and scores it against a dataset's
saved test split.
"""
from __future__ import annotations

import numpy as np
import torch

from models.metrics import compute_metrics
from models.mlp import MLPSurrogate


class Evaluator:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = MLPSurrogate(
            input_dim=self.ckpt["input_dim"],
            output_dim=self.ckpt["output_dim"],
            hidden_dims=self.ckpt["hidden_dims"],
            activation=self.ckpt["activation"],
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.eval()

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        ckpt = self.ckpt
        Xn = (X_raw - ckpt["x_mean"]) / ckpt["x_std"]
        with torch.no_grad():
            pred_n = self.model(torch.tensor(Xn, dtype=torch.float32).to(self.device)).cpu().numpy()
        pred_log = pred_n * ckpt["y_std"] + ckpt["y_mean"]
        return np.exp(pred_log) - ckpt["eps"]

    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        """Predict and score on the checkpoint's saved test split. Returns
        (y_true, y_pred, metrics_physical, metrics_log)."""
        test_idx = self.ckpt["test_idx"]
        y_true = Y[test_idx]
        y_pred = self.predict(X[test_idx])

        metrics_phys = compute_metrics(y_true, y_pred)
        metrics_log = compute_metrics(
            np.log(y_true + self.ckpt["eps"]), np.log(y_pred + self.ckpt["eps"])
        )
        return y_true, y_pred, metrics_phys, metrics_log
