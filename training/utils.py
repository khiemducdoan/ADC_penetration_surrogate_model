"""Data splitting, normalization, and DataLoader helpers for training.

Split is a plain random split at the row level (70/15/15 by default): every
row already corresponds to one (params, t) pair, so rows are i.i.d. given the
sampling scheme in data/synthetic/generate.py.

Targets are normalized in log-space (log(C + eps), then standardized) since
concentration spans many orders of magnitude.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def normalize_inputs(X: np.ndarray, train_idx: np.ndarray):
    mean, std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-12
    return (X - mean) / std, mean, std


def normalize_targets(Y: np.ndarray, train_idx: np.ndarray, eps: float = EPS):
    Ylog = np.log(Y + eps)
    mean, std = Ylog[train_idx].mean(), Ylog[train_idx].std() + 1e-12
    return (Ylog - mean) / std, Ylog, mean, std


def make_loader(X: np.ndarray, Y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X[idx], dtype=torch.float32),
        torch.tensor(Y[idx], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
