"""Data splitting, normalization, and DataLoader helpers for training.

Split is group-aware (70/15/15 by default): the n_times rows of a single
(c0, D, r) condition are correlated (same trajectory, different t only), so
splitting at the row level would leak a condition's dynamics into training
via other t values of the very condition later shown at test time, and
inflate test R2 into pure time-interpolation rather than a genuine test of
generalization to unseen physical conditions. Splitting whole condition
groups into train/val/test avoids that.

Targets are normalized in log-space (log(C + eps), then standardized) since
concentration spans many orders of magnitude.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-8


def split_indices(n: int, seed: int, fracs=(0.70, 0.15, 0.15), groups: np.ndarray | None = None):
    """Split `n` row indices into train/val/test.

    If `groups` is given (one group id per row, e.g. condition_id), the split
    is done at the group level so all rows of a group land in the same split
    - required whenever rows within a group are correlated. Otherwise falls
    back to a plain row-level random split.
    """
    rng = np.random.default_rng(seed)

    if groups is None:
        idx = rng.permutation(n)
        n_train = int(fracs[0] * n)
        n_val = int(fracs[1] * n)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    unique_groups = rng.permutation(np.unique(groups))
    n_g = len(unique_groups)
    n_train_g = int(fracs[0] * n_g)
    n_val_g = int(fracs[1] * n_g)
    train_groups = set(unique_groups[:n_train_g])
    val_groups = set(unique_groups[n_train_g:n_train_g + n_val_g])
    test_groups = set(unique_groups[n_train_g + n_val_g:])

    row_idx = np.arange(n)
    train_idx = row_idx[np.isin(groups, list(train_groups))]
    val_idx = row_idx[np.isin(groups, list(val_groups))]
    test_idx = row_idx[np.isin(groups, list(test_groups))]
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
