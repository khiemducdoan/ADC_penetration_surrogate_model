"""Physics-consistency and regime-conditional checks for the surrogate,
beyond plain regression metrics (MAE/RMSE/R2 in models/metrics.py).

Plain aggregate metrics can look good while hiding two kinds of failure a
physical surrogate must not have: (1) error concentrated in a specific
region of x, t, or (D,r) space, and (2) predictions that violate physics
the network was never explicitly told about (boundary conditions,
non-negativity, monotonic shape, the known steady-state limit).
"""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import numpy as np

from data.synthetic.solver import steady_state_profile


def error_by_time_regime(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 3) -> dict:
    """Mean absolute error binned by `t` (column 3 of X) into `n_bins`
    quantile groups — reveals whether early transient or near-steady-state
    profiles are harder to predict."""
    t = X[:, 3]
    edges = np.quantile(t, np.linspace(0, 1, n_bins + 1))
    result = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (t >= lo) & (t <= hi if i == n_bins - 1 else t < hi)
        if mask.sum() == 0:
            continue
        err = np.abs(y_true[mask] - y_pred[mask])
        result[f"t_bin{i}_[{lo:.1e},{hi:.1e}]"] = float(err.mean())
    return result


def error_by_penetration_depth(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, L: float, n_bins: int = 3) -> dict:
    """Mean absolute error binned by lambda/L = sqrt(D/r)/L — reveals whether
    steep (small lambda, fast decay) or flat (large lambda) profiles are
    harder to predict."""
    log_D, log_r = X[:, 1], X[:, 2]
    lam_over_L = np.sqrt(10.0 ** log_D / 10.0 ** log_r) / L
    edges = np.quantile(lam_over_L, np.linspace(0, 1, n_bins + 1))
    result = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (lam_over_L >= lo) & (lam_over_L <= hi if i == n_bins - 1 else lam_over_L < hi)
        if mask.sum() == 0:
            continue
        err = np.abs(y_true[mask] - y_pred[mask])
        result[f"lambdaL_bin{i}_[{lo:.2f},{hi:.2f}]"] = float(err.mean())
    return result


def check_boundary_conditions(X: np.ndarray, y_pred: np.ndarray) -> dict:
    """Does the surrogate respect the two boundary conditions it was never
    explicitly told about? Dirichlet: C(x=0,t) should equal c0. Neumann:
    the profile should be flat (zero flux) near x=L."""
    c0 = 10.0 ** X[:, 0]
    left_val = y_pred[:, 0]
    left_rel_err = np.abs(left_val - c0) / np.maximum(c0, 1e-12)

    flux_L = y_pred[:, -1] - y_pred[:, -2]
    flux_scale = np.maximum(np.abs(y_pred[:, 0]), 1e-12)

    return {
        "bc_dirichlet_mean_rel_err": float(left_rel_err.mean()),
        "bc_dirichlet_max_rel_err": float(left_rel_err.max()),
        "bc_neumann_mean_abs_flux": float(np.abs(flux_L / flux_scale).mean()),
    }


def check_non_negativity(y_pred: np.ndarray) -> dict:
    """Concentration is a physical quantity and can never be negative."""
    return {
        "frac_negative": float((y_pred < 0).mean()),
        "min_value": float(y_pred.min()),
    }


def check_monotonicity(y_pred: np.ndarray, tol_frac: float = 0.02) -> dict:
    """Fraction of predicted profiles that are NOT monotonically
    non-increasing from x=0 to x=L — the expected shape for this
    diffusion-decay problem (source at x=0, decaying outward). Tolerance is
    relative to each profile's own peak value, since concentration scale
    varies row to row with c0."""
    diffs = np.diff(y_pred, axis=1)
    scale = np.maximum(np.abs(y_pred).max(axis=1, keepdims=True), 1e-12)
    violations = (diffs > tol_frac * scale).any(axis=1)
    return {"frac_non_monotonic": float(violations.mean())}


def check_steady_state_limit(X: np.ndarray, y_pred: np.ndarray, x_grid: np.ndarray, L: float, t_quantile: float = 0.9) -> dict:
    """For rows whose t falls in the top decile (closest to steady state),
    compare the prediction against the true closed-form steady state —
    something the network was never given directly."""
    t = X[:, 3]
    threshold = np.quantile(t, t_quantile)
    mask = t >= threshold
    if mask.sum() == 0:
        return {}
    errs = []
    for i in np.where(mask)[0]:
        log_c0, log_D, log_r, _ = X[i]
        c0, D, r = 10.0 ** log_c0, 10.0 ** log_D, 10.0 ** log_r
        c_ss = steady_state_profile(x_grid, D, r, c0, L)
        errs.append(np.max(np.abs(y_pred[i] - c_ss)) / max(np.max(c_ss), 1e-12))
    return {"steady_state_max_rel_err": float(np.mean(errs))}


def benchmark_inference_speed(model, X_sample: np.ndarray, device: str = "cpu", n_repeats: int = 5) -> dict:
    """Surrogate throughput via repeated forward passes — the actual payoff
    metric for a surrogate (how much faster than the physics solver it
    replaces)."""
    import torch

    model.eval()
    X_t = torch.tensor(X_sample, dtype=torch.float32).to(device)
    with torch.no_grad():
        model(X_t)  # warmup
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model(X_t)
            times.append(time.perf_counter() - t0)
    best = min(times)
    return {
        "inference_time_s_per_batch": best,
        "inference_throughput_profiles_per_s": X_sample.shape[0] / best,
    }


def _expand_range(rng, factor: float):
    lo, hi = rng
    lo_log, hi_log = np.log10(lo), np.log10(hi)
    center, half = (lo_log + hi_log) / 2, (hi_log - lo_log) / 2 * factor
    return [10.0 ** (center - half), 10.0 ** (center + half)]


def extrapolation_gap(evaluator, sim_cfg, sampling_cfg, in_range_mae: float, expand_factor: float = 1.5, n_conditions: int = 200) -> dict:
    """Sample fresh conditions with c0/D/r ranges expanded beyond what the
    model was trained on (ground truth is still exact — the analytical
    solver is valid for any (c0,D,r), only the surrogate's training data was
    restricted). Reports how much worse the surrogate gets outside its
    training domain, relative to its in-range error."""
    from data.synthetic.generate import build_dataset

    expanded_cfg = copy.deepcopy(sim_cfg)
    expanded_cfg = SimpleNamespace(
        c0_range=_expand_range(sim_cfg.c0_range, expand_factor),
        D_range=_expand_range(sim_cfg.D_range, expand_factor),
        r_range=_expand_range(sim_cfg.r_range, expand_factor),
        L=sim_cfg.L, nx=sim_cfg.nx, t_max=sim_cfg.t_max, n_modes=sim_cfg.n_modes,
    )
    sampling_expanded = SimpleNamespace(
        n_conditions=n_conditions, n_times=4, seed=123, validate_frac=0.0,
    )
    X_ext, Y_ext, _, _, _ = build_dataset(expanded_cfg, sampling_expanded)

    y_pred_ext = evaluator.predict(X_ext)
    mae_ext = float(np.mean(np.abs(Y_ext - y_pred_ext)))

    return {
        "extrapolation_mae": mae_ext,
        "extrapolation_gap_ratio": mae_ext / max(in_range_mae, 1e-12),
    }


def run_all_checks(evaluator, X: np.ndarray, Y: np.ndarray, y_pred: np.ndarray, x_grid: np.ndarray, L: float, sim_cfg=None, sampling_cfg=None) -> dict:
    """Run every check above on an already-computed (X, Y, y_pred) test set
    and return one flat dict of metrics, prefixed by check name."""
    metrics = {}
    for k, v in error_by_time_regime(X, Y, y_pred).items():
        metrics[f"regime/{k}"] = v
    for k, v in error_by_penetration_depth(X, Y, y_pred, L).items():
        metrics[f"regime/{k}"] = v
    for k, v in check_boundary_conditions(X, y_pred).items():
        metrics[f"physics/{k}"] = v
    for k, v in check_non_negativity(y_pred).items():
        metrics[f"physics/{k}"] = v
    for k, v in check_monotonicity(y_pred).items():
        metrics[f"physics/{k}"] = v
    for k, v in check_steady_state_limit(X, y_pred, x_grid, L).items():
        metrics[f"physics/{k}"] = v

    if sim_cfg is not None and sampling_cfg is not None:
        in_range_mae = float(np.mean(np.abs(Y - y_pred)))
        for k, v in extrapolation_gap(evaluator, sim_cfg, sampling_cfg, in_range_mae).items():
            metrics[f"extrapolation/{k}"] = v

    return metrics
