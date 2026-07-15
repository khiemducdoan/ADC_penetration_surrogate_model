"""
Synthetic dataset generation for the 1D diffusion-degradation surrogate.

Samples (c0, D, r) log-uniformly (per the `sampling` config), evaluates the
analytical transient solution at several log-spaced times per condition (per
the `simulation` config), and returns everything needed to train the NN
surrogate:

    X = (log10 c0, log10 D, log10 r, t)   ->   y = C(x, t)  on a fixed grid x

Also returns `condition_id`, tagging each row with which (c0, D, r) condition
it came from - the n_times rows of one condition are correlated (same
trajectory, different t), so a downstream train/val/test split must keep all
of a condition's rows in the same split (see training.utils.split_indices).

A random subset is cross-checked against the Crank-Nicolson FDM solver
(solver.fdm_crank_nicolson) to confirm the analytical solution is being
evaluated correctly before it is used to label many conditions blindly.
"""
from __future__ import annotations

import numpy as np

from data.synthetic.solver import analytical_transient_profile, fdm_crank_nicolson


def sample_log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return 10 ** rng.uniform(np.log10(low), np.log10(high), size=size)


def build_dataset(sim_cfg, sampling_cfg):
    """Build the synthetic dataset from a `simulation` and a `sampling` config
    (either of the Hydra configs in configs/simulation, configs/sampling, or
    any object exposing the same attributes)."""
    rng = np.random.default_rng(sampling_cfg.seed)
    n_conditions = sampling_cfg.n_conditions
    n_times = sampling_cfg.n_times

    c0_vals = sample_log_uniform(rng, *sim_cfg.c0_range, n_conditions)
    D_vals = sample_log_uniform(rng, *sim_cfg.D_range, n_conditions)
    r_vals = sample_log_uniform(rng, *sim_cfg.r_range, n_conditions)

    L, nx = sim_cfg.L, sim_cfg.nx
    x_grid = np.linspace(0.0, L, nx)
    t_points = np.logspace(np.log10(sim_cfg.t_max / 1000), np.log10(sim_cfg.t_max), n_times)

    X = np.empty((n_conditions * n_times, 4), dtype=np.float64)
    Y = np.empty((n_conditions * n_times, nx), dtype=np.float64)
    # Which condition each row came from. All n_times rows of one condition
    # share a value here - needed for a group-aware train/val/test split, since
    # those rows are highly correlated (same physical trajectory, different t)
    # and are not i.i.d. the way a naive row-level split assumes.
    condition_id = np.empty(n_conditions * n_times, dtype=np.int64)

    row = 0
    for cond_idx, (c0, D, r) in enumerate(zip(c0_vals, D_vals, r_vals)):
        profiles = analytical_transient_profile(
            x_grid, t_points, D, r, c0, L, n_modes=sim_cfg.n_modes
        )  # (nx, n_times)
        for j, t in enumerate(t_points):
            X[row] = (np.log10(c0), np.log10(D), np.log10(r), t)
            Y[row] = profiles[:, j]
            condition_id[row] = cond_idx
            row += 1

    # Cross-check a random subset against the independent FDM solver. Capped
    # at a small absolute number regardless of n_conditions - this is a
    # sanity check on the analytical formula, not something that needs to
    # scale with dataset size, and each check costs its own FDM march.
    n_check = min(max(1, int(sampling_cfg.validate_frac * n_conditions)), 20)
    check_idx = rng.choice(n_conditions, size=n_check, replace=False)
    max_rel_err = 0.0
    for idx in check_idx:
        c0, D, r = c0_vals[idx], D_vals[idx], r_vals[idx]
        _, fdm_res = fdm_crank_nicolson(D, r, c0, L, nx, t_points)
        ana = analytical_transient_profile(x_grid, t_points, D, r, c0, L, n_modes=sim_cfg.n_modes)
        for j, t in enumerate(t_points):
            num = fdm_res[float(t)]
            denom = max(np.max(np.abs(ana[:, j])), 1e-12)
            rel_err = np.max(np.abs(ana[:, j] - num)) / denom
            max_rel_err = max(max_rel_err, rel_err)
    print(f"[validation] analytical vs FDM, worst-case relative error over "
          f"{n_check} sampled conditions: {max_rel_err:.3e}")

    return X, Y, x_grid, t_points, condition_id
