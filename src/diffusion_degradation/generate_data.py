"""
Synthetic dataset generation for the 1D diffusion-degradation surrogate.

Samples (c0, D, r) log-uniformly, evaluates the analytical transient
solution at several log-spaced times per condition, and stores everything
needed to train the NN surrogate:

    X = (log10 c0, log10 D, log10 r, t)   ->   y = C(x, t)  on a fixed grid x

Usage:
    python generate_data.py --n_conditions 2000 --n_times 8 --out ../../outputs/dataset.npz

For the full Lot 1 deliverable (Tache 1.2), re-run with --n_conditions 100000.
A random subset is cross-checked against the Crank-Nicolson FDM solver
(solver.fdm_crank_nicolson) to confirm the analytical solution is being
evaluated correctly before it is used to label 100k conditions blindly.
"""
from __future__ import annotations

import argparse

import numpy as np

from solver import analytical_transient_profile, fdm_crank_nicolson

# Parameter ranges, loosely inspired by Table 1 of Thurber & Wittrup (2012)
# for antibody transport in a Krogh-cylinder cross-section. Adjust freely -
# they only need to bracket the regimes you care about.
C0_RANGE = (1e-2, 1e1)      # arbitrary concentration unit (e.g. nM)
D_RANGE = (1.0, 50.0)       # um^2/s
R_RANGE = (1e-6, 1e-3)      # 1/s  (degradation / internalization-like rate)
L = 150.0                   # um, domain length (~ Krogh cylinder radius)
NX = 100                    # spatial grid points
T_MAX = 3 * 24 * 3600.0     # 3 days, in seconds


def sample_log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return 10 ** rng.uniform(np.log10(low), np.log10(high), size=size)


def build_dataset(n_conditions: int, n_times: int, seed: int = 0, validate_frac: float = 0.01):
    rng = np.random.default_rng(seed)

    c0_vals = sample_log_uniform(rng, *C0_RANGE, n_conditions)
    D_vals = sample_log_uniform(rng, *D_RANGE, n_conditions)
    r_vals = sample_log_uniform(rng, *R_RANGE, n_conditions)

    x_grid = np.linspace(0.0, L, NX)
    # shared log-spaced time points (skip t=0, trivial all-zero profile)
    t_points = np.logspace(np.log10(T_MAX / 1000), np.log10(T_MAX), n_times)

    X = np.empty((n_conditions * n_times, 4), dtype=np.float64)
    Y = np.empty((n_conditions * n_times, NX), dtype=np.float64)

    row = 0
    for c0, D, r in zip(c0_vals, D_vals, r_vals):
        profiles = analytical_transient_profile(x_grid, t_points, D, r, c0, L, n_modes=150)  # (NX, n_times)
        for j, t in enumerate(t_points):
            X[row] = (np.log10(c0), np.log10(D), np.log10(r), t)
            Y[row] = profiles[:, j]
            row += 1

    # Cross-check a random subset against the independent FDM solver. Capped
    # at a small absolute number regardless of n_conditions - this is a
    # sanity check on the analytical formula, not something that needs to
    # scale with dataset size, and each check costs its own FDM march.
    n_check = min(max(1, int(validate_frac * n_conditions)), 20)
    check_idx = rng.choice(n_conditions, size=n_check, replace=False)
    max_rel_err = 0.0
    for idx in check_idx:
        c0, D, r = c0_vals[idx], D_vals[idx], r_vals[idx]
        _, fdm_res = fdm_crank_nicolson(D, r, c0, L, NX, t_points)
        ana = analytical_transient_profile(x_grid, t_points, D, r, c0, L, n_modes=150)
        for j, t in enumerate(t_points):
            num = fdm_res[float(t)]
            denom = max(np.max(np.abs(ana[:, j])), 1e-12)
            rel_err = np.max(np.abs(ana[:, j] - num)) / denom
            max_rel_err = max(max_rel_err, rel_err)
    print(f"[validation] analytical vs FDM, worst-case relative error over "
          f"{n_check} sampled conditions: {max_rel_err:.3e}")

    return X, Y, x_grid, t_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_conditions", type=int, default=2000,
                         help="number of (c0, D, r) triples to sample (100000 for the full Lot 1 run)")
    parser.add_argument("--n_times", type=int, default=8, help="time points per condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="../../outputs/dataset.npz")
    args = parser.parse_args()

    X, Y, x_grid, t_points = build_dataset(args.n_conditions, args.n_times, args.seed)

    np.savez_compressed(
        args.out,
        X=X, Y=Y, x_grid=x_grid, t_points=t_points,
        c0_range=C0_RANGE, D_range=D_RANGE, r_range=R_RANGE, L=L,
    )
    print(f"Saved {X.shape[0]} samples ({args.n_conditions} conditions x {args.n_times} times) to {args.out}")
    print(f"X shape={X.shape}  Y shape={Y.shape}")


if __name__ == "__main__":
    main()
