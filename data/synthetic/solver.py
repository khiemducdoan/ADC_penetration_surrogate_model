"""
1D diffusion-degradation solver: analytical (eigenfunction expansion) and
numerical (Crank-Nicolson finite-difference) solutions of

    dc/dt = D * d2c/dx2 - r * c ,     x in [0, L]

Boundary conditions:
    c(0, t)   = c0        (Dirichlet, fixed source e.g. vessel wall / channel)
    dc/dx(L,t) = 0         (Neumann / zero-flux, symmetry at outer tissue edge)

Initial condition:
    c(x, 0) = 0

Two solution methods are provided:
  - analytical_transient_profile / steady_state_profile: exact (series)
    solution, valid for homogeneous D and r. Used to generate synthetic
    data quickly and to validate the numerical solver.
  - fdm_crank_nicolson: general implicit finite-difference solver. Needed
    once D(x) or r(x) become spatially heterogeneous, where the analytical
    solution no longer applies.
"""
from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def steady_state_profile(x: np.ndarray, D: float, r: float, c0: float, L: float) -> np.ndarray:
    """Exact steady-state solution: D*c'' - r*c = 0, c(0)=c0, c'(L)=0.

    c_ss(x) = c0 * cosh((L - x) / lambda) / cosh(L / lambda),  lambda = sqrt(D / r)

    `lambda` is the characteristic penetration depth.
    """
    lam = np.sqrt(D / r)
    return c0 * np.cosh((L - x) / lam) / np.cosh(L / lam)


def analytical_transient_profile(
    x: np.ndarray,
    t: np.ndarray | float,
    D: float,
    r: float,
    c0: float,
    L: float,
    n_modes: int = 200,
) -> np.ndarray:
    """Exact transient solution via eigenfunction expansion.

    Write c(x,t) = c_ss(x) - w(x,t). w solves the same PDE with homogeneous
    BCs w(0,t)=0, w_x(L,t)=0 and initial condition w(x,0) = c_ss(x).
    Its eigenfunctions are sin(k_n x) with k_n = (2n-1)*pi/(2L), decaying as
    exp(-(D k_n^2 + r) t). Coefficients B_n are found by projecting the
    initial condition onto the eigenbasis (orthogonal on [0, L]).

    Returns an array of shape (len(x),) if t is scalar, or (len(x), len(t))
    if t is a 1D array.
    """
    x = np.asarray(x, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    c_ss = steady_state_profile(x, D, r, c0, L)

    n = np.arange(1, n_modes + 1)
    k_n = (2 * n - 1) * np.pi / (2 * L)
    mu_n = D * k_n**2 + r

    xq = np.linspace(0.0, L, 2000)
    css_q = steady_state_profile(xq, D, r, c0, L)
    sin_nq = np.sin(np.outer(k_n, xq))
    I_n = _trapezoid(css_q[None, :] * sin_nq, xq, axis=1)
    B_n = (2.0 / L) * I_n

    sin_nx = np.sin(np.outer(k_n, x))              # (n_modes, len(x))
    decay = np.exp(-mu_n[:, None] * t_arr[None, :])  # (n_modes, len(t))
    w = sin_nx.T @ (B_n[:, None] * decay)             # (len(x), len(t))
    profile = c_ss[:, None] - w

    if np.ndim(t) == 0:
        return profile[:, 0]
    return profile


def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Tridiagonal solver (Thomas algorithm), avoids a scipy dependency."""
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    out = np.empty(n)
    out[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


def fdm_crank_nicolson(
    D: float,
    r: float,
    c0: float,
    L: float,
    nx: int,
    t_eval: np.ndarray,
    max_steps: int = 3000,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Implicit Crank-Nicolson finite-difference solver (unconditionally stable).

    Dirichlet BC at x=0 (c=c0), zero-flux BC at x=L implemented with a
    mirrored ghost node (c[nx] = c[nx-2]).

    Each requested time in `t_eval` is reached by an independent march from
    t=0, with its own step count (an accuracy heuristic, capped at
    `max_steps` for speed). This keeps early, short-time requests well
    resolved even when other requested times are orders of magnitude larger
    - a single shared dt sized for the largest t would otherwise under-
    resolve the early ones.

    Returns (x_grid, {t: profile}) for every requested time in `t_eval`.
    """
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]
    t_eval = np.sort(np.asarray(t_eval, dtype=float))

    n_unknown = nx - 1  # unknowns are nodes i = 1 .. nx-1 (i=0 is Dirichlet)
    results: dict[float, np.ndarray] = {}

    for t_target in t_eval:
        if np.isclose(t_target, 0.0):
            c_cur = np.zeros(nx)
            c_cur[0] = c0
            results[float(t_target)] = c_cur
            continue

        dt_accuracy = min(0.4 * dx * dx / D, 0.05 / r) if r > 0 else 0.4 * dx * dx / D
        nsteps = int(np.clip(np.ceil(t_target / dt_accuracy), 1, max_steps))
        dt = t_target / nsteps

        alpha = D * dt / (2 * dx * dx)
        beta = r * dt / 2.0

        a = np.full(n_unknown, -alpha)
        b = np.full(n_unknown, 1 + 2 * alpha + beta)
        c_diag = np.full(n_unknown, -alpha)
        a[-1] = -2 * alpha   # mirrored ghost node at the outer (zero-flux) boundary
        c_diag[-1] = 0.0

        c_cur = np.zeros(nx)
        c_cur[0] = c0

        for _ in range(nsteps):
            d = alpha * c_cur[:-2] + (1 - 2 * alpha - beta) * c_cur[1:-1] + alpha * c_cur[2:]
            d_last = 2 * alpha * c_cur[-2] + (1 - 2 * alpha - beta) * c_cur[-1]
            d = np.append(d, d_last)
            d[0] += alpha * c0  # known Dirichlet value contributes to the i=1 equation

            u = _thomas_solve(a, b, c_diag, d)
            c_cur = np.concatenate(([c0], u))

        results[float(t_target)] = c_cur

    return x, results


if __name__ == "__main__":
    # Quick self-check: analytical series solution vs. Crank-Nicolson FDM.
    D, r, c0, L = 10.0, 1e-3, 1.0, 100.0
    x = np.linspace(0, L, 101)
    t_eval = [0.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]

    _, res = fdm_crank_nicolson(D, r, c0, L, 101, t_eval)
    for t in t_eval[1:]:
        c_ana = analytical_transient_profile(x, t, D, r, c0, L, n_modes=300)
        c_num = res[t]
        rel_err = np.max(np.abs(c_ana - c_num)) / max(np.max(c_ana), 1e-12)
        print(f"t={t:8.1f}  max relative error (analytical vs FDM) = {rel_err:.3e}")
