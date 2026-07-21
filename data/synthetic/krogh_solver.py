"""
Spatial Krogh-cylinder solver: antibody transport + reversible antigen
binding in tumor tissue, in cylindrical radial coordinates r in [Rcap, Rkrogh],
driven by a time-varying plasma antibody concentration.

    d[Ab_free]/dt  = D * (1/r) * d/dr( r * d[Ab_free]/dr )
                       - (kon/eps) * [Ab_free] * [Ag] + koff * [Ab_bound]
    d[Ab_bound]/dt = (kon/eps) * [Ab_free] * [Ag] - koff * [Ab_bound] - ke * [Ab_bound]
    d[Ag]/dt       = Rs - (kon/eps) * [Ab_free] * [Ag] + koff * [Ab_bound] - ke * [Ag]

Boundary conditions:
    -D * d[Ab_free]/dr |_{r=Rcap}   = P * ( [Ab]_plasma(t) - [Ab_free]/eps )   (Robin)
        d[Ab_free]/dr |_{r=Rkrogh} = 0                                          (Neumann)

Only [Ab_free] diffuses -- [Ab_bound] (membrane-bound complex) and [Ag]
(cell-surface antigen) are immobile, so their equations are purely local
reaction ODEs at each radial node.

This is a nonlinear system (the kon*Ab_free*Ag term couples all three
species) with no closed-form solution, so it is solved by the method of
lines: a conservative finite-volume discretization in r reduces the PDE to a
large system of stiff ODEs in t, integrated with an implicit solver
(`scipy.integrate.solve_ivp`, BDF/Radau). See
docs/THURBER_KROGH_PDE_MATH.md for the full derivation.

As a validation, the volume-averaged total antibody concentration from this
spatial model is compared against the compartmental (0D) analytical solution
of Thurber & Wittrup (2012, J Theor Biol 314:57-68) -- `compartmental_ab_ratio`
below -- which the spatial model must reduce to once diffusion has
equilibrated the radial profile (see docs for the regime where this holds).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp


@dataclass
class KroghParams:
    """Physical parameters. Defaults are representative orders of magnitude
    from Thurber, Zajic & Wittrup (J Nucl Med 2007;48:995-999) and Thurber &
    Wittrup (2012, Tables 1-2) -- adjust freely for a specific antibody/
    antigen system. All times are in seconds, lengths in um, concentrations
    in nM (matches matlab/krogh_binding_pde.m so the two implementations are
    directly comparable)."""

    D: float = 10.0          # um^2/s   - antibody diffusion coefficient in tissue
    Rcap: float = 10.0       # um       - capillary (vessel) radius
    Rkrogh: float = 75.0     # um       - Krogh cylinder outer radius
    eps: float = 0.24        # -        - interstitial (void) volume fraction
    P: float = 3.0e-3        # um/s     - vascular permeability

    kon: float = 1.0e-3      # 1/(nM*s) - antibody-antigen association rate
    Kd: float = 1.0          # nM       - equilibrium dissociation constant
    ke: float = 1.109 / 86400.0  # 1/s  - internalization/turnover rate

    Ag0: float = 100.0       # nM       - baseline (pre-dose) effective antigen concentration

    Ab_plasma0: float = 10.0                 # nM  - plasma Ab concentration at t=0
    A_frac: float = 0.6
    ka: float = 5.00 / 86400.0               # 1/s - alpha phase (fast distribution)
    B_frac: float = 0.4
    kb: float = 0.05 / 86400.0               # 1/s - beta phase (slow clearance)

    koff: float = field(init=False)          # 1/s - derived: kon * Kd
    Rs: float = field(init=False)            # nM/s - derived: synthesis rate holding Ag0 at baseline

    def __post_init__(self) -> None:
        self.koff = self.kon * self.Kd
        self.Rs = self.ke * self.Ag0

    def plasma_pk(self, t: np.ndarray | float) -> np.ndarray | float:
        """Two-phase (bi-exponential) plasma antibody concentration [Ab]_plasma(t)."""
        return self.Ab_plasma0 * (
            self.A_frac * np.exp(-self.ka * t) + self.B_frac * np.exp(-self.kb * t)
        )

    @property
    def kex(self) -> float:
        """Extravasation rate constant, Thurber & Wittrup (2012) definition."""
        return 2.0 * self.P * self.Rcap / self.Rkrogh**2


# --------------------------------------------------------------------------
# Spatial discretization: conservative finite-volume operator for
# D * (1/r) d/dr(r d.../dr), Robin BC at r=Rcap, Neumann (zero-flux) at r=Rkrogh.
# See docs/THURBER_KROGH_PDE_MATH.md Step 3 for the derivation.
# --------------------------------------------------------------------------

def build_radial_grid(Rcap: float, Rkrogh: float, nr: int) -> np.ndarray:
    return np.linspace(Rcap, Rkrogh, nr)


def build_diffusion_operator(r: np.ndarray, D: float, P: float, eps: float):
    """Build the (N,N) sparse matrix `Lop` and length-N vector `robin_coef` such
    that, for the free-antibody field c (length N),

        d c / dt |_diffusion  =  Lop @ c  +  robin_coef * Ab_plasma(t)

    reproduces the finite-volume discretization of
    D*(1/r) d/dr(r dc/dr) with:
      - a Robin (mixed) condition at r[0] = Rcap encoding
        -D dc/dr|_Rcap = P*(Ab_plasma(t) - c[0]/eps)
      - a zero-flux (Neumann) condition at r[-1] = Rkrogh.
    """
    n = len(r)
    dr = r[1] - r[0]
    r_half = r[:-1] + dr / 2.0  # r_{i+1/2} for i = 0 .. n-2, length n-1

    main = np.zeros(n)
    lower = np.zeros(n - 1)  # sub-diagonal, lower[i] couples row i+1 to col i
    upper = np.zeros(n - 1)  # super-diagonal, upper[i] couples row i to col i+1
    robin_coef = np.zeros(n)

    # Interior rows i = 1 .. n-2: standard conservative 3-point stencil.
    for i in range(1, n - 1):
        r_m = r_half[i - 1]  # r_{i-1/2}
        r_p = r_half[i]      # r_{i+1/2}
        coef = D / (r[i] * dr * dr)
        lower[i - 1] = coef * r_m
        main[i] = -coef * (r_m + r_p)
        upper[i] = coef * r_p

    # Row 0 (r = Rcap): eliminate ghost node c[-1] using the Robin BC
    #   (c[1] - c[-1]) / (2 dr) = -(P/D) * (Ab_plasma(t) - c[0]/eps)
    #   => c[-1] = c[1] + K * (Ab_plasma(t) - c[0]/eps),  K = 2*dr*P/D
    r_mHalf0 = r[0] - dr / 2.0   # r_{-1/2}, extrapolated (Rcap > dr/2 for any reasonable grid)
    r_pHalf0 = r_half[0]         # r_{1/2}
    coef0 = D / (r[0] * dr * dr)
    K = 2.0 * dr * P / D
    main[0] = -coef0 * (r_mHalf0 + r_pHalf0) - coef0 * r_mHalf0 * K / eps
    upper[0] = coef0 * (r_mHalf0 + r_pHalf0)
    robin_coef[0] = coef0 * r_mHalf0 * K

    # Row n-1 (r = Rkrogh): zero-flux ghost c[n] = c[n-2] (mirror), substituted
    # into the same 3-point stencil as the interior rows (r_{n-3/2} + r_{n-1/2}
    # collapses to 2*r[-1], since the two faces are dr/2 below and above r[-1]).
    r_mHalfN = r_half[-1]           # r_{n-3/2}
    r_pHalfN = r[-1] + dr / 2.0     # r_{n-1/2} (ghost cell edge)
    coefN = D / (r[-1] * dr * dr)
    face_sum = r_mHalfN + r_pHalfN  # = 2*r[-1]
    lower[-1] = coefN * face_sum
    main[-1] = -coefN * face_sum

    Lop = sparse.diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")
    return Lop, robin_coef


# --------------------------------------------------------------------------
# Method-of-lines right-hand side and analytic Jacobian for the full
# 3N-dimensional stiff ODE system (state y = [Ab_free, Ab_bound, Ag]).
# --------------------------------------------------------------------------

def _unpack(y: np.ndarray, n: int):
    return y[0:n], y[n:2 * n], y[2 * n:3 * n]


def make_rhs(params: KroghParams, Lop, robin_coef: np.ndarray, n: int):
    kon, koff, ke, eps, Rs = params.kon, params.koff, params.ke, params.eps, params.Rs

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        free, bound, ag = _unpack(y, n)
        reaction = (kon / eps) * free * ag
        dfree = Lop @ free + robin_coef * params.plasma_pk(t) - reaction + koff * bound
        dbound = reaction - koff * bound - ke * bound
        dag = Rs - reaction + koff * bound - ke * ag
        return np.concatenate([dfree, dbound, dag])

    return rhs


def make_jac(params: KroghParams, Lop, n: int):
    kon, koff, ke, eps = params.kon, params.koff, params.ke, params.eps
    I = sparse.identity(n, format="csr")

    def jac(t: float, y: np.ndarray) -> sparse.csr_matrix:
        free, bound, ag = _unpack(y, n)
        d_ag = sparse.diags(ag)
        d_free = sparse.diags(free)
        koneps = kon / eps

        d_free_d_free = Lop - koneps * d_ag
        d_free_d_bound = koff * I
        d_free_d_ag = -koneps * d_free

        d_bound_d_free = koneps * d_ag
        d_bound_d_bound = -(koff + ke) * I
        d_bound_d_ag = koneps * d_free

        d_ag_d_free = -koneps * d_ag
        d_ag_d_bound = koff * I
        d_ag_d_ag = -koneps * d_free - ke * I

        return sparse.bmat(
            [
                [d_free_d_free, d_free_d_bound, d_free_d_ag],
                [d_bound_d_free, d_bound_d_bound, d_bound_d_ag],
                [d_ag_d_free, d_ag_d_bound, d_ag_d_ag],
            ],
            format="csr",
        )

    return jac


def solve_krogh_pde(
    params: KroghParams,
    nr: int = 100,
    t_eval: np.ndarray | None = None,
    t_max: float = 10 * 86400.0,
    method: str = "BDF",
    rtol: float = 1e-7,
    atol: float = 1e-9,
):
    """Solve the Krogh-cylinder PDE system by the method of lines.

    Returns (r, t, Ab_free, Ab_bound, Ag) where Ab_free/Ab_bound/Ag have
    shape (len(t), nr).
    """
    r = build_radial_grid(params.Rcap, params.Rkrogh, nr)
    Lop, robin_coef = build_diffusion_operator(r, params.D, params.P, params.eps)

    if t_eval is None:
        t_eval = np.unique(np.concatenate([[0.0], np.logspace(0, np.log10(t_max), 200)]))

    y0 = np.concatenate([np.zeros(nr), np.zeros(nr), np.full(nr, params.Ag0)])

    rhs = make_rhs(params, Lop, robin_coef, nr)
    jac = make_jac(params, Lop, nr)

    sol = solve_ivp(
        rhs, (0.0, t_eval[-1]), y0, t_eval=t_eval, method=method,
        jac=jac, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    Ab_free = sol.y[0:nr, :].T
    Ab_bound = sol.y[nr:2 * nr, :].T
    Ag = sol.y[2 * nr:3 * nr, :].T
    return r, sol.t, Ab_free, Ab_bound, Ag


# --------------------------------------------------------------------------
# Compartmental (0D) analytical reference: Thurber & Wittrup (2012) Eq. 6-8.
# Used to validate the spatial solver in the sub-saturating regime.
# --------------------------------------------------------------------------

def compartmental_ab_ratio(params: KroghParams, t: np.ndarray) -> np.ndarray:
    """[Ab]_total(t) / [Ab]_plasma0 from the closed-form compartmental solution
    (Thurber & Wittrup 2012, Eq. 7-8), using kex derived from this model's
    P, Rcap, Rkrogh (Eq. 5 of the same paper)."""
    kex = params.kex
    Ag_eps, Kd, ke = params.Ag0, params.Kd, params.ke
    Omega = kex * Kd / (Ag_eps + Kd) + ke * Ag_eps / (Ag_eps + Kd)
    A, ka, B, kb = params.A_frac, params.ka, params.B_frac, params.kb
    return kex * (
        A / (Omega - ka) * (np.exp(-ka * t) - np.exp(-Omega * t))
        + B / (Omega - kb) * (np.exp(-kb * t) - np.exp(-Omega * t))
    )


def volume_averaged(r: np.ndarray, field_rt: np.ndarray) -> np.ndarray:
    """Volume (area) average of a (nt, nr) field over the annulus [r[0], r[-1]],
    weighted by r dr (cylindrical shell area element)."""
    area_norm = (r[-1] ** 2 - r[0] ** 2) / 2.0
    return np.trapezoid(r[None, :] * field_rt, r, axis=1) / area_norm


if __name__ == "__main__":
    params = KroghParams()
    print(f"Derived rates: kex = {params.kex:.4e} /s ({params.kex * 86400:.4f} /day), "
          f"koff = {params.koff:.4e} /s, ke = {params.ke:.4e} /s")

    # --- Grid-convergence self-check (Richardson-style: nr doubling) ---
    t_max = 10 * 86400.0
    t_eval = np.unique(np.concatenate([[0.0], np.logspace(0, np.log10(t_max), 60)]))
    errs = []
    for nr in (25, 50, 100, 200):
        r, t, free, bound, ag = solve_krogh_pde(params, nr=nr, t_eval=t_eval, t_max=t_max)
        Ab_total_avg = volume_averaged(r, free + bound)
        errs.append(Ab_total_avg)
    # Compare successive resolutions at the finest common set of time points.
    e1 = np.max(np.abs(errs[1] - errs[0])[1:])
    e2 = np.max(np.abs(errs[2] - errs[1])[1:])
    e3 = np.max(np.abs(errs[3] - errs[2])[1:])
    print(f"Grid convergence (max diff in volume-avg [Ab]_total, successive nr doublings):")
    print(f"  nr 25->50:  {e1:.4e}   nr 50->100: {e2:.4e}   nr 100->200: {e3:.4e}")
    print(f"  observed order ~ log2(e1/e2) = {np.log2(e1 / e2):.2f}, "
          f"log2(e2/e3) = {np.log2(e2 / e3):.2f}  (expect ~2 for a 2nd-order scheme)")

    # --- Validation against the compartmental (0D) analytical model ---
    r, t, free, bound, ag = solve_krogh_pde(params, nr=100, t_eval=t_eval, t_max=t_max)
    Ab_total_avg = volume_averaged(r, free + bound)
    Ab_total_compartmental = params.Ab_plasma0 * compartmental_ab_ratio(params, t)
    rel_err = np.abs(Ab_total_avg - Ab_total_compartmental) / np.maximum(Ab_total_compartmental, 1e-12)
    print(f"\nSpatial-avg vs. compartmental Thurber model: "
          f"max rel. error = {np.max(rel_err[1:]):.3e}, median = {np.median(rel_err[1:]):.3e}")
    print(f"Solved: {len(t)} time points x {len(r)} radial points. "
          f"min/max Ab_free = [{free.min():.4g}, {free.max():.4g}] nM")
