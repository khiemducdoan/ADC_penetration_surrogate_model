# Mathematical Derivation of the Numerical Solver — Thurber/Krogh-Cylinder Antibody–Antigen System

**Companion code:** [`data/synthetic/krogh_solver.py`](../data/synthetic/krogh_solver.py) (solver) and [`data/synthetic/krogh_visualize.py`](../data/synthetic/krogh_visualize.py) (figures). A MATLAB implementation of the same model (`pdepe`-based) exists at [`matlab/krogh_binding_pde.m`](../matlab/krogh_binding_pde.m) for cross-reference; the compartmental (0D) reduction it validates against is worked out separately in [`ANALYTICAL_SOLVER_MATH.md`](ANALYTICAL_SOLVER_MATH.md)'s sibling model, [`matlab/thurber_model.m`](../matlab/thurber_model.m).

This document derives, from first principles, the spatial discretization and time-integration scheme used to solve the full **Krogh-cylinder** reaction–diffusion system: three coupled species (free antibody, antibody–antigen complex, free antigen) diffusing and reacting in the radial gap between a tumor capillary and the surrounding tissue. Unlike the 1D linear problem in `ANALYTICAL_SOLVER_MATH.md`, this system has **no closed-form solution** — the reaction term couples all three equations nonlinearly — so the entire document is about building and justifying a numerical method, not a formula.

---

## Step 0 — Problem statement

**Geometry.** A single capillary of radius `R_cap` supplies a cylindrical shell of tissue out to radius `R_Krogh` (the *Krogh cylinder*: half the distance to the next capillary, the natural repeating unit of a capillary bed). Antibody enters the tissue only by crossing the capillary wall at `r = R_cap`; by symmetry with the neighboring capillary, nothing crosses `r = R_Krogh`. The domain is `r ∈ [R_cap, R_Krogh]`, and by the cylindrical symmetry of the geometry, concentrations depend on `r` and `t` only (no angular or axial dependence).

**State variables** (concentrations, nM, all functions of `(r,t)`):

- `[Ab]_free(r,t)` — antibody diffusing freely through the interstitium
- `[Ab]_bound(r,t)` — antibody bound to cell-surface antigen (immobile — it moves only by unbinding, not by diffusion)
- `[Ag](r,t)` — free (unbound) cell-surface antigen (also immobile)

**Governing PDEs** (cylindrical coordinates, no angular/axial dependence):

$$
\frac{\partial [Ab]_{free}}{\partial t} = D\,\frac{1}{r}\frac{\partial}{\partial r}\!\left(r\,\frac{\partial [Ab]_{free}}{\partial r}\right) - \frac{k_{on}}{\varepsilon}[Ab]_{free}[Ag] + k_{off}[Ab]_{bound}
$$

$$
\frac{\partial [Ab]_{bound}}{\partial t} = \frac{k_{on}}{\varepsilon}[Ab]_{free}[Ag] - k_{off}[Ab]_{bound} - k_e[Ab]_{bound}
$$

$$
\frac{\partial [Ag]}{\partial t} = R_s - \frac{k_{on}}{\varepsilon}[Ab]_{free}[Ag] + k_{off}[Ab]_{bound} - k_e[Ag]
$$

**Boundary conditions** (only `[Ab]_free` has a spatial derivative, so only it needs BCs):

$$
-D\left.\frac{\partial [Ab]_{free}}{\partial r}\right|_{r=R_{cap}} = P\left([Ab]_{plasma}(t) - \frac{[Ab]_{free}}{\varepsilon}\right) \qquad \text{(Robin / mixed)}
$$

$$
\left.\frac{\partial [Ab]_{free}}{\partial r}\right|_{r=R_{Krogh}} = 0 \qquad \text{(Neumann / zero-flux, symmetry)}
$$

**Initial conditions:** `[Ab]_free(r,0) = 0`, `[Ab]_bound(r,0) = 0`, `[Ag](r,0) = Ag_0` (tissue starts unexposed, antigen at its pre-dose baseline).

**Parameters:** `D` (diffusion coefficient), `k_on, k_off = k_on K_d` (binding kinetics), `k_e` (internalization/turnover of both bound antibody and antigen), `ε` (interstitial void fraction — free-phase concentrations are per unit *interstitial* volume, so they get divided by `ε` where they interact with the whole-tissue-basis Robin BC and reaction terms use the same convention consistently — see Step 1), `P` (vascular permeability), `R_s = k_e Ag_0` (antigen synthesis rate that sustains the baseline `Ag_0` in the absence of antibody, so the system starts at its own steady state for `[Ag]` before any drug arrives), and `[Ab]_plasma(t)` (a prescribed, externally-supplied plasma PK curve — see Step 1).

**Why this is a strictly harder problem than the 1D linear case.** Three couplings appear here that were absent before:

1. **Multiple species** instead of one — three coupled fields instead of one.
2. **Nonlinear reaction** `[Ab]_free]\cdot[Ag]` — the equations are not separable and superposition (the technique behind the entire `ANALYTICAL_SOLVER_MATH.md` derivation) does not apply.
3. **Time-dependent boundary forcing** `[Ab]_plasma(t)` instead of a constant `c_0` — even the linear diffusion sub-problem alone no longer has a time-independent "steady state" to split off.

Any one of these would already break the eigenfunction-expansion method; together they rule out a closed-form solution entirely. What remains available is: (a) a numerical solution of the full spatial system (this document), and (b) an independent 0D **compartmental** ODE model (Thurber & Wittrup 2012) that results from *assuming* the radial profile is always at quasi-steady-state and averaging over the shell — this gives no spatial information, but supplies a closed-form check on the volume-averaged total antibody, which Step 8 uses to validate the numerical solver.

---

## Step 1 — Physical reading of each term

- **Diffusion operator** `D (1/r) ∂/∂r(r ∂c/∂r)` is the radial part of the Laplacian in cylindrical coordinates with no angular/axial dependence — the natural generalization of `D ∂²c/∂x²` from the 1D slab case to a cylindrical shell. The `1/r` weighting reflects that a thin shell at large `r` has more circumference (more "space" to spread into) than one at small `r`, which dilutes concentration purely from geometry, independent of any reaction.
- **`ε` (interstitial void fraction).** Tissue is not open space — most of the volume is cells; antibody only occupies the extracellular/interstitial fraction `ε`. `[Ab]_free` is defined per unit *interstitial* volume; `[Ab]_free]/\varepsilon` converts it to a per-total-tissue-volume basis, which is what physically leaves/enters through the capillary wall (Robin BC) and what the reaction term needs when `[Ag]` is also stated on a whole-tissue basis.
- **Reaction term `(k_on/\varepsilon)[Ab]_{free}[Ag]`** is standard mass-action binding kinetics; the `1/\varepsilon` factor is the same unit conversion as above, needed because `[Ab]_free` and `[Ag]` are stated on different volume bases.
- **`k_e`** lumps every first-order loss of bound antibody and antigen (receptor-mediated internalization + intracellular degradation) — the spatial-model analogue of the `-r c` loss term in the 1D problem, but now acting only on the *bound* species (bound complex is what gets internalized), not on `[Ab]_free` directly.
- **Robin BC at `r=R_cap`.** This is Starling's-law-type flux: flux across the vessel wall is proportional to the concentration *gradient across the wall* — plasma concentration `[Ab]_plasma(t)` on one side, tissue concentration `[Ab]_free]/\varepsilon` on the other — with proportionality constant `P` (permeability). It replaces the Dirichlet condition `c(0,t)=c_0` of the 1D model: here the wall does not clamp the tissue concentration outright, it only pushes flux across at a finite rate, which is the more physically realistic capillary-wall model this project's 1D `ANALYTICAL_SOLVER_MATH.md` deliberately simplified away.
- **Neumann BC at `r=R_Krogh`.** Same symmetry argument as the 1D model's `x=L` — the midpoint between two capillaries sees equal contributions from both sides, so no net flux crosses it.
- **`[Ab]_plasma(t)`, two-phase PK.** A bi-exponential `A e^{-k_a t} + B e^{-k_b t}` (fast "alpha" distribution phase, slow "beta" clearance phase) is the standard empirical form for IgG plasma pharmacokinetics; it is *not* itself computed by this solver — it is a forcing function imposed at the boundary, exactly like `c_0` was a fixed forcing value in the 1D model, except now it varies in time.

---

## Step 2 — Why method of lines

The general strategy — used throughout numerical PDE work and directly applicable here — is the **method of lines (MOL)**: discretize *only* the spatial derivatives, leaving time continuous. This converts the PDE system into a (large) system of coupled ODEs in `t`,

$$
\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}), \qquad \mathbf{y}(t) \in \mathbb{R}^{3N},
$$

where `N` is the number of radial grid points and the factor of 3 comes from the three species. This is attractive here specifically because:

1. **The reaction terms are purely local** (no spatial derivative) — they only ever couple the three species *at the same radial point*, never at different points. So discretizing space only touches the diffusion term; the reaction terms translate into the ODE system completely unchanged, node by node.
2. **A mature, adaptive, implicit ODE integrator can be reused** off the shelf (`scipy.integrate.solve_ivp`) instead of hand-building a bespoke implicit PDE time-stepper — the same design choice MATLAB's `pdepe` makes internally (it is itself a MOL wrapper around `ode15s`, a BDF integrator).
3. **Stiffness is handled uniformly.** As shown in Step 6, both the diffusion operator and the reaction kinetics can be stiff for physically reasonable parameters; an adaptive implicit integrator handles both sources of stiffness without the user having to hand-tune a time step.

The two remaining questions are: how to discretize the diffusion operator in `r` (Step 3), and how to integrate the resulting ODE system in `t` (Steps 6–7).

---

## Step 3 — Conservative finite-volume discretization in r

**Grid.** Place `N` uniformly spaced nodes `r_0=R_cap < r_1 < \dots < r_{N-1}=R_{Krogh}`, spacing `\Delta r = (R_{Krogh}-R_{cap})/(N-1)`. Only `[Ab]_free` needs this treatment; `[Ab]_bound` and `[Ag]` are just `N` independent local unknowns each, one per node, with no spatial coupling between them (this is the key simplification enabled by them being genuinely immobile — see the box in Step 4 for why this is actually a *cleaner* choice than the artificial-diffusion workaround MATLAB's `pdepe` requires).

**Deriving the scheme from a physical conservation law (not just Taylor series).** The clean way to discretize a divergence-form operator like `(1/r)\partial_r(r\,\partial_r c)` so that it remains numerically conservative (no spurious mass creation/destruction) is to integrate the flux balance over a small control volume, rather than naively finite-differencing derivatives term by term. Define **cell faces** halfway between nodes, `r_{i\pm1/2} = r_i \pm \Delta r/2`. The radial flux (per unit height of the cylinder) is `q(r,t) = -D\,\partial_r c`, so integrating `\partial_t c = -(1/r)\partial_r(r q)` over the annular shell `[r_{i-1/2}, r_{i+1/2}]` (area element `r\,dr\,d\theta`, and everything is `\theta`-independent) and dividing through by `\int r\,dr` over that shell gives an *exact* statement of "concentration change = net flux in through the two bounding circles, per unit shell volume":

$$
\frac{d c_i}{dt} = \frac{2}{r_{i+1/2}^2 - r_{i-1/2}^2}\Big[r_{i-1/2}\,q(r_{i-1/2}) - r_{i+1/2}\,q(r_{i+1/2})\Big] .
$$

Approximating `q(r_{i\pm1/2}) \approx -D\,(c_{i\pm1}-c_i)/\Delta r` (a standard centered difference, second-order accurate for the flux *at the face*) and using `r_{i+1/2}^2-r_{i-1/2}^2 = 2 r_i \Delta r` (exact for the uniform grid defined above) gives the **conservative 3-point stencil** actually implemented in `build_diffusion_operator`:

$$
\left(Lc\right)_i \;:=\; \frac{D}{r_i\,\Delta r^2}\Big[r_{i-1/2}\,c_{i-1} - \big(r_{i-1/2}+r_{i+1/2}\big)\,c_i + r_{i+1/2}\,c_{i+1}\Big], \qquad i=1,\dots,N-2 .
$$

**Why go through the control-volume argument instead of just discretizing `\partial_r(r\,\partial_r c)` and dividing by `r_i`?** Both give the same interior stencil here (uniform grid), but the control-volume form is what generalizes correctly to the two boundary rows below, because it is derived from *flux balance*, and flux balance is exactly what a boundary condition specifies (a flux, or a relation determining one). Working from "what flux crosses each face" keeps the boundary treatment consistent with the interior discretization instead of being a separate, ad hoc patch.

**Consistency (this is the discrete operator's truncation error, i.e. why the stencil is a valid approximation at all).** Taylor-expand `c_{i\pm1} = c_i \pm \Delta r\,c_i' + \tfrac12\Delta r^2 c_i'' \pm \tfrac16\Delta r^3 c_i''' + O(\Delta r^4)` and `r_{i\pm1/2}=r_i\pm\Delta r/2`, substitute into `(Lc)_i`, and collect powers of `\Delta r`. The `O(\Delta r^0)` terms reproduce `D(c_i'' + c_i'/r_i) = D\,(1/r)(r c')'|_{r_i}` exactly; the leading error term is `O(\Delta r^2)`, so **the interior scheme is second-order accurate** — consistent with a standard centered finite-volume/finite-difference discretization. This is confirmed empirically in Step 9 (grid-refinement study).

---

## Step 4 — Discretizing the boundary conditions (ghost-node elimination)

The stencil above needs `c_{i-1}` and `c_{i+1}` — both exist for interior nodes, but at `i=0` the "node" `c_{-1}` lies outside the domain, and at `i=N-1` the node `c_N` does too. The standard technique is to introduce a fictitious **ghost node** just outside the domain, write the boundary condition as a discrete relation involving it, solve for the ghost value, and substitute it back into the same interior stencil — this way the boundary rows use *exactly* the same conservative stencil as the interior, just with one neighbor eliminated algebraically. This keeps the boundary treatment second-order accurate and automatically consistent with the interior scheme (no separate one-sided-difference formula needed, which would typically drop to first order).

### Inner boundary (`i=0`, Robin condition at `r=R_cap`)

Approximate the derivative in the BC with a centered difference through the ghost node `c_{-1}`:

$$
\left.\frac{\partial c}{\partial r}\right|_{r_0} \approx \frac{c_1 - c_{-1}}{2\Delta r} = -\frac{P}{D}\left([Ab]_{plasma}(t) - \frac{c_0}{\varepsilon}\right)
\;\;\Longrightarrow\;\;
c_{-1} = c_1 + K\left([Ab]_{plasma}(t) - \frac{c_0}{\varepsilon}\right), \quad K := \frac{2\Delta r\,P}{D}.
$$

Substitute this `c_{-1}` into the interior formula evaluated at `i=0` (using `r_{-1/2}=r_0-\Delta r/2`, which is positive and well-defined as long as `\Delta r < 2 R_{cap}` — true for any reasonable grid since `R_{cap}` is a physical vessel radius, not a coordinate singularity like `r=0` would be):

$$
(Lc)_0 = \underbrace{\frac{D}{r_0\Delta r^2}(r_{-1/2}+r_{1/2})}_{\text{coefficient on } c_1}c_1 \;-\; \underbrace{\left[\frac{D}{r_0\Delta r^2}(r_{-1/2}+r_{1/2}) + \frac{D}{r_0\Delta r^2}r_{-1/2}\frac{K}{\varepsilon}\right]}_{\text{coefficient on } c_0}c_0 \;+\; \underbrace{\frac{D}{r_0\Delta r^2}r_{-1/2}K}_{=:\,\rho_0}\,[Ab]_{plasma}(t).
$$

The first two terms are linear in the unknowns `c_0, c_1` exactly like an interior row (so they fold into the same sparse matrix `Lop`); the last term is a **known forcing function of time**, independent of the state — it is *not* part of `Lop`, it is added separately to the right-hand side at every evaluation (this is `robin_coef[0] * params.plasma_pk(t)` in the code). Physically: the negative coefficient on `c_0` (beyond the usual diffusion coefficient) is a permeability-driven *outflow* term — tissue antibody can leak back into the vessel — while the forcing term is the permeability-driven *inflow* from plasma. This is exactly the two-way exchange a Robin condition represents; a Dirichlet condition (as in the 1D model) is the `P\to\infty` limit where inflow/outflow are infinitely fast and `c_0` gets clamped to `\varepsilon\,[Ab]_{plasma}(t)` outright.

### Outer boundary (`i=N-1`, Neumann condition at `r=R_Krogh`)

Zero flux: `\partial_r c|_{r_{N-1}} \approx (c_N - c_{N-2})/(2\Delta r) = 0 \;\Rightarrow\; c_N = c_{N-2}` (a **mirror** ghost node — physically, "reflect" the profile across the symmetry plane, which is the discrete statement that nothing distinguishes crossing outward from crossing back inward at a true symmetry boundary). Substituting into the interior stencil at `i=N-1`:

$$
(Lc)_{N-1} = \frac{D}{r_{N-1}\Delta r^2}\Big[r_{N-3/2}\,c_{N-2} - (r_{N-3/2}+r_{N-1/2})\,c_{N-1} + r_{N-1/2}\,c_N\Big]\Big|_{c_N = c_{N-2}} = \frac{D}{r_{N-1}\Delta r^2}(r_{N-3/2}+r_{N-1/2})\big(c_{N-2}-c_{N-1}\big),
$$

using `c_N=c_{N-2}` to combine the two `c_{N-2}` coefficients into one. Note `r_{N-3/2}+r_{N-1/2} = 2 r_{N-1}` exactly (the two faces are `\Delta r/2` below and above `r_{N-1}`), so this simplifies to `(2D/\Delta r^2)(c_{N-2}-c_{N-1})` — no forcing term, purely linear, exactly as expected for a homogeneous (zero-flux) boundary condition. This row, too, folds entirely into `Lop`.

> **A note on what the MATLAB reference implementation does differently, and why.** `matlab/krogh_binding_pde.m` uses MATLAB's built-in `pdepe`, which requires *every* PDE component with a nonzero `\partial c/\partial t` coefficient to also carry a flux term depending on a spatial derivative — it cannot directly represent a pure reaction ODE. That script works around this by giving `[Ab]_bound` and `[Ag]` a tiny artificial diffusivity (`D_reg`, chosen small enough that its diffusion length over the whole simulated time is negligible next to the domain size). The finite-volume solver here has no such restriction: `[Ab]_bound` and `[Ag]` are implemented as exactly what they physically are — local reaction ODEs with **zero** spatial coupling — which is both simpler and marginally more accurate (no artificial diffusion at all, rather than a deliberately negligible one).

---

## Step 5 — Assembling the full ODE system

Collect the three fields at all `N` nodes into one state vector `\mathbf{y} = (\,[Ab]_{free,0..N-1},\ [Ab]_{bound,0..N-1},\ [Ag]_{0..N-1}\,) \in \mathbb{R}^{3N}`. Writing `L_{op}` for the `N\times N` sparse tridiagonal matrix assembled in Step 3–4 (interior rows + the two boundary rows, all *without* the plasma-forcing term) and `\boldsymbol{\rho}=(\rho_0,0,\dots,0)` for the forcing vector, the semi-discrete system is:

$$
\frac{d\,[Ab]_{free}}{dt} = L_{op}\,[Ab]_{free} + \boldsymbol{\rho}\,[Ab]_{plasma}(t) - \frac{k_{on}}{\varepsilon}\,[Ab]_{free}\odot[Ag] + k_{off}\,[Ab]_{bound}
$$

$$
\frac{d\,[Ab]_{bound}}{dt} = \frac{k_{on}}{\varepsilon}\,[Ab]_{free}\odot[Ag] - (k_{off}+k_e)\,[Ab]_{bound}
$$

$$
\frac{d\,[Ag]}{dt} = R_s\,\mathbf{1} - \frac{k_{on}}{\varepsilon}\,[Ab]_{free}\odot[Ag] + k_{off}\,[Ab]_{bound} - k_e\,[Ag]
$$

where `\odot` is the elementwise (Hadamard) product — each node's reaction term depends only on that same node's concentrations, exactly the "purely local" structure noted in Step 2. This is precisely `make_rhs` in the code: `Lop @ free` is one sparse matrix–vector product (`O(N)` cost, since `Lop` is tridiagonal/banded), and every other term is an elementwise array operation.

---

## Step 6 — Why this system is stiff (and what that means for time integration)

A system of ODEs is **stiff** when it contains widely separated timescales, forcing an *explicit* method's stable step size down to the fastest scale even though the solution itself is smooth on the slow scale — wasting enormous numbers of tiny steps just to remain stable, not to remain accurate. Two independent sources of stiffness are present here:

1. **Diffusion.** For an explicit scheme, the stability restriction on a diffusion operator with grid spacing `\Delta r` scales as `\Delta t \lesssim \Delta r^2/(2D)`. With `D=10\ \mu m^2/s`, `\Delta r \approx (75-10)/99 \approx 0.66\ \mu m` (the grid used by `krogh_visualize.py`), this bound is `\Delta t \lesssim 0.66^2/20 \approx 0.022\ s` — yet the simulation runs out to `t_{max}=10` days `=864{,}000\ s`, i.e. roughly `4\times10^7` explicit steps would be needed for stability alone, regardless of how smoothly the solution actually evolves at late times.
2. **Reaction kinetics.** The linearized reaction Jacobian (Step 7) has an eigenvalue of order `k_{on}[Ag]/\varepsilon`. With `k_{on}=10^{-3}\ \text{nM}^{-1}\text{s}^{-1}`, `\varepsilon=0.24`, `[Ag]\approx100\ nM`, this is `\sim0.4\ s^{-1}` — a relaxation time of `\sim2.4\ s` — again orders of magnitude faster than the `k_e\sim10^{-5}\ s^{-1}` (day-scale) and `k_b\sim6\times10^{-7}\ s^{-1}` (two-week-scale) processes that govern the *observable* long-time behavior (visible in the validation plot, `figG`, where the total-antibody curve rises over ~2 days and decays over ~10).

**Consequence:** an explicit integrator (forward Euler, explicit Runge–Kutta) is not merely inaccurate here, it is numerically *unstable* unless it takes steps far smaller than needed to resolve the physics anyone actually cares about. The appropriate tool is an **implicit** method, which trades a linear (or, for a nonlinear ODE, nonlinear) solve at each step for unconditional (or much less restrictive) stability, so the step size can instead be chosen by *accuracy* considerations on the slow, physically relevant timescale.

---

## Step 7 — Time integration: implicit stiff solver + analytic Jacobian

`solve_krogh_pde` integrates the system with `scipy.integrate.solve_ivp(..., method="BDF")`. **BDF** (Backward Differentiation Formulas) is a family of implicit linear multistep methods purpose-built for stiff systems — the same class of method underlying MATLAB's `ode15s`, which is what `pdepe` (the MATLAB reference implementation) calls internally. (`method="Radau"`, an implicit Runge–Kutta alternative, is also stiff-appropriate and available as a drop-in swap via the same `method` argument, useful as a cross-check with a differently-structured solver — see Step 9.)

**Every implicit step requires solving a nonlinear system** `\mathbf{y}_{n+1} - \Delta t\,\beta\,\mathbf{f}(t_{n+1},\mathbf{y}_{n+1}) = (\text{known terms from previous steps})` for `\mathbf{y}_{n+1}`, and `solve_ivp`'s BDF implementation does this with a Newton iteration — which needs the Jacobian `\partial\mathbf{f}/\partial\mathbf{y}` at every step (or every few steps, when reused). Supplying it **analytically**, rather than letting `solve_ivp` estimate it by finite differences, is both faster (no extra `f` evaluations to build a numerical Jacobian) and more accurate. Differentiating the RHS in Step 5 termwise:

$$
\frac{\partial}{\partial [Ab]_{free}}\left(\frac{d\,[Ab]_{free}}{dt}\right) = L_{op} - \frac{k_{on}}{\varepsilon}\,\mathrm{diag}([Ag]), \qquad
\frac{\partial}{\partial [Ab]_{bound}}\left(\frac{d\,[Ab]_{free}}{dt}\right) = k_{off}\,I, \qquad
\frac{\partial}{\partial [Ag]}\left(\frac{d\,[Ab]_{free}}{dt}\right) = -\frac{k_{on}}{\varepsilon}\,\mathrm{diag}([Ab]_{free})
$$

and similarly for the `[Ab]_{bound}` and `[Ag]` rows (`make_jac` in the code implements all nine `N\times N` blocks). Every block is either the fixed tridiagonal `L_{op}`, a multiple of the identity, or a diagonal matrix built from the current state — so the full `3N\times3N` Jacobian is assembled as a **sparse block matrix** (`scipy.sparse.bmat`) with `O(N)` nonzeros, and each Newton solve during a BDF step costs `O(N)` (banded/sparse LU) rather than the `O(N^3)` a dense solve would cost. This mirrors, at the level of *this* solver's data structures, the same efficiency argument `solver.py`'s `_thomas_solve` makes for the 1D Crank–Nicolson scheme's tridiagonal system.

**Adaptive step size.** `solve_ivp` chooses `\Delta t` internally to satisfy the requested local error tolerances `rtol, atol` (defaults `1e-7, 1e-9` here) at every accepted step — small steps automatically where the solution changes fast (the initial transient as antibody first extravasates), large steps where it is slowly varying (the late-time decay). This is precisely the behavior needed given the many-decades-wide range of timescales identified in Step 6, and is why `t_eval` is requested on a **log-spaced grid** (`np.logspace`) covering `1\ s` to `10` days — matching how fast the underlying dynamics actually move, rather than a linear grid that would over-resolve late time and under-resolve early time.

---

## Step 8 — Validation 1: reduction to the compartmental (0D) model

A fully independent check on both the physics and the numerics: Thurber & Wittrup (2012) derive a **compartmental** (spatially-averaged, 0D) ODE for total tumor antibody `[Ab]_{total}(t)` by assuming the radial profile is always close to a *quasi-steady* Ab-free-diffusion profile (valid once diffusion equilibrates faster than the plasma-clearance/internalization timescales — the "sub-saturating" regime, `[Ab]_{total} \ll [Ag]`, used by the default parameters here) and integrating the full spatial problem over the shell. Their closed-form result (their Eq. 7–8, reproduced in `compartmental_ab_ratio`):

$$
\frac{[Ab]_{total}(t)}{[Ab]_{plasma,0}} = k_{ex}\left[\frac{A}{\Omega-k_a}\big(e^{-k_at}-e^{-\Omega t}\big) + \frac{B}{\Omega-k_b}\big(e^{-k_bt}-e^{-\Omega t}\big)\right], \quad
\Omega = \frac{k_{ex}K_d}{Ag_0+K_d} + \frac{k_eAg_0}{Ag_0+K_d}, \quad
k_{ex} = \frac{2PR_{cap}}{R_{Krogh}^2}.
$$

This has **no spatial resolution at all** — it is a single ODE for one lumped quantity — but it uses the *same* underlying rate constants (`k_{ex}` is derived from the same `P, R_{cap}, R_{Krogh}` that appear in this solver's Robin BC), so it is a meaningful check: if the spatial PDE is discretized and integrated correctly, its **volume-averaged** total antibody

$$
\overline{[Ab]_{total}}(t) = \frac{2}{R_{Krogh}^2-R_{cap}^2}\int_{R_{cap}}^{R_{Krogh}} r\,\big([Ab]_{free}(r,t)+[Ab]_{bound}(r,t)\big)\,dr
$$

(`volume_averaged` in the code, using the trapezoidal rule with the `r`-weighting that makes it a true area/volume average in cylindrical coordinates) should track the compartmental prediction closely in the sub-saturating regime. Running `krogh_solver.py`'s self-check with the default parameters gives a **maximum relative error of ~1.1%, median ~1.0%** over the full 10-day simulation (`figG_validation.png` shows the two curves visually indistinguishable) — strong evidence that both the spatial discretization and the nonlinear reaction/BC terms are implemented consistently with the reference model, using an entirely independent numerical method (finite-volume MOL vs. a closed-form ODE solution) and derivation path.

---

## Step 9 — Validation 2: grid-refinement (empirical order of convergence)

Because Step 3 predicted the interior scheme is second-order accurate (`O(\Delta r^2)`), that prediction is directly testable: solve the same problem at `N = 25, 50, 100, 200` and look at how fast the volume-averaged total-antibody trajectory converges as `\Delta r` is halved. If the true error is `C\,\Delta r^p`, then doubling resolution should shrink the difference between successive solutions by a factor `2^p`, i.e. `\log_2(e_{k}/e_{k+1}) \approx p`. Running this (the `__main__` block of `krogh_solver.py`) gives:

```
nr 25->50:  4.25e-02   nr 50->100: 1.01e-02   nr 100->200: 2.48e-03
observed order ~ log2(e1/e2) = 2.07, log2(e2/e3) = 2.03
```

Both ratios are within a few percent of the predicted order `p=2` — an independent confirmation of the truncation-error analysis in Step 3, obtained purely numerically (no reference to the compartmental model at all), and a check that specifically targets the *spatial* discretization in isolation from everything else (Step 8 could in principle mask a spatial discretization error if it happened to cancel against a different error source; grid refinement cannot, since it only varies `N`).

The two validations are complementary: Step 8 checks against an independent *physical* model (different equations, same physics, same parameters), Step 9 checks internal *numerical* consistency (same equations, varying discretization). Agreement on both gives good confidence that the residual ~1% discrepancy in Step 8 is the physical model-reduction error of the compartmental approximation itself (finite `Ag_0/K_d`, not-quite-instantaneous radial equilibration) rather than a numerical artifact of this solver.

---

## Step 10 — Interpreting the physics visible in the figures

- **`figD` (antigen heatmap) shows depletion of `[Ag]` nearest the vessel wall, recovering back to baseline `Ag_0` toward `R_{Krogh}`.** This is the spatial signature of the **binding-site barrier** effect (Thurber, Zajic & Wittrup, J Nucl Med 2007) — antibody entering near the vessel encounters plentiful free antigen and binds avidly before it can diffuse further, so the outer tissue can remain relatively unexposed even while the vessel-adjacent region is heavily bound. A well-mixed (0D) model cannot show this at all — it is a genuinely spatial phenomenon and part of the reason a compartmental model needs its own quasi-steady-state assumption (Step 8) to be valid in the first place.
- **`figB`/`figE` (free-antibody heatmap/profiles) show a temporal peak (~day 2, matching the plasma-PK alpha-to-beta-phase transition) rather than monotonic accumulation.** This is expected: `[Ab]_{free}` tracks (with some lag from diffusion/binding) the plasma driving concentration `[Ab]_plasma(t)`, which itself has a fast initial decline (`k_a` phase) before settling into its much slower terminal (`k_b`) decline — the local free-antibody profile at any `r` mirrors this two-phase shape once the initial transient (governed by the diffusion length `\sqrt{Dt}` reaching that `r`) has passed.
- **`figG` (validation).** Beyond validating the numerics (Step 8), this plot is itself the physical bridge between this document and `ANALYTICAL_SOLVER_MATH.md`: the peak-then-decay shape here is the *fully spatially-resolved* counterpart of what the earlier document's simpler 1D model was a first approximation to — the same "how much drug reaches and stays in the tumor" question, now answered with the antigen-binding and capillary-permeability physics the 1D model abstracted away.

---

## Step 11 — Map back to the code

| Math object | Symbol | Code (`krogh_solver.py`) |
|---|---|---|
| Radial grid, spacing | `r_i, \Delta r` | `build_radial_grid`, `r` |
| Cell/face radii | `r_{i\pm1/2}` | `r_half` |
| Interior finite-volume stencil | `(Lc)_i` | `main[i], lower[i-1], upper[i]` loop in `build_diffusion_operator` |
| Inner boundary (Robin) row + forcing | `(Lc)_0, \rho_0` | `main[0], upper[0], robin_coef[0]` |
| Outer boundary (Neumann) row | `(Lc)_{N-1}` | `main[-1], lower[-1]` |
| Assembled diffusion operator | `L_{op}` | `Lop` (`scipy.sparse` tridiagonal) |
| Full state vector | `\mathbf{y}=([Ab]_{free},[Ab]_{bound},[Ag])` | `y` (length `3N`), unpacked by `_unpack` |
| Semi-discrete ODE system | `d\mathbf{y}/dt=\mathbf{f}(t,\mathbf{y})` | `make_rhs(...)` |
| Analytic Jacobian | `\partial\mathbf{f}/\partial\mathbf{y}` | `make_jac(...)` (`scipy.sparse.bmat`) |
| Stiff implicit time integration | BDF | `solve_ivp(..., method="BDF", jac=jac)` in `solve_krogh_pde` |
| Compartmental cross-check | `[Ab]_{total}(t)/[Ab]_{plasma,0}` | `compartmental_ab_ratio` |
| Volume (area) average | `\overline{[Ab]_{total}}(t)` | `volume_averaged` |

---

## References

- Thurber GM, Wittrup KD. *A mechanistic compartmental model for total antibody uptake in tumors.* J Theor Biol. 2012;314:57-68. (`papers/2012 Thurber cylinder JTB.pdf`) — source of the Krogh-cylinder geometry, boundary conditions, and compartmental reduction (Step 8) this solver validates against.
- Thurber GM, Zajic SC, Wittrup KD. *Theoretic criteria for antibody penetration into solid tumors and micrometastases.* J Nucl Med. 2007;48(6):995-999 — source of the spatial Krogh-cylinder model this solver directly numerically integrates, and of the "binding-site barrier" phenomenon discussed in Step 10.
- LeVeque RJ. *Finite Difference Methods for Ordinary and Partial Differential Equations.* SIAM, 2007 — standard reference for the conservative finite-volume/ghost-node methodology used in Steps 3–4, and for stiffness/implicit time-integration (Steps 6–7).
- Companion documents in this repo: [`ANALYTICAL_SOLVER_MATH.md`](ANALYTICAL_SOLVER_MATH.md) (the simpler 1D linear analogue, solved in closed form), [`HUONG_DAN_PROJECT.md`](HUONG_DAN_PROJECT.md) (project roadmap, Vietnamese), [`../README.md`](../README.md) (project overview).
