# Mathematical Derivation of the Analytical Solver — 1D Diffusion–Degradation

**Companion code:** [`src/diffusion_degradation/solver.py`](../src/diffusion_degradation/solver.py) — functions `steady_state_profile` and `analytical_transient_profile`.

This document derives the closed-form (analytical) solution used by the solver, **from first principles, one step at a time**. Each step only uses results already established in the steps before it — read in order. Steps 1–8 reconstruct exactly what the code computes; Steps 9–12 go a bit further (closed-form Fourier coefficients, convergence analysis, worked numbers) to *analyze* the solution rather than just state it. Step 13 gives a few self-check exercises so you can verify you can reproduce the derivation yourself.

---

## Step 0 — Problem statement

We solve the linear reaction–diffusion equation on a finite interval:

$$
\frac{\partial c}{\partial t} = D\,\frac{\partial^2 c}{\partial x^2} - r\,c ,
\qquad x \in [0, L],\ t \ge 0
$$

with boundary conditions

$$
c(0, t) = c_0 \quad \text{(Dirichlet — fixed source)}
$$
$$
\left.\frac{\partial c}{\partial x}\right|_{x=L} = 0 \quad \text{(Neumann — zero flux)}
$$

and initial condition

$$
c(x, 0) = 0 .
$$

**Physical reading (Krogh-cylinder analogy).** `x=0` is the vessel wall, held at constant concentration `c₀` (e.g. plasma antibody/ADC concentration). `x=L` is the outer radius of the tissue cylinder surrounding the vessel — by symmetry with the neighboring vessel, no flux crosses this plane. `D` (µm²/s) is the diffusion coefficient of the molecule in tissue; `r` (s⁻¹) lumps every first-order loss process (internalization, degradation, or — with the sign flipped — pseudo-first-order binding when the tissue is far from antigen saturation, cf. Thurber & Wittrup 2012, eq. 1–3). No molecule is present in the tissue before dosing, hence `c(x,0)=0`.

**Why an analytical solution is worth deriving at all.** The PDE is *linear* with *constant coefficients* `D, r` — this is what makes a closed-form solution possible. It is also why `generate_data.py` can synthesize hundreds of thousands of ground-truth profiles cheaply (evaluate a formula) instead of running a PDE solver for each one; the numerical Crank–Nicolson solver (`fdm_crank_nicolson`) exists only to *validate* this formula and to handle the case — spatially varying `D(x), r(x)` — where linearity, and hence this whole derivation, breaks down (Task 1.4).

---

## Step 1 — Characteristic length scale

Before solving anything, non-dimensional reasoning tells us what shape of answer to expect. Balancing the diffusion term against the reaction term,

$$
D\,\frac{c}{\lambda^2} \sim r\,c \quad\Longrightarrow\quad \lambda = \sqrt{\frac{D}{r}} .
$$

`λ` (units: length) is the **penetration depth** — the distance over which a concentration imposed at `x=0` survives before reaction/loss consumes it faster than diffusion can replenish it. Two limits:

- `λ ≫ L`: diffusion wins everywhere in `[0,L]` — expect a nearly flat profile equal to `c₀`.
- `λ ≪ L`: reaction wins away from the source — expect the profile to collapse to ~0 within a thin layer near `x=0`.

We will find `λ` re-appearing explicitly in the steady-state solution (Step 2) — confirming the scaling argument was right — and it is the same characteristic quantity used throughout the antibody/ADC tumor-penetration literature (Thurber et al. 2007; Thurber & Weissleder 2011) to describe how far a molecule diffuses into tissue before internalization/degradation removes it; it underlies the Biot-number discussion in Thurber & Wittrup (2012) even though that paper's headline result is a lumped, 0-D reduction of the same physics.

---

## Step 2 — Steady-state solution

**Strategy for the full problem.** The PDE is linear with a *time-independent* inhomogeneous boundary condition (`c(0,t)=c₀` for all `t`). This is the classical setting for splitting the solution into a **steady part** (satisfies the PDE and both boundary conditions, no time dependence) plus a **transient part** (decays to zero, absorbs the initial condition mismatch). We build the steady part first.

**Where the boundary conditions for `c_ss` come from.** The original BCs `c(0,t)=c_0` and `∂c/∂x|_{x=L}=0` hold for *every* `t ≥ 0` — the source concentration at the wall never changes, and the no-flux plane at `x=L` is a permanent symmetry condition. `c_ss` is defined as the value `c(x,t)` settles to once time-dependence has died out, so it is itself one particular instance of `c(x,t)` (namely, in the `t → ∞` limit). Since the two BCs are true at *every* `t`, they remain true in that limit as well:

$$
c(0,t) = c_0\ \ \forall t \;\Longrightarrow\; c_{ss}(0) = c_0, \qquad
\left.\frac{\partial c}{\partial x}\right|_{x=L} = 0\ \ \forall t \;\Longrightarrow\; c_{ss}'(L) = 0 .
$$

No new physics is introduced here — `c_ss` simply inherits, unchanged, the same two conditions the full solution `c(x,t)` was already required to satisfy.

Set `∂c/∂t = 0` in the PDE (this is literally the definition of "steady state": concentration no longer changes with time at any `x`), which removes `t` entirely and leaves an ODE purely in `x`:

$$
D\,c_{ss}''(x) - r\,c_{ss}(x) = 0, \qquad c_{ss}(0) = c_0,\quad c_{ss}'(L) = 0 .
$$

**Solving the ODE — from characteristic equation to general solution.** This is a constant-coefficient, linear, homogeneous, second-order ODE, so we try an exponential ansatz `c_ss(x) = e^{kx}` (exponentials are the natural guess because their derivatives are proportional to themselves, matching the structure of every term in the equation). Then `c_ss''(x) = k^2 e^{kx}`, and substituting into the ODE:

$$
D\,k^2 e^{kx} - r\,e^{kx} = 0 \;\Longrightarrow\; e^{kx}\left(Dk^2 - r\right) = 0 .
$$

Since `e^{kx} ≠ 0` for any `x`, the only way this can hold is if the bracketed factor vanishes — this is the **characteristic equation**:

$$
Dk^2 - r = 0 \;\Longrightarrow\; k^2 = \frac{r}{D} \;\Longrightarrow\; k = \pm\sqrt{\frac{r}{D}} = \pm\frac{1}{\lambda}, \qquad \lambda = \sqrt{D/r}\ \text{ from Step 1.}
$$

It turns a differential equation into an algebraic one for `k`. With two distinct real roots `k=+1/λ` and `k=-1/λ`, the general solution is the linear combination `c_ss(x) = C_1 e^{x/\lambda} + C_2 e^{-x/\lambda}`. Re-expressing this basis via `cosh(u)=(e^u+e^{-u})/2`, `sinh(u)=(e^u-e^{-u})/2` (a pure change of basis — any `C_1, C_2` maps to some `A, B` and vice versa) gives an equivalent, more convenient form:

$$
c_{ss}(x) = A\cosh(x/\lambda) + B\sinh(x/\lambda).
$$

The reason to prefer this form: `cosh(0)=1` and `sinh(0)=0`, so applying the boundary condition at `x=0` below collapses immediately instead of requiring a 2×2 solve.

**Apply `c_ss(0) = c₀`:** since `cosh(0)=1, sinh(0)=0`, this gives `A = c₀`.

**Apply `c_ss'(L) = 0`:** differentiate,

$$
c_{ss}'(x) = \frac{A}{\lambda}\sinh(x/\lambda) + \frac{B}{\lambda}\cosh(x/\lambda),
$$

so at `x=L`:

$$
\frac{c_0}{\lambda}\sinh(L/\lambda) + \frac{B}{\lambda}\cosh(L/\lambda) = 0
\quad\Longrightarrow\quad
B = -c_0\,\tanh(L/\lambda).
$$

Substituting back:

$$
c_{ss}(x) = c_0\left[\cosh(x/\lambda) - \tanh(L/\lambda)\sinh(x/\lambda)\right]
= c_0\,\frac{\cosh(x/\lambda)\cosh(L/\lambda) - \sinh(x/\lambda)\sinh(L/\lambda)}{\cosh(L/\lambda)}.
$$

The numerator is exactly the hyperbolic-cosine subtraction identity `cosh(A)cosh(B) − sinh(A)sinh(B) = cosh(A−B)` with `A = L/λ, B = x/λ`:

$$
\boxed{c_{ss}(x) = c_0\,\frac{\cosh\!\left(\dfrac{L-x}{\lambda}\right)}{\cosh\!\left(\dfrac{L}{\lambda}\right)}}
$$

This is exactly `steady_state_profile(x, D, r, c0, L)` in the code. **Sanity checks:** `c_ss(0) = c₀·cosh(L/λ)/cosh(L/λ) = c₀` ✓; and one can differentiate to confirm `c_ss'(L) = -c_0 \sinh(0)/(\lambda\cosh(L/\lambda)) = 0` ✓.

---

## Step 3 — Homogenizing the boundary conditions

The steady-state solution satisfies both boundary conditions but not the initial condition (`c_ss(x) ≠ 0` in general). Define the **deviation from steady state**:

$$
w(x,t) \;:=\; c_{ss}(x) - c(x,t).
$$

We now derive what PDE/BC/IC `w` itself satisfies, using only: (a) `c` solves the original PDE/BC/IC, (b) `c_ss` solves the steady problem from Step 2.

**PDE for `w`.** Since `c_ss` does not depend on `t`, `∂w/∂t = -∂c/∂t`. Substituting `c = c_ss - w` into the original PDE:

$$
-\frac{\partial w}{\partial t} = D\left(c_{ss}'' - \frac{\partial^2 w}{\partial x^2}\right) - r\left(c_{ss} - w\right)
= \underbrace{\left(D c_{ss}'' - r\,c_{ss}\right)}_{=\,0\ \text{by Step 2}} - D\frac{\partial^2 w}{\partial x^2} + r w .
$$

The bracketed term vanishes by construction of `c_ss`, leaving

$$
\frac{\partial w}{\partial t} = D\,\frac{\partial^2 w}{\partial x^2} - r\,w .
$$

**`w` solves the *same* PDE as `c`** — expected, since the PDE is linear and `c_ss` is itself a (time-independent) solution.

**Boundary conditions for `w`.** At `x=0`: `w(0,t) = c_ss(0) - c(0,t) = c_0 - c_0 = 0`. At `x=L`: `w_x(L,t) = c_ss'(L) - c_x(L,t) = 0 - 0 = 0`. Both boundary conditions are now **homogeneous** — this is the entire point of the transformation, since homogeneous BCs are what makes separation of variables (Step 4) work.

**Initial condition for `w`.** `w(x,0) = c_ss(x) - c(x,0) = c_ss(x) - 0 = c_ss(x)`.

So the transformed problem is: find `w(x,t)` solving the *homogeneous*-BC diffusion–reaction equation, starting from `w(x,0) = c_ss(x)`. This is now a textbook Sturm–Liouville problem.

---

## Step 4 — Separation of variables

Seek solutions of the homogeneous problem of product form `w(x,t) = X(x)\,T(t)`. Substituting into `w_t = D w_{xx} - r w`:

$$
X(x) T'(t) = D\,X''(x)\,T(t) - r\,X(x)\,T(t)
\quad\Longrightarrow\quad
\frac{T'(t)}{T(t)} = D\,\frac{X''(x)}{X(x)} - r .
$$

The left side depends only on `t`, the right side only on `x`; both must equal the same constant, call it `-μ` (the sign is chosen so that `μ>0` gives decaying, physically sensible solutions):

$$
T'(t) = -\mu\,T(t) \qquad\Longrightarrow\qquad T(t) = e^{-\mu t}
$$

$$
D\,X''(x) - r\,X(x) = -\mu\,X(x)
\quad\Longrightarrow\quad
X''(x) + k^2 X(x) = 0, \qquad k^2 := \frac{\mu - r}{D} .
$$

This is a harmonic-oscillator ODE with general solution `X(x) = A\cos(kx) + B\sin(kx)`.

---

## Step 5 — Applying the homogeneous boundary conditions (the eigenvalue problem)

**`X(0)=0`** (from `w(0,t)=0`, and `T(t)` is not identically zero): `A\cos(0) + B\sin(0) = A = 0`. So `X(x) = B\sin(kx)`.

**`X'(L)=0`** (from `w_x(L,t)=0`): `X'(x) = Bk\cos(kx)`, so `Bk\cos(kL) = 0`. Non-trivial solutions require `B\neq0, k\neq0`, hence

$$
\cos(kL) = 0 \quad\Longrightarrow\quad kL = \frac{(2n-1)\pi}{2},\quad n=1,2,3,\dots
$$

$$
\boxed{k_n = \frac{(2n-1)\pi}{2L}}, \qquad X_n(x) = \sin(k_n x).
$$

Only a discrete, countable set of spatial frequencies `k_n` is compatible with the boundary conditions — these are the **eigenfunctions** `\sin(k_n x)` of the Sturm–Liouville problem, with **eigenvalues**

$$
\mu_n = D k_n^2 + r
$$

(from the definition `k^2 = (\mu-r)/D` in Step 4, solved for `μ`). This matches `k_n = (2*n - 1) * np.pi / (2 * L)` and `mu_n = D * k_n**2 + r` in `analytical_transient_profile`. Note each eigenfunction *individually* satisfies `X_n'(L) = k_n\cos(k_nL) = 0` — a fact we reuse in Step 9.

---

## Step 6 — Orthogonality of the eigenfunctions

To extract how much of each eigenfunction is present in the initial condition (Step 7), we need `\{\sin(k_n x)\}` to be an **orthogonal set** on `[0,L]`. Using the product-to-sum identity `\sin A \sin B = \tfrac12[\cos(A-B) - \cos(A+B)]`:

$$
\int_0^L \sin(k_n x)\sin(k_m x)\,dx
= \frac{1}{2}\left[\frac{\sin\big((k_n-k_m)L\big)}{k_n-k_m} - \frac{\sin\big((k_n+k_m)L\big)}{k_n+k_m}\right], \qquad n\neq m.
$$

Since `k_n L = (2n-1)\pi/2`, we get `(k_n-k_m)L = (n-m)\pi` and `(k_n+k_m)L = (n+m-1)\pi` — both **integer multiples of `π`**, so both sine terms vanish:

$$
\int_0^L \sin(k_n x)\sin(k_m x)\,dx = 0, \qquad n \neq m.
$$

For `n=m`, using `\sin^2\theta = \tfrac12(1-\cos2\theta)`:

$$
\int_0^L \sin^2(k_n x)\,dx = \frac{L}{2} - \frac{\sin(2k_nL)}{4k_n} = \frac{L}{2},
$$

because `2k_nL = (2n-1)\pi` makes `\sin(2k_nL)=0` too. So the norm is the **same for every mode**, `\|\sin(k_n\cdot)\|^2 = L/2` — a convenient simplification that gives the clean `2/L` prefactor in the projection formula below.

---

## Step 7 — Fourier coefficients (matching the initial condition)

We now expand `w(x,0) = c_{ss}(x)` in the eigenbasis:

$$
w(x,t) = \sum_{n=1}^{\infty} B_n \sin(k_n x)\, e^{-\mu_n t}, \qquad
w(x,0) = \sum_{n=1}^\infty B_n \sin(k_n x) \overset{!}{=} c_{ss}(x).
$$

Multiply both sides by `\sin(k_m x)`, integrate over `[0,L]`, and use orthogonality (Step 6) to collapse the sum to a single term:

$$
\int_0^L c_{ss}(x)\sin(k_m x)\,dx = B_m \int_0^L \sin^2(k_m x)\,dx = B_m \cdot \frac{L}{2}
$$

$$
\boxed{B_n = \frac{2}{L}\int_0^L c_{ss}(x)\sin(k_n x)\,dx}
$$

This is exactly what the code computes — numerically, via `_trapezoid` over a 2000-point fine grid (`I_n`, then `B_n = (2.0/L) * I_n`) rather than symbolically. Step 9 derives the closed form and explains why the code nonetheless integrates numerically.

---

## Step 8 — Assembling the full solution

Putting Steps 2, 5, and 7 together:

$$
w(x,t) = \sum_{n=1}^\infty B_n \sin(k_n x)\, e^{-\mu_n t}
$$

$$
\boxed{c(x,t) = c_{ss}(x) - w(x,t) = c_{ss}(x) - \sum_{n=1}^\infty B_n \sin(k_n x)\, e^{-\mu_n t}}
$$

This is precisely the return value of `analytical_transient_profile`: `profile = c_ss[:, None] - w`, with `w = sin_nx.T @ (B_n[:, None] * decay)` and `decay = exp(-mu_n * t)`. In practice the sum is truncated at `n_modes` terms (default 200 in the function signature; 150 is used in `generate_data.py`, 300 in the `solver.py` self-check) — Step 10 quantifies why that many terms are needed.

---

## Step 9 — Verification (does this actually solve the problem?)

It is worth checking the assembled solution against every original requirement, term by term:

1. **PDE.** Each term `B_n\sin(k_nx)e^{-\mu_nt}` solves `w_t = Dw_{xx}-rw` by construction (Step 4–5), and a sum of solutions of a linear PDE is again a solution (superposition). `c_{ss}(x)` solves the steady PDE (`c_{ss,t}=0` trivially). So `c = c_{ss} - w` solves the original PDE. ✓
2. **Left boundary.** `c(0,t) = c_{ss}(0) - \sum B_n\sin(0)e^{-\mu_nt} = c_0 - 0 = c_0`. ✓ (every `\sin(k_n\cdot0})=0`.)
3. **Right boundary.** `\partial_x c(L,t) = c_{ss}'(L) - \sum B_n k_n\cos(k_nL)e^{-\mu_nt} = 0 - 0 = 0`, using `c_{ss}'(L)=0` (Step 2) and `\cos(k_nL)=0` for every mode (Step 5). ✓
4. **Initial condition.** `c(x,0) = c_{ss}(x) - \sum B_n\sin(k_nx) = c_{ss}(x) - c_{ss}(x) = 0` **by construction** of `B_n` in Step 7 (this is exactly what the projection was designed to guarantee, in the sense of `L^2` convergence of the Fourier sine series; see Step 10 for how many terms are needed for this equality to hold to a given numerical tolerance). ✓

All four requirements are met — the derivation is self-consistent, independent of the empirical analytical-vs-FDM comparison in the code (which is a second, independent check — see Step 10).

---

## Step 10 — Going further: closed-form coefficients, and why the code still integrates numerically

Because `c_{ss}(x)` has an explicit `cosh` formula (Step 2), the integral defining `B_n` (Step 7) can be evaluated **exactly**, without numerical quadrature. This is not needed to run the code, but it is worth doing once, both as an independent check on `B_n` and because it exposes the *decay rate* of the coefficients, which controls how many modes (`n_modes`) are actually needed.

Write `a := 1/\lambda = \sqrt{r/D}`, so `c_{ss}(x) = c_0\cosh(a(L-x))/\cosh(aL)`. Substitute `u = L-x` in the integral:

$$
I_n := \int_0^L \cosh(a(L-x))\sin(k_nx)\,dx = \int_0^L \cosh(au)\,\sin\big(k_n(L-u)\big)\,du .
$$

Expand `\sin(k_n(L-u)) = \sin(k_nL)\cos(k_nu) - \cos(k_nL)\sin(k_nu)`. From Step 5, `\cos(k_nL)=0` and (checking the pattern `n=1,2,3,\dots \to \sin(k_nL)=\sin(\pi/2),\sin(3\pi/2),\sin(5\pi/2),\dots = +1,-1,+1,\dots`) `\sin(k_nL) = (-1)^{n-1}`. So the second term drops out entirely:

$$
I_n = (-1)^{n-1}\int_0^L \cosh(au)\cos(k_nu)\,du .
$$

Using the standard antiderivative `\int\cosh(au)\cos(ku)\,du = \dfrac{a\sinh(au)\cos(ku) + k\cosh(au)\sin(ku)}{a^2+k^2}` (verified by differentiating), evaluated at `u=0` (gives `0`) and `u=L` (using `\cos(k_nL)=0,\ \sin(k_nL)=(-1)^{n-1}` again):

$$
\int_0^L \cosh(au)\cos(k_nu)\,du = \frac{k_n\cosh(aL)\,(-1)^{n-1}}{a^2+k_n^2}
$$

$$
I_n = (-1)^{n-1}\cdot\frac{k_n\cosh(aL)(-1)^{n-1}}{a^2+k_n^2} = \frac{k_n\cosh(aL)}{a^2+k_n^2}
$$

(the two `(-1)^{n-1}` factors square to `1`). Finally, from Step 7, `B_n = \frac{2}{L}\cdot\frac{c_0}{\cosh(aL)}\cdot I_n`:

$$
\boxed{B_n = \frac{2c_0\,k_n}{L\left(a^2+k_n^2\right)} = \frac{2c_0\,D\,k_n}{L\,\mu_n}}, \qquad a^2=\frac{r}{D},\ \ \mu_n = D(a^2+k_n^2)
$$

**Why the code does not use this formula.** `generate_data.py` and `analytical_transient_profile` compute `B_n` by numerical quadrature (`_trapezoid`) instead of this closed form. This is a deliberate simplicity/robustness trade-off: the numerical projection works unchanged if `c_{ss}(x)` is ever replaced by a different (e.g. numerically-obtained) steady-state profile — which is exactly what happens in Task 1.4 once `D(x), r(x)` become spatially heterogeneous and the `cosh` closed form no longer exists. The closed form above is still useful as an independent correctness check on the numerical `B_n` (see Step 13, Exercise 2) and as the basis for the convergence analysis in Step 11.

---

## Step 11 — Convergence and truncation error (why `n_modes ≈ 150–300`)

For large `n`, `k_n \approx (2n-1)\pi/(2L)` grows linearly in `n`, so in the closed form of Step 10,

$$
B_n = \frac{2c_0 k_n}{L(a^2+k_n^2)} \;\xrightarrow[n\to\infty]{}\; \frac{2c_0}{L k_n} = O\!\left(\frac{1}{n}\right).
$$

The Fourier coefficients decay only **algebraically** (like `1/n`), not exponentially. In the worst case (`t=0`, where `e^{-\mu_nt}=1` for every mode), the partial sum after `N` terms has a tail error that only shrinks like `O(1/N)` — this is why the code needs on the order of hundreds of modes (`n_modes=150–300`) rather than, say, 10, to reconstruct `c_{ss}(x)` accurately at `t=0^+`.

For `t>0`, the picture improves dramatically: the extra factor `e^{-\mu_nt} = e^{-(Dk_n^2+r)t}` decays **Gaussian-fast** in `n` (since `k_n\propto n`), so the tail

$$
\left|\sum_{n>N} B_n\sin(k_nx)e^{-\mu_nt}\right| \;\lesssim\; \sum_{n>N} \frac{2c_0}{Lk_n}\,e^{-D k_n^2 t}
$$

is dominated by the exponential once `D k_N^2 t \gtrsim 1`, i.e. once `N \gtrsim \frac{L}{\pi}\sqrt{2/(Dt)}`. **Fewer modes are needed for later times, more are needed for early times** — exactly the pattern that would be expected from a diffusion process (high-frequency spatial detail near the source dies out fastest).

This prediction matches the numbers already recorded in [`HUONG_DAN_PROJECT.md §4.8`](HUONG_DAN_PROJECT.md): comparing the analytical series (`n_modes=300`) against the independent Crank–Nicolson solver for `D=10, r=10^{-3}, L=100`, the worst-case relative error **decreases monotonically with `t`** — `2.7\times10^{-5}` at `t=50\,s` down to `2.1\times10^{-6}` at `t\ge5000\,s` — consistent with slower-converging high-frequency modes mattering more at early times. This is an independent, purely numerical confirmation of the analysis above; the two ways of checking the solution (Step 9's symbolic verification and this empirical FDM comparison) agree.

---

## Step 12 — Worked numerical example

Take the parameters from the `solver.py` self-check: `D=10\ \mu m^2/s,\ r=10^{-3}\ s^{-1},\ c_0=1,\ L=100\ \mu m`. Then:

$$
\lambda = \sqrt{D/r} = \sqrt{10/0.001} = \sqrt{10000} = 100\ \mu m = L .
$$

A notable coincidence in this test case — the penetration depth exactly equals the domain length, i.e. `cosh(L/\lambda) = \cosh(1) \approx 1.5431`, so the steady-state profile decays to about `c_{ss}(L)/c_0 = 1/\cosh(1) \approx 0.648` at the far boundary rather than to zero (with `\lambda \ll L` it would decay to nearly 0; with `\lambda \gg L` it would stay near `c_0` everywhere — see Step 1).

First three modes (`a=1/\lambda=0.01\ \mu m^{-1}`):

| `n` | `k_n` (µm⁻¹) | `μ_n` (s⁻¹) | `B_n` (closed form, Step 10) |
|---|---|---|---|
| 1 | 0.015708 | 0.003467 | 0.906 |
| 2 | 0.047124 | 0.023207 | 0.406 |
| 3 | 0.078540 | 0.062685 | 0.251 |

The `B_n` column shows the expected `O(1/n)`-ish decrease from Step 11 (not exactly `1/n` yet at such small `n`, since the `a^2` term in the denominator is still comparable to `k_n^2` — the pure `1/n` asymptotic only kicks in once `k_n \gg a`). You can reproduce this table directly from the boxed formulas in Steps 5 and 10, or numerically by calling `analytical_transient_profile` with `n_modes=1,2,3` and inspecting the internal `B_n` array.

---

## Step 13 — Self-check exercises

Try these without looking back, then check against the referenced step.

1. **Limit check.** What does `c_{ss}(x)` (Step 2) reduce to as `r\to0` (no loss at all)? *(Hint: `\lambda\to\infty`, both `\cosh` terms `\to1`.)* Does the physical answer make sense? — *Answer: `c_{ss}(x)\to c_0` for all `x`: with no loss term, steady state is just the source value propagated everywhere, matching the `\lambda\gg L` limit of Step 1.*
2. **Independent check of `B_1`.** Using the closed form from Step 10 with the numbers from Step 12, verify `B_1 \approx 0.906` by hand, then compare it against what the numerical quadrature in the code would produce for the same `n=1` (they should agree to several significant digits — this is a good way to confirm you haven't mis-signed something).
3. **Why does `\cos(k_nL)=0` matter twice?** It is used both to fix the allowed `k_n` in Step 5 *and* to simplify the integral in Step 10. Explain in one sentence why the same condition shows up in both places. — *Answer: it is the boundary condition `X_n'(L)=0` itself, restated; every eigenfunction individually satisfies the zero-flux condition, which is exactly why it also kills the boundary term when integrating by parts / evaluating the antiderivative at `u=L` in Step 10.*
4. **Truncation order-of-magnitude.** Using the bound in Step 11, roughly how many modes `N` are needed for the tail to be negligible at `t=50\,s`, with `D=10, L=100`? *(Compute `N \gtrsim \frac{L}{\pi}\sqrt{2/(Dt)}`.)* — *Answer: `N \gtrsim \frac{100}{\pi}\sqrt{2/(10\cdot50)} = \frac{100}{\pi}\sqrt{0.004} \approx 31.8\times0.0632 \approx 2`. This is a coarse lower bound (it only says when the exponential starts to dominate the `1/n` prefactor, not full numerical convergence) — consistent with why the code still defaults to hundreds of modes for a comfortable safety margin across the whole time range it is asked to evaluate, including much smaller `t`.*

---

## Step 14 — Map back to the code

| Math object | Symbol | Code (`solver.py`) |
|---|---|---|
| Penetration depth `\lambda=\sqrt{D/r}` | `\lambda` | `lam` in `steady_state_profile` |
| Steady-state profile | `c_{ss}(x)` | `steady_state_profile(...)` |
| Eigenvalues (spatial frequencies) | `k_n=(2n-1)\pi/(2L)` | `k_n` |
| Eigenvalues (decay rates) | `\mu_n = Dk_n^2+r` | `mu_n` |
| Fourier projection (Step 7) | `B_n = \frac2L\int_0^Lc_{ss}\sin(k_nx)\,dx` | `I_n` via `_trapezoid`, then `B_n` |
| Transient deviation | `w(x,t)=\sum B_n\sin(k_nx)e^{-\mu_nt}` | `w = sin_nx.T @ (B_n[:,None]*decay)` |
| Full solution | `c(x,t)=c_{ss}(x)-w(x,t)` | `profile = c_ss[:, None] - w` |
| Truncation | `n \le N` | `n_modes` parameter (default 200; 150 in `generate_data.py`, 300 in the self-check) |

---

## References

- Thurber GM, Wittrup KD. *A mechanistic compartmental model for total antibody uptake in tumors.* J Theor Biol. 2012;314:57-68. (`papers/2012 Thurber cylinder JTB.pdf`) — source of the Krogh-cylinder geometry and boundary conditions this PDE mirrors.
- Thurber GM, Zajic SC, Wittrup KD. *Theoretic criteria for antibody penetration into solid tumors and micrometastases.* J Nucl Med. 2007;48(6):995-999 — the spatial (Krogh cylinder) model underlying the compartmental reduction in the 2012 paper; source of the "penetration depth" concept used in Step 1.
- Companion documents in this repo: [`HUONG_DAN_PROJECT.md`](HUONG_DAN_PROJECT.md) (theory + full inline code, Vietnamese), [`LO_TRINH_HOC.md`](LO_TRINH_HOC.md) (study roadmap), [`../README.md`](../README.md) (project overview and Methods summary).
