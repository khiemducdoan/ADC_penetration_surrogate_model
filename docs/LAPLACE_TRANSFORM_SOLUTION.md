# Solving the Diffusion–Degradation PDE via Laplace Transform (Frequency-Domain Method)

**Companion code:** [`data/synthetic/solver.py`](../data/synthetic/solver.py) — this document derives the *same* solution computed by `analytical_transient_profile`, but through a completely different route: transforming the PDE into an algebraic problem in a **frequency variable `s`**, instead of the eigenfunction-expansion approach documented in `ANALYTICAL_SOLVER_MATH.md`. The two derivations must agree — Step 8 verifies this numerically against the actual code.

Why "frequency transform" and not "Fourier transform": the natural transform for an *initial value problem* (`t ≥ 0`, one-sided) is the **Laplace transform**, not the Fourier transform (which assumes the signal exists for all `t ∈ (−∞,∞)`). The Laplace variable `s` is often called a "complex frequency" — for `s = iω` purely imaginary, the Laplace transform reduces exactly to the Fourier transform. So this derivation is the frequency-domain method appropriate to this problem.

---

## Step 0 — Problem statement (same as before)

$$
\frac{\partial c}{\partial t} = D\,\frac{\partial^2 c}{\partial x^2} - r\,c ,
\qquad x \in [0, L],\ t \ge 0
$$

$$
c(0, t) = c_0 \quad \text{(Dirichlet)}, \qquad
\left.\frac{\partial c}{\partial x}\right|_{x=L} = 0 \quad \text{(Neumann, zero-flux)}, \qquad
c(x, 0) = 0 .
$$

---

## Step 1 — Take the Laplace transform in time

Define

$$
C(x, s) = \mathcal{L}\{c(x,t)\}(s) = \int_0^\infty c(x,t)\, e^{-st}\, dt .
$$

The Laplace transform turns **time-derivatives into algebra** using the standard rule
$\mathcal{L}\{\partial c/\partial t\} = sC(x,s) - c(x,0)$. Since the initial condition is `c(x,0) = 0`, this simplifies to just `sC(x,s)` — no leftover term. Spatial derivatives pass straight through the integral (`x` and `t` are independent), so `∂²c/∂x² → d²C/dx²`.

Applying this to the PDE:

$$
sC(x,s) = D\,\frac{d^2 C}{dx^2} - rC(x,s)
\quad\Longrightarrow\quad
D\,\frac{d^2 C}{dx^2} - (s+r)\,C(x,s) = 0 .
$$

**This is the key idea of the method**: the PDE (2 independent variables) has become an **ordinary** differential equation in `x` alone, for each fixed value of `s`. All the time-dependence is now hidden inside the parameter `s`.

---

## Step 2 — Transform the boundary conditions

- `c(0,t) = c_0` for `t ≥ 0` is a constant (as a function of `t`), and $\mathcal{L}\{c_0\} = c_0/s$, so:
  $$C(0, s) = \frac{c_0}{s}.$$
- `∂c/∂x(L,t) = 0` for all `t` transforms directly (Laplace transform commutes with the `x`-derivative):
  $$\left.\frac{dC}{dx}\right|_{x=L} = 0.$$

---

## Step 3 — Solve the transformed ODE

Let

$$
\alpha(s) = \sqrt{\frac{s+r}{D}} .
$$

(Using `α` rather than `λ` or `μ` to avoid clashing with the penetration depth `λ=√(D/r)` and the decay rates `μ_n` from the other derivation — `α` depends on `s`, they don't.)

The general solution of $D C'' - (s+r)C = 0$ is

$$
C(x,s) = A(s)\cosh(\alpha x) + B(s)\sinh(\alpha x).
$$

Apply the two transformed boundary conditions:

- $C(0,s) = A(s) = c_0/s$.
- $C'(x,s) = A\alpha\sinh(\alpha x) + B\alpha\cosh(\alpha x)$, so at $x=L$:
  $$A\sinh(\alpha L) + B\cosh(\alpha L) = 0 \;\Longrightarrow\; B = -A\tanh(\alpha L).$$

Substituting back:

$$
C(x,s) = A(s)\Big[\cosh(\alpha x) - \tanh(\alpha L)\sinh(\alpha x)\Big]
= A(s)\,\frac{\cosh(\alpha L)\cosh(\alpha x) - \sinh(\alpha L)\sinh(\alpha x)}{\cosh(\alpha L)} .
$$

Using the identity $\cosh(a)\cosh(b) - \sinh(a)\sinh(b) = \cosh(a-b)$ with $a=\alpha L,\ b=\alpha x$:

$$
\boxed{\,C(x,s) = \frac{c_0}{s}\cdot\frac{\cosh\big(\alpha(s)(L-x)\big)}{\cosh\big(\alpha(s)L\big)}\,}, \qquad \alpha(s)=\sqrt{\frac{s+r}{D}} .
$$

This closed-form expression **is the complete solution — in the frequency domain**. Everything that follows is about inverting it back to the time domain.

---

## Step 4 — Where are the singularities of `C(x,s)`?

The inverse Laplace transform is computed via the Bromwich integral, which — for a function like this with only poles (no branch cuts, since `cosh` and `sinh` of `α x` are entire functions of `α`, and `α(s)` itself has no branch point at any of the poles found below) — reduces to a **sum of residues** at every pole:

$$
c(x,t) = \sum_{\text{poles } s_k} \operatorname{Res}_{s=s_k}\Big[C(x,s)\,e^{st}\Big].
$$

`C(x,s)` has two sources of poles:

**(a) The explicit `1/s` factor** → a simple pole at `s = 0`.

**(b) Zeros of `cosh(αL)` in the denominator.** Since `cosh(z)` is never zero for real `z`, these zeros must occur at purely imaginary `z`: $\cosh(z)=0 \iff z = i\frac{(2n-1)\pi}{2}$ for integer `n`. So:

$$
\alpha L = i\,\frac{(2n-1)\pi}{2} \;\Longrightarrow\; \alpha = i\,k_n, \qquad k_n := \frac{(2n-1)\pi}{2L}
$$

— **exactly the same eigenvalues `k_n` used in the eigenfunction-expansion derivation.** Substituting back into $\alpha^2 = (s+r)/D$:

$$
(ik_n)^2 = \frac{s+r}{D} \;\Longrightarrow\; -k_n^2 = \frac{s+r}{D} \;\Longrightarrow\; s = s_n := -(Dk_n^2 + r) = -\mu_n
$$

where $\mu_n = Dk_n^2+r$ is exactly the decay rate from the other derivation. **This is the payoff of the transform method**: the poles of `C(x,s)` in the complex `s`-plane sit precisely at `s=0` and at `s = −μ_n` for every `n` — the "frequencies" at which the system naturally responds are read directly off the algebra, with no separation-of-variables ansatz needed up front.

---

## Step 5 — Residue at `s = 0`: recovers the steady state

Near `s=0`, the denominator `cosh(αL)` is regular and nonzero (as noted above, `cosh` of a real argument is never zero, and `α(0) = √(r/D) = 1/λ` is real), so the only singularity is the explicit `1/s`. The residue of a simple pole from a `1/s` factor is just the numerator evaluated at `s=0`:

$$
\operatorname{Res}_{s=0}\big[C(x,s)e^{st}\big] = c_0\,\frac{\cosh\big(\alpha(0)(L-x)\big)}{\cosh(\alpha(0)L)} = c_0\,\frac{\cosh\big((L-x)/\lambda\big)}{\cosh(L/\lambda)} = c_{ss}(x).
$$

This is **exactly** `steady_state_profile` from the code — the `s=0` pole (zero frequency = "DC component") is, unsurprisingly, the part of the signal that survives as `t → ∞`.

---

## Step 6 — Residue at `s = s_n = -μ_n`: the decaying modes

This pole comes from the zero of `cosh(αL)` in the denominator, so use $\operatorname{Res} = N(s_n)/D'(s_n)$ for a simple zero of the denominator $D(s) := \cosh(\alpha(s)L)$:

$$
D'(s) = \sinh(\alpha L)\cdot L \cdot \frac{d\alpha}{ds}, \qquad \frac{d\alpha}{ds} = \frac{1}{2D\alpha} .
$$

At $s=s_n$, $\alpha_n = ik_n$, so $\sinh(\alpha_n L) = \sinh\!\big(i k_n L\big) = i\sin(k_nL)$, giving

$$
D'(s_n) = \frac{L\,i\sin(k_nL)}{2Di k_n} = \frac{L\sin(k_nL)}{2Dk_n}.
$$

The numerator (including the `e^{st}` factor and the `c_0/s` term) evaluated at $s_n$, using $\cosh(ik_n(L-x)) = \cos(k_n(L-x))$:

$$
N(s_n) = \frac{c_0\cos\big(k_n(L-x)\big)}{s_n}\,e^{s_nt}.
$$

Now simplify $\cos(k_n(L-x))$ using $k_nL = (2n-1)\pi/2$ (so $\cos(k_nL)=0$ and $\sin(k_nL) = (-1)^{n+1}$):

$$
\cos\big(k_nL - k_nx\big) = \cos(k_nL)\cos(k_nx) + \sin(k_nL)\sin(k_nx) = (-1)^{n+1}\sin(k_nx).
$$

Putting it together, the $(-1)^{n+1}$ cancels between numerator and denominator:

$$
\operatorname{Res}_{s=s_n}\big[C(x,s)e^{st}\big]
= \frac{c_0(-1)^{n+1}\sin(k_nx)\,e^{s_nt}/s_n}{L(-1)^{n+1}/(2Dk_n)}
= \frac{2c_0Dk_n}{Ls_n}\,\sin(k_nx)\,e^{s_nt}.
$$

Substituting $s_n=-\mu_n$:

$$
\boxed{\operatorname{Res}_{s=s_n}\big[C(x,s)e^{st}\big] = -\frac{2c_0Dk_n}{L\mu_n}\,\sin(k_nx)\,e^{-\mu_nt}}
$$

---

## Step 7 — Sum all residues: the full time-domain solution

$$
c(x,t) = \underbrace{c_{ss}(x)}_{\text{pole }s=0} \;-\; \sum_{n=1}^{\infty} \underbrace{\frac{2c_0Dk_n}{L\mu_n}}_{B_n}\,\sin(k_nx)\,e^{-\mu_nt}.
$$

This has the **same shape** as the eigenfunction-expansion result in `ANALYTICAL_SOLVER_MATH.md` ($c(x,t) = c_{ss}(x) - \sum_n B_n\sin(k_nx)e^{-\mu_nt}$), but here the coefficients fall out of the algebra as a **closed form**,

$$
B_n = \frac{2c_0Dk_n}{L\mu_n} = \frac{2c_0Dk_n}{L(Dk_n^2+r)},
$$

with no integral to evaluate — whereas the eigenfunction-expansion derivation gets $B_n$ by *projecting* the initial condition onto the orthogonal basis $\{\sin(k_nx)\}$, an integral that `solver.py` evaluates numerically (`_trapezoid`, 2000 quadrature points) rather than in closed form. Both routes must produce the same numbers; the transform method just happens to give a cheaper formula for this particular (homogeneous, constant-coefficient) case.

---

## Step 8 — Numerical check against the actual code

Plugging the closed-form $B_n$ into the same series structure and comparing against `analytical_transient_profile` (which computes $B_n$ via numerical projection) at `D=10 µm²/s, r=10⁻³ s⁻¹, L=100 µm, n_modes=300`:

| `t` (s) | max relative error vs. `analytical_transient_profile` |
|---|---|
| 50 | 9.6e-08 |
| 200 | 3.1e-08 |
| 1000 | 2.0e-09 |
| 5000 | 2.0e-15 |

The two independently-derived formulas for $c(x,t)$ agree to numerical precision (the residual ~1e-8 is explained by mode truncation and by `solver.py`'s $B_n$ being a numerical quadrature rather than the exact closed form) — confirming both derivations describe the same physical solution.

---

## Why bother with two derivations of the same thing

- **Eigenfunction expansion** (`ANALYTICAL_SOLVER_MATH.md`) is the standard approach taught for finite-domain BVPs: guess a separable form, find eigenfunctions that satisfy the (homogenized) boundary conditions, project the initial condition onto them.
- **Laplace transform** (this document) needs no upfront ansatz — the poles of the transformed solution `C(x,s)` *automatically* reveal the correct eigenvalues `k_n` and decay rates `μ_n` as a byproduct of algebra (finding zeros of `cosh(αL)`), and gives a closed-form `B_n` for free in this linear, constant-coefficient case.
- The method generalizes more gracefully to problems where a clean orthogonal eigenbasis is harder to guess (e.g. more complex boundary conditions, or coupled PDE systems like the ADC/receptor-binding model in `papers/2015-04-07-PLOS-ADC-modeling-paperjournal.pone.0118977.pdf`) — though for the *linear* case they must always agree, as verified above. It stops working once the PDE becomes nonlinear (e.g. saturable receptor binding), because the Laplace transform of a product of two unknown functions of `t` is not a simple product of their transforms — the same nonlinearity that breaks eigenfunction expansion breaks this method too.
