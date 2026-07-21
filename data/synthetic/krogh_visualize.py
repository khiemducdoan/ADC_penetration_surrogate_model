"""
Generate the figure set for the spatial Krogh-cylinder model (krogh_solver.py):
plasma PK driving curve, radial-vs-time heatmaps for each species, radial
snapshot profiles, a 3D surface, and the validation plot against the
compartmental Thurber & Wittrup (2012) analytical model.

Run: python3 data/synthetic/krogh_visualize.py
Figures are written to outputs/figures/krogh/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from data.synthetic.krogh_solver import (
    KroghParams,
    compartmental_ab_ratio,
    solve_krogh_pde,
    volume_averaged,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "krogh"

# Sequential colormap (perceptually uniform) for concentration-field heatmaps
# and 3D surfaces; a single-hue viridis sample for the ordered time snapshots.
SEQ_CMAP = "viridis"


def _time_axis_ticks(t_max: float) -> tuple[list[float], list[str]]:
    tickvals = [1, 10, 1e2, 1e3, 1e4, 1e5, t_max]
    return tickvals, [f"{v:g}" for v in tickvals]


def plot_plasma_pk(params: KroghParams, t: np.ndarray, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t / 86400.0, params.plasma_pk(t), lw=2.5, color="#B5541A")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$[Ab]_{plasma}$ (nM)")
    ax.set_title(r"Plasma antibody concentration (drives the BC at $r=R_{cap}$)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "figA_plasma_pk.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_heatmaps(r: np.ndarray, t: np.ndarray, fields: dict, out_dir: Path) -> list[Path]:
    tickvals, ticklabels = _time_axis_ticks(t[-1])
    paths = []
    for key, (field, label, fname) in fields.items():
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        im = ax.pcolormesh(
            np.log10(t[1:]), r, field[1:, :].T, shading="auto", cmap=SEQ_CMAP,
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"{label} (nM)")
        ax.set_xlabel("Time t (s)  [log scale]")
        ax.set_ylabel(r"Radius r ($\mu$m)")
        ax.set_title(f"{label}(r, t) — Krogh cylinder, "
                     f"$R_{{cap}}$={r[0]:g} $\\mu$m to $R_{{Krogh}}$={r[-1]:g} $\\mu$m")
        ax.set_xticks(np.log10(tickvals))
        ax.set_xticklabels(ticklabels)
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_radial_profiles(r, t, free, bound, ag, out_dir: Path) -> Path:
    snap_days = np.array([0.01, 0.1, 0.5, 1, 3, 7, 10])
    snap_t = np.minimum(snap_days * 86400.0, t[-1])
    snap_idx = [int(np.searchsorted(t, tt)) for tt in snap_t]
    colors = plt.get_cmap(SEQ_CMAP)(np.linspace(0.05, 0.95, len(snap_idx)))
    ab_total = free + bound

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (free, r"$[Ab]_{free}$ (nM)", "Free antibody"),
        (bound, r"$[Ab]_{bound}$ (nM)", "Bound antibody"),
        (ag, r"$[Ag]$ (nM)", "Free antigen (depletion near vessel = binding-site barrier)"),
        (ab_total, r"$[Ab]_{total}$ = free + bound (nM)", "Total antibody"),
    ]
    for ax, (field, ylabel, title) in zip(axes.ravel(), panels):
        for k, idx in enumerate(snap_idx):
            ax.plot(r, field[idx, :], lw=2, color=colors[k],
                    label=f"t={snap_days[k]:.2g}d")
        ax.set_xlabel(r"r ($\mu$m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Radial profiles at snapshot times")
    fig.tight_layout()
    out_path = out_dir / "figE_radial_profiles.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_surface3d(r, t, free, out_dir: Path) -> Path:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    tickvals, ticklabels = _time_axis_ticks(t[-1])
    T, R = np.meshgrid(np.log10(t[1:]), r)
    Z = free[1:, :].T

    fig = plt.figure(figsize=(9.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, R, Z, cmap=SEQ_CMAP, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.6, label=r"$[Ab]_{free}$ (nM)")
    ax.set_xlabel("Time t (s) [log]")
    ax.set_ylabel(r"Radius r ($\mu$m)")
    ax.set_zlabel(r"$[Ab]_{free}$ (nM)")
    ax.set_title("3D surface: free antibody concentration in the Krogh cylinder")
    ax.set_xticks(np.log10(tickvals))
    ax.set_xticklabels(ticklabels)
    ax.view_init(elev=25, azim=-125)
    out_path = out_dir / "figF_surface3d.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_validation(params: KroghParams, r, t, free, bound, out_dir: Path) -> Path:
    ab_total_avg = volume_averaged(r, free + bound)
    ab_total_compartmental = params.Ab_plasma0 * compartmental_ab_ratio(params, t)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(t / 86400.0, ab_total_avg, lw=2.5, color="#2C6E8C",
            label="Spatial PDE (this model, radially averaged)")
    ax.plot(t / 86400.0, ab_total_compartmental, lw=2, ls="--", color="#B5541A",
            label="Compartmental (Thurber & Wittrup 2012, Eq. 7-8)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"Volume-averaged $[Ab]_{total}$ (nM)")
    ax.set_title("Validation: spatial model reduces to the compartmental limit")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "figG_validation.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = KroghParams()

    t_max = 10 * 86400.0
    t_eval = np.unique(np.concatenate([[0.0], np.logspace(0, np.log10(t_max), 200)]))
    r, t, free, bound, ag = solve_krogh_pde(params, nr=100, t_eval=t_eval, t_max=t_max)
    print(f"Solved: {len(t)} time points x {len(r)} radial points. "
          f"min/max Ab_free = [{free.min():.4g}, {free.max():.4g}] nM")

    paths = [plot_plasma_pk(params, t, OUT_DIR)]
    paths += plot_heatmaps(
        r, t,
        {
            "free": (free, r"$[Ab]_{free}$", "figB_Ab_free_heatmap.png"),
            "bound": (bound, r"$[Ab]_{bound}$", "figC_Ab_bound_heatmap.png"),
            "ag": (ag, r"$[Ag]$", "figD_Ag_heatmap.png"),
        },
        OUT_DIR,
    )
    paths.append(plot_radial_profiles(r, t, free, bound, ag, OUT_DIR))
    paths.append(plot_surface3d(r, t, free, OUT_DIR))
    paths.append(plot_validation(params, r, t, free, bound, OUT_DIR))

    print(f"\nSaved {len(paths)} figures to {OUT_DIR}:")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
