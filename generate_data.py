"""
Entry point: generate the synthetic dataset (Hydra-configured).

Usage:
    python generate_data.py
    python generate_data.py sampling.n_conditions=100000 sampling.n_times=8
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from data.synthetic.generate import build_dataset


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    X, Y, x_grid, t_points, condition_id = build_dataset(cfg.simulation, cfg.sampling)

    out_path = Path(cfg.dataset_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X, Y=Y, x_grid=x_grid, t_points=t_points, condition_id=condition_id,
        c0_range=cfg.simulation.c0_range, D_range=cfg.simulation.D_range,
        r_range=cfg.simulation.r_range, L=cfg.simulation.L,
    )
    print(f"Saved {X.shape[0]} samples ({cfg.sampling.n_conditions} conditions x "
          f"{cfg.sampling.n_times} times) to {out_path}")
    print(f"X shape={X.shape}  Y shape={Y.shape}")


if __name__ == "__main__":
    main()
