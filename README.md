# Physical Surrogate Model for Drug Penetration in Tissue

This project explores how to build a fast neural network **surrogate model** that learns to reproduce the behavior of a physics-based simulation, as a step toward studying how a drug spreads and penetrates through tissue.

## Why this matters

Simulating how a drug diffuses through tissue with classic physics equations (partial differential equations, PDEs) is accurate but slow, and can only be solved for one set of conditions at a time. A surrogate model is a neural network trained on many simulation results so that, once trained, it can predict new outcomes almost instantly, for new conditions it has not seen before.

## What this repo currently does

The current code covers a simplified starting problem: a 1D diffusion-decay process, where a substance enters from one side, spreads through space, and is gradually lost over time (for example through absorption or decay). This is a reduced version of a more complex, real-world case, used here to validate the full pipeline before scaling up.

The pipeline has four steps:

1. **Physics solver** — computes the concentration profile over space and time, both with an exact analytical formula and with a numerical method, and cross-checks the two against each other.
2. **Data generation** — samples many random physical conditions and computes the resulting profiles, producing a synthetic dataset.
3. **Surrogate model training** — trains a small neural network to predict the profile directly from the physical conditions, without solving the physics equations.
4. **Evaluation** — compares the surrogate's predictions against the true simulation results and reports accuracy metrics and plots.

Configuration is managed with [Hydra](https://hydra.cc/): every parameter (physics ranges, sampling size, network architecture, training schedule) lives in a YAML file under `configs/` and can be overridden from the command line without touching code. Training runs are logged to [Weights & Biases](https://wandb.ai/).

## Beyond the 1D case: the Thurber/Krogh-cylinder model

`data/synthetic/krogh_solver.py` solves the fuller, nonlinear physics this 1D problem was a first approximation to: the Krogh-cylinder model of Thurber, Zajic & Wittrup (2007) / Thurber & Wittrup (2012), tracking free antibody, antibody-antigen complex, and free antigen as three coupled fields in radial (cylindrical) coordinates, with a permeability-limited (Robin) boundary condition driven by a time-varying plasma PK curve. It has no closed-form solution, so it's solved numerically by the method of lines (conservative finite-volume discretization in `r`, stiff implicit time integration in `t`), validated both by grid-refinement and by comparison to Thurber & Wittrup's closed-form compartmental (0D) reduction. See [`docs/THURBER_KROGH_PDE_MATH.md`](docs/THURBER_KROGH_PDE_MATH.md) for the full derivation.

```bash
python3 data/synthetic/krogh_solver.py          # solve + print convergence/validation self-checks
python3 -m data.synthetic.krogh_visualize        # generate figures into outputs/figures/krogh/
```

## Project structure

```
configs/
├── config.yaml               # top-level config (paths, wandb, hydra settings)
├── simulation/                # physics domain parameters
├── sampling/                  # dataset sampling parameters
└── training/                  # model + optimization parameters

data/synthetic/
├── solver.py                  # 1D analytical + numerical physics solver
├── krogh_solver.py             # Thurber/Krogh-cylinder spatial PDE solver (numerical, method of lines)
├── krogh_visualize.py          # figure generation for the Krogh-cylinder solver
└── generate.py                 # synthetic dataset generation logic (1D model)

models/
├── mlp.py                     # neural network architecture
├── losses.py                  # loss functions
└── metrics.py                 # MAE / RMSE / R2

training/
├── trainer.py                 # training loop (+ wandb logging)
├── callbacks.py                # early stopping
└── utils.py                    # data split, normalization, dataloaders

evaluation/
├── evaluator.py                # reload a checkpoint and score it
└── visualizer.py                # true-vs-predicted plots

utils/            # general helpers (e.g. reproducibility seeding)
notebooks/        # exploration notebook
outputs/          # generated dataset, trained model, metrics, figures

generate_data.py  # entry point: build the synthetic dataset
train.py          # entry point: train the surrogate
evaluate.py       # entry point: evaluate a trained surrogate
setup.py
```

## Installation

```bash
pip install -r requirements.txt
# or, to install the project as a package:
pip install -e .
```

Requires numpy, torch, matplotlib, hydra-core, omegaconf, and wandb (run `wandb login` once if you haven't already).

## Usage

All commands run from the repository root. Any config value can be overridden on the command line (Hydra syntax: `group.key=value`).

```bash
# 1. Check the solver (compares analytical vs numerical solution)
python3 data/synthetic/solver.py

# 2. Generate a synthetic dataset
python3 generate_data.py sampling.n_conditions=2000 sampling.n_times=8

# 3. Train the surrogate model (logs to Weights & Biases by default)
python3 train.py training.epochs=200
python3 train.py wandb.enabled=false   # to disable logging

# 4. Evaluate the trained model
python3 evaluate.py
```

This produces a trained model checkpoint, training history, test metrics, and comparison plots in `outputs/`.

## Next steps

Future work aims to make the physics model more realistic (more variables, non-uniform conditions) and extend the surrogate to generalize across a wider range of scenarios, moving toward a general-purpose physical surrogate model.
