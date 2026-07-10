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

## Project structure

```
src/diffusion_degradation/
├── solver.py         # analytical + numerical physics solver
├── generate_data.py  # synthetic dataset generation
├── model.py          # neural network architecture
├── train.py          # training loop
└── evaluate.py       # evaluation and plots

notebooks/     # exploration notebook
outputs/       # generated dataset, trained model, metrics, figures
```

## Installation

```bash
pip install -r requirements.txt
```

Requires numpy, torch, and matplotlib.

## Usage

Run all commands from `src/diffusion_degradation/`:

```bash
cd src/diffusion_degradation

# 1. Check the solver (compares analytical vs numerical solution)
python3 solver.py

# 2. Generate a synthetic dataset
python3 generate_data.py --n_conditions 2000 --n_times 8 --out ../../outputs/dataset.npz

# 3. Train the surrogate model
python3 train.py --dataset ../../outputs/dataset.npz --epochs 200

# 4. Evaluate the trained model
python3 evaluate.py --dataset ../../outputs/dataset.npz
```

This produces a trained model checkpoint, training history, test metrics, and comparison plots in `outputs/`.

## Next steps

Future work aims to make the physics model more realistic (more variables, non-uniform conditions) and extend the surrogate to generalize across a wider range of scenarios, moving toward a general-purpose physical surrogate model.
