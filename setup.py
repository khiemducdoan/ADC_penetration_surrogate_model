from setuptools import find_packages, setup

setup(
    name="adc-penetration-surrogate",
    version="0.1.0",
    description="Physical surrogate model for drug diffusion in tissue",
    packages=find_packages(include=[
        "data", "data.*",
        "models", "models.*",
        "training", "training.*",
        "evaluation", "evaluation.*",
        "utils", "utils.*",
    ]),
    install_requires=[
        "numpy>=1.24",
        "torch>=2.1",
        "matplotlib>=3.7",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "wandb>=0.16",
    ],
    python_requires=">=3.9",
)
