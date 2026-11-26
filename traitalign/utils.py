"""
Small, dependency-light helpers shared across the package.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python's random, NumPy, and Torch (CPU + all CUDA devices) so a
    single run of the whole pipeline is reproducible for a given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🌱 Global seed set to {seed}")
