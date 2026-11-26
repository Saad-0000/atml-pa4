"""Deterministic seeding utilities for ATML PA4."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
