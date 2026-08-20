"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch.

    ``deterministic`` is off by default: cuDNN's deterministic kernels cost a
    noticeable amount of throughput, which is expensive inside Kaggle's 12h
    session budget. Turn it on when reproducing a specific result matters more
    than speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
