"""Mixed-precision shims.

torch 2.4 moved ``torch.cuda.amp.GradScaler``/``autocast`` to ``torch.amp`` and
started warning on the old spelling. Kaggle's image version drifts, so both
paths are supported rather than pinning the notebook to one torch release.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def make_scaler(enabled: bool) -> Any:
    """GradScaler that is a no-op when AMP is off or there is no CUDA device."""
    enabled = enabled and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(enabled: bool) -> Any:
    if not (enabled and torch.cuda.is_available()):
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)
