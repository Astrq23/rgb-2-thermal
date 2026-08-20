"""Checkpoint save/load built for Kaggle's 12-hour session ceiling.

A long pix2pix run will not finish in one session, so a checkpoint has to carry
everything needed to continue as if nothing happened: both networks, both
optimisers, both schedulers, the AMP scaler, the epoch counter and the RNG
state. Anything omitted shows up as a visible discontinuity in the loss curve
when the run resumes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def save_checkpoint(path: Path | str, **state: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["rng"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    # Write to a temporary file first: a session killed mid-write must not
    # destroy the only good checkpoint.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: Path | str, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rng(state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, ValueError):
            # Resuming on a different GPU count is fine; the RNG just restarts.
            pass


def latest_checkpoint(directory: Path | str) -> Optional[Path]:
    directory = Path(directory)
    candidate = directory / "last.pt"
    return candidate if candidate.exists() else None
