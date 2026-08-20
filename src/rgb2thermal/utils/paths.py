"""Filesystem helpers for locating datasets across Kaggle / local layouts."""

from __future__ import annotations

import glob as _glob
import os
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

#: Where Kaggle mounts attached datasets (read-only).
KAGGLE_INPUT = Path("/kaggle/input")
#: Kaggle's writable scratch directory (~20 GB).
KAGGLE_WORKING = Path("/kaggle/working")


def on_kaggle() -> bool:
    """True when running inside a Kaggle notebook session."""
    return KAGGLE_INPUT.exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_output_dir() -> Path:
    """Writable directory for checkpoints/outputs, Kaggle-aware."""
    return KAGGLE_WORKING / "outputs" if on_kaggle() else Path("outputs")


def expand_roots(patterns: Sequence[str]) -> list[Path]:
    """Expand a list of path patterns into existing directories.

    Patterns may contain shell wildcards, which is what makes the adapters
    tolerant of Kaggle mirrors that use a slightly different slug than expected
    (e.g. ``/kaggle/input/*dronevehicle*``). Results are de-duplicated while
    preserving the order the patterns were declared in, so the most specific
    pattern a config lists is also the first root searched.
    """
    seen: set[Path] = set()
    roots: list[Path] = []
    for pattern in patterns:
        expanded = _glob.glob(os.path.expanduser(str(pattern)))
        # A literal path with no wildcard still needs an existence check.
        candidates = expanded if expanded else [str(pattern)]
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(path)
    return roots


def iter_images(root: Path, exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    """Recursively collect image files under ``root``, sorted for determinism.

    ``os.walk`` is used rather than ``Path.rglob`` because these datasets hold
    tens of thousands of files and we want a single pass that also lets us skip
    directories we never care about.
    """
    exts = {e.lower() for e in exts}
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip archive/metadata dirs that would only add noise.
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            if Path(name).suffix.lower() in exts:
                found.append(Path(dirpath) / name)
    found.sort()
    return found


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
