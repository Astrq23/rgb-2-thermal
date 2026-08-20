"""Lightweight CSV metric logging.

No TensorBoard or W&B: a Kaggle session is ephemeral and its outputs are
downloaded as files, so a CSV that pandas can read straight back is both the
simplest and the most durable option.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Optional


class CSVLogger:
    """Append-only metric log with a header inferred from the first row."""

    def __init__(self, path: Path | str, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: Optional[list[str]] = None

        if resume and self.path.exists():
            with open(self.path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header:
                    self.fieldnames = header
        elif self.path.exists():
            self.path.unlink()

    def log(self, **row: Any) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row)
            with open(self.path, "w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


class Stopwatch:
    """Elapsed-time helper used to warn before Kaggle's session limit."""

    def __init__(self) -> None:
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def format(self) -> str:
        seconds = int(self.elapsed)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
