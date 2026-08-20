"""Make ``src/`` importable when the package has not been pip-installed.

Kaggle notebooks sometimes run a cell before ``pip install -e .`` has finished,
or after a kernel restart that dropped the editable install. Importing this
module first makes every script work either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
