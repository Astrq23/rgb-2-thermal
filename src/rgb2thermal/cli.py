"""Console entry points declared in pyproject.toml.

These are thin conveniences (``r2t-inspect``, ``r2t-manifest``) for when the
package is pip-installed; the canonical interface is ``python scripts/*.py``,
which is what the notebooks use.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(script_name: str) -> int:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    module = __import__(script_name)
    return int(module.main())


def inspect_main() -> int:
    return _run("inspect_datasets")


def manifest_main() -> int:
    return _run("build_manifest")
