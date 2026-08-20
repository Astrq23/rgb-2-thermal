"""Guards that what runs locally is what actually reaches GitHub.

These exist because of a real failure: `.gitignore` carried an unanchored
`data/` rule intended for the dataset directory at the project root, and it also
matched `src/rgb2thermal/data/`. Ten files -- the adapters, the manifest builder
and the whole normalisation layer -- were silently excluded from the repository
and pushed missing. Every test still passed, because tests read the working
tree, not git.

The Kaggle notebooks clone from GitHub, so an untracked source file is invisible
here and fatal there. That gap is exactly what these tests close.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def is_git_repo() -> bool:
    try:
        git("rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_git = pytest.mark.skipif(not is_git_repo(), reason="not a git checkout")


def tracked_files() -> set[str]:
    return {line for line in git("ls-files").splitlines() if line}


@requires_git
def test_every_source_file_is_tracked():
    """A source file that git ignores does not exist as far as Kaggle knows."""
    tracked = tracked_files()
    on_disk = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, (
        "Source files exist on disk but are not tracked by git:\n  "
        + "\n  ".join(missing)
        + "\nCheck .gitignore for an unanchored rule (use /data/, not data/)."
    )


@requires_git
def test_configs_scripts_and_notebooks_are_tracked():
    tracked = tracked_files()
    missing: list[str] = []
    for folder, pattern in (("configs", "*.yaml"), ("scripts", "*.py"), ("notebooks", "*.ipynb")):
        for path in (REPO_ROOT / folder).rglob(pattern):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative not in tracked:
                missing.append(relative)
    assert not missing, f"Untracked files the notebooks depend on: {missing}"


@requires_git
def test_gitignore_rules_for_artifacts_are_root_anchored():
    """`data/` must be `/data/`, or it swallows any nested folder of that name.

    Checked as a rule rather than only by outcome, so the mistake is caught when
    the rule is written instead of after a package quietly disappears.
    """
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    risky = {"data", "outputs", "checkpoints", "runs", "subset", "predictions"}

    offenders = [
        line
        for line in (raw.strip() for raw in lines)
        if line and not line.startswith("#") and line.rstrip("/").lstrip("/") in risky
        and not line.startswith("/")
    ]
    assert not offenders, (
        f"Unanchored .gitignore rule(s) that can match nested packages: {offenders}. "
        "Prefix them with '/' to limit them to the repository root."
    )


@requires_git
def test_no_dataset_or_checkpoint_files_committed():
    """The inverse guard: licensed imagery and weights must stay out of git."""
    heavy = [
        name
        for name in tracked_files()
        if name.lower().endswith((".pt", ".pth", ".ckpt", ".jpg", ".jpeg", ".npy", ".npz"))
    ]
    assert not heavy, f"Data or weights committed to the repository: {heavy}"


@requires_git
def test_package_is_importable_from_tracked_files_only():
    """Every module the package imports must itself be tracked.

    Walking the real package directory catches a subpackage that was created but
    never added -- the precise shape of the original failure.
    """
    tracked = tracked_files()
    package = REPO_ROOT / "src" / "rgb2thermal"
    for init in package.rglob("__init__.py"):
        relative = init.relative_to(REPO_ROOT).as_posix()
        assert relative in tracked, f"Subpackage not tracked: {relative}"
