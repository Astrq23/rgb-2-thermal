"""Keep the documentation honest.

Docs rot in ways that are invisible until someone follows them and gets stuck:
a script gets renamed, a table of contents drifts from the headings, a relative
link points at a file that moved. Each of those wastes the reader's time far
from where the mistake was made, so they are cheap to assert here.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = sorted(
    path
    for path in REPO_ROOT.rglob("*.md")
    if not any(part in {".git", "node_modules", ".venv"} for part in path.parts)
)

#: [text](target) -- ignores images and bare URLs.
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")
SCRIPT = re.compile(r"scripts/([A-Za-z0-9_]+\.py)")
CONFIG = re.compile(r"configs/(?:datasets/)?([A-Za-z0-9_*]+\.yaml)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


def slugify(heading: str) -> str:
    """Approximate GitHub's anchor generation.

    Lowercase, strip inline markup and punctuation, collapse spaces to hyphens.
    Unicode letters (Vietnamese diacritics included) are preserved, which is
    what GitHub does.
    """
    text = re.sub(r"`([^`]*)`", r"\1", heading)          # inline code
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"[*_]", "", text)                     # emphasis
    text = text.lower().strip()

    kept = [
        char
        for char in text
        if char.isalnum() or char in " -" or unicodedata.category(char).startswith("M")
    ]
    # Consecutive hyphens are NOT collapsed: GitHub drops a character such as an
    # em dash but keeps the spaces that surrounded it, so "dataset - buoc"
    # becomes "dataset--buoc". Collapsing here would reject correct anchors.
    return "".join(kept).replace(" ", "-")


def test_markdown_files_exist():
    assert MARKDOWN, "no markdown documentation found"


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_relative_links_resolve(path: Path):
    """A link to a moved or renamed file is a dead end for the reader."""
    broken: list[str] = []
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        target = target.split(" ")[0]
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = (path.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, f"{path.name} links to missing file(s): {broken}"


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_referenced_scripts_exist(path: Path):
    referenced = set(SCRIPT.findall(path.read_text(encoding="utf-8")))
    missing = sorted(n for n in referenced if not (REPO_ROOT / "scripts" / n).exists())
    assert not missing, f"{path.name} documents missing script(s): {missing}"


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_referenced_configs_exist(path: Path):
    """Guards against documenting a config that was renamed or never added."""
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []
    for name in set(CONFIG.findall(text)):
        if "*" in name:  # a glob; existence of the directory is enough
            continue
        if not (REPO_ROOT / "configs" / name).exists() and not (
            REPO_ROOT / "configs" / "datasets" / name
        ).exists():
            missing.append(name)
    assert not missing, f"{path.name} documents missing config(s): {missing}"


def test_usage_guide_table_of_contents_matches_its_headings():
    """A drifting table of contents is the classic long-document failure."""
    path = REPO_ROOT / "docs" / "USAGE.md"
    text = path.read_text(encoding="utf-8")
    anchors = {slugify(title) for _, title in HEADING.findall(text)}

    broken = [
        target
        for target in LINK.findall(text)
        if target.startswith("#") and target[1:] not in anchors
    ]
    assert not broken, (
        f"USAGE.md has anchor link(s) with no matching heading: {broken}\n"
        f"Available anchors: {sorted(anchors)}"
    )


def test_usage_guide_covers_the_two_mandatory_steps():
    """Dataset probing and the alignment check are the steps that cannot be
    skipped, because neither failure reports itself as an error."""
    text = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
    assert "inspect_datasets.py" in text
    assert "check_alignment.py" in text
    assert "max_hours" in text
