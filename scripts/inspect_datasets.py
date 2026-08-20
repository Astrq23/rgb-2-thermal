#!/usr/bin/env python
"""Show what is actually mounted, and whether each adapter can read it.

**Run this first on Kaggle.** Community dataset mirrors sometimes reorganise the
upstream folder layout, and every downstream failure ("0 pairs", "manifest is
empty") traces back to here. The report has two halves:

1. a directory tree of ``/kaggle/input`` with image counts and example names;
2. a per-dataset probe: which root matched, how many pairs the adapter found,
   and a few resolved pairs to eyeball.

If a dataset reports 0 pairs, compare its real tree against
``configs/datasets/<name>.yaml`` and fix the globs there -- no Python change
should be necessary.

    python scripts/inspect_datasets.py
    python scripts/inspect_datasets.py --root /kaggle/input --depth 4
    python scripts/inspect_datasets.py --dataset dronevehicle
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401

from rgb2thermal.config import load_dataset_specs
from rgb2thermal.data.adapters.registry import build_adapter
from rgb2thermal.utils.paths import IMAGE_EXTS, on_kaggle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        default=None,
        help="Directory to tree-print. Defaults to /kaggle/input on Kaggle, ./data otherwise.",
    )
    parser.add_argument("--depth", type=int, default=3, help="Max tree depth (default: 3)")
    parser.add_argument(
        "--dataset-specs",
        nargs="+",
        default=["configs/datasets/*.yaml"],
        help="Glob(s) for dataset spec YAML files",
    )
    parser.add_argument("--dataset", default=None, help="Probe only this dataset by name")
    parser.add_argument(
        "--examples", type=int, default=3, help="Example pairs to print per dataset"
    )
    parser.add_argument("--no-tree", action="store_true", help="Skip the directory tree")
    return parser.parse_args()


def print_tree(root: Path, max_depth: int) -> None:
    """Directory tree annotated with image counts and one example filename."""
    if not root.exists():
        print(f"  (missing: {root})")
        return

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)

        if depth > max_depth:
            dirnames[:] = []
            continue

        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        images = [f for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS]

        indent = "  " * depth
        label = root.name if depth == 0 else current.name
        summary = ""
        if images:
            example = sorted(images)[0]
            summary = f"  [{len(images):,} images, e.g. {example}]"
        elif filenames:
            summary = f"  [{len(filenames):,} non-image files]"

        print(f"{indent}{label}/{summary}")

        # A leaf full of images: do not descend further, it is just noise.
        if depth == max_depth and dirnames:
            print(f"{indent}  ... ({len(dirnames)} more subdirectories)")


def probe(spec, examples: int) -> None:
    adapter = build_adapter(spec)
    roots = adapter.roots()

    print(f"\n--- {spec.name} " + "-" * max(0, 60 - len(spec.name)))
    print(f"  adapter      : {spec.adapter}")
    print(f"  enabled      : {spec.enabled}   paired: {spec.paired}")
    print(f"  search_roots : {spec.search_roots}")

    if not roots:
        print("  STATUS       : NOT MOUNTED -- attach the Kaggle dataset, or fix search_roots")
        return

    print(f"  matched root : {roots[0]}")
    records = adapter.discover()

    if not records:
        print("  STATUS       : root found but 0 pairs matched  <-- FIX configs/datasets/*.yaml")
        print("                 Compare the tree above with the expected layout.")
        return

    kind = "pairs" if spec.paired else "thermal-only images"
    print(f"  STATUS       : OK -- {len(records):,} {kind}")
    print(f"  groups       : {len({r.group_key for r in records}):,}")
    hints = {r.split_hint for r in records}
    print(f"  split hints  : {sorted(hints)}")

    step = max(1, len(records) // max(1, examples))
    for record in records[::step][:examples]:
        print(f"    RGB     : {record.rgb_path or '(none)'}")
        print(f"    THERMAL : {record.thermal_path}")
        print(f"    group={record.group_key}  split_hint={record.split_hint}")
        print()


def main() -> int:
    args = parse_args()

    root = Path(args.root) if args.root else (Path("/kaggle/input") if on_kaggle() else Path("data"))

    if not args.no_tree:
        print("=" * 72)
        print(f"DIRECTORY TREE: {root}  (depth {args.depth})")
        print("=" * 72)
        print_tree(root, args.depth)

    print()
    print("=" * 72)
    print("ADAPTER PROBE")
    print("=" * 72)

    specs = load_dataset_specs(args.dataset_specs)
    if args.dataset:
        specs = [s for s in specs if s.name == args.dataset]
        if not specs:
            print(f"No dataset spec named {args.dataset!r}")
            return 1

    for spec in specs:
        probe(spec, args.examples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
