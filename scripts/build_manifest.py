#!/usr/bin/env python
"""Scan every configured dataset and write the unified manifest.

    python scripts/build_manifest.py
    python scripts/build_manifest.py --out /kaggle/working/manifest.csv
    python scripts/build_manifest.py --dataset-specs configs/datasets/dronevehicle.yaml

Datasets that are not mounted are reported and skipped, so this works with any
subset of the four attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from rgb2thermal.config import load_config, load_dataset_specs
from rgb2thermal.data.manifest import build_manifest, save_manifest, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/pix2pix_uav.yaml", help="Training config (for split settings)")
    parser.add_argument("--dataset-specs", nargs="+", default=None, help="Override the dataset spec globs")
    parser.add_argument("--out", default=None, help="Where to write manifest.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    specs = load_dataset_specs(args.dataset_specs or cfg.data.dataset_specs)

    print(f"Loaded {len(specs)} dataset spec(s): {', '.join(s.name for s in specs)}\n")

    frame = build_manifest(
        specs,
        ratios=tuple(cfg.data.split_ratios),  # type: ignore[arg-type]
        seed=cfg.data.split_seed,
        respect_split_hint=cfg.data.respect_split_hint,
        verbose=not args.quiet,
    )

    print()
    print("=" * 72)
    print(summarize(frame))
    print("=" * 72)

    if frame.empty:
        print("\nNothing to write. Run `python scripts/inspect_datasets.py` next.")
        return 1

    out = Path(args.out) if args.out else Path(cfg.data.manifest)
    save_manifest(frame, out)
    print(f"\nWrote {len(frame):,} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
