#!/usr/bin/env python
"""Render RGB/thermal overlays so misregistration can be caught by eye.

Run this **before** the first real training job. Metrics will not warn you about
a misaligned dataset -- the model just learns to blur, and the loss curve looks
perfectly healthy while it does. FLIR ADAS in particular pairs two cameras with
different fields of view; its `preprocess.rgb.fov_crop` is an estimate that
needs tuning against these overlays.

    python scripts/check_alignment.py
    python scripts/check_alignment.py --dataset flir_v2 --n 6
    python scripts/check_alignment.py --dataset flir_v2 --fov-crop 0.55

Read the 4th panel: thermal edges are drawn in red over the RGB image. If they
sit on the RGB structures, the pair is aligned. If they float, adjust
`fov_crop` in configs/datasets/<name>.yaml -- or set `enabled: false`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from rgb2thermal.config import load_config, load_dataset_specs
from rgb2thermal.data.manifest import load_manifest
from rgb2thermal.data.normalize import PreprocessSpec, load_rgb, load_thermal
from rgb2thermal.utils.paths import default_output_dir, ensure_dir
from rgb2thermal.viz import save_alignment_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/pix2pix_uav.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset", default=None, help="Only this dataset (default: all)")
    parser.add_argument("--n", type=int, default=4, help="Overlays per dataset")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument(
        "--fov-crop",
        type=float,
        default=None,
        help="Override rgb.fov_crop for this run, to try a value before editing the YAML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    specs = load_dataset_specs(cfg.data.dataset_specs)
    preprocess = {spec.name: PreprocessSpec.from_dict(spec.preprocess) for spec in specs}

    manifest_path = Path(args.manifest or cfg.data.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run scripts/build_manifest.py first.")
        return 1

    frame = load_manifest(manifest_path)
    frame = frame[frame["rgb_path"] != ""]
    if args.dataset:
        frame = frame[frame["dataset"] == args.dataset]
    if frame.empty:
        print("No paired rows to check.")
        return 1

    out_dir = ensure_dir(Path(args.out) if args.out else default_output_dir() / "alignment")

    for dataset_name, subset in frame.groupby("dataset"):
        spec = preprocess.get(dataset_name, PreprocessSpec())
        if args.fov_crop is not None:
            spec.rgb.fov_crop = args.fov_crop

        # Spread the samples across the whole dataset rather than taking the
        # first N, which would all come from one scene.
        step = max(1, len(subset) // max(1, args.n))
        chosen = subset.iloc[::step].head(args.n)

        print(f"\n{dataset_name}: {len(subset):,} pairs, rendering {len(chosen)} overlay(s)")
        if args.fov_crop is not None:
            print(f"  using rgb.fov_crop={args.fov_crop}")

        for index, (_, row) in enumerate(chosen.iterrows()):
            rgb = load_rgb(row["rgb_path"], spec.rgb)
            thermal = load_thermal(row["thermal_path"], spec.thermal)
            path = save_alignment_overlay(
                rgb,
                thermal,
                out_dir / f"{dataset_name}_{index:02d}.png",
                title=f"{dataset_name}  |  {Path(row['rgb_path']).name}  |  rgb {rgb.size}  thermal {thermal.size}",
            )
            print(f"  {path}")

    print(f"\nOverlays written to {out_dir}")
    print("Check panel 4: thermal edges (red) must sit on the RGB structures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
