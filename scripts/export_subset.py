#!/usr/bin/env python
"""Export a small, pre-processed subset that replaces the huge source datasets.

**The problem this solves.** The four source datasets are tens of gigabytes,
almost all of it resolution we throw away: DroneVehicle ships 840x712 frames
that training immediately crops and resizes to 256x256, and FLIR's RGB is larger
still. Every Kaggle session pays the attach cost for that full resolution, and
then decodes it again on every epoch.

Run this **once**, with the big datasets attached. It applies each dataset's
geometric fixes (border crop, FOV crop), resizes to just above the training
resolution, and writes a flat folder that is typically 20-50x smaller. Save the
notebook output as a Kaggle Dataset and attach *that* from then on: sessions
start in seconds and the DataLoader stops being the bottleneck.

    # once, with the originals attached
    python scripts/export_subset.py --manifest /kaggle/working/manifest.csv \\
        --out /kaggle/working/subset --per-dataset 8000

    # afterwards, with only the exported dataset attached
    python scripts/build_manifest.py --dataset-specs configs/datasets/subset.yaml

Metadata is preserved in the path (`<dataset>/rgb/<group>~<seq>.jpg`), so the
subset keeps its per-dataset sampling weights, its per-dataset metric breakdown
and its leakage-free scene grouping. Thermal-only sources are written to
`reference/<dataset>/thermal/`. See `configs/datasets/subset.yaml` and
`configs/datasets/subset_reference.yaml`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from rgb2thermal.config import load_config, load_dataset_specs
from rgb2thermal.data.manifest import load_manifest
from rgb2thermal.data.normalize import PreprocessSpec, load_rgb, load_thermal
from rgb2thermal.utils.paths import ensure_dir
from PIL import Image
from tqdm.auto import tqdm

#: Kept above the training crop size so random-crop jitter still has room.
DEFAULT_EXPORT_SIZE = 288
UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/pix2pix_uav.yaml")
    parser.add_argument("--manifest", default=None, help="Manifest built from the originals")
    parser.add_argument("--out", default="subset", help="Output directory")
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_EXPORT_SIZE,
        help=f"Short side of exported images (default: {DEFAULT_EXPORT_SIZE})",
    )
    parser.add_argument(
        "--per-dataset",
        type=int,
        default=8000,
        help="Max pairs to keep per source dataset (0 = all). Default: 8000",
    )
    parser.add_argument(
        "--reference-max",
        type=int,
        default=2000,
        help="Max thermal-only reference images to keep (HIT-UAV). Default: 2000",
    )
    parser.add_argument("--quality", type=int, default=94, help="JPEG quality (default: 94)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe(text: str) -> str:
    return UNSAFE.sub("_", str(text)).strip("_") or "x"


def resize_short_side(image: Image.Image, target: int) -> Image.Image:
    """Downscale so the shorter side is ``target``; never upscale.

    Upscaling would inflate the export without adding information -- a 96px
    source stays 96px and simply contributes a lower-resolution sample.
    """
    width, height = image.size
    if min(width, height) <= target:
        return image
    scale = target / min(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)


def select(subset, limit: int):
    """Take an evenly spaced slice, not the first N.

    The first N rows of a video dataset are all the same scene; striding keeps
    the scene diversity that makes the subset worth training on.
    """
    if not limit or len(subset) <= limit:
        return subset
    step = max(1, len(subset) // limit)
    return subset.iloc[::step].head(limit)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    specs = load_dataset_specs(cfg.data.dataset_specs)
    preprocess = {spec.name: PreprocessSpec.from_dict(spec.preprocess) for spec in specs}
    viewpoints = {spec.name: spec.viewpoint for spec in specs}

    manifest_path = Path(args.manifest or cfg.data.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run scripts/build_manifest.py first.")
        return 1
    frame = load_manifest(manifest_path)

    out_root = Path(args.out)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    written = Counter()
    skipped = Counter()

    for dataset_name, subset in frame.groupby("dataset"):
        is_reference = (subset["rgb_path"] == "").all()
        limit = args.reference_max if is_reference else args.per_dataset
        chosen = select(subset, limit)

        kind = "thermal-only" if is_reference else "pairs"
        print(f"\n{dataset_name}: {len(subset):,} {kind} -> exporting {len(chosen):,}")

        spec = preprocess.get(dataset_name, PreprocessSpec())
        # Thermal-only sources live under reference/ so the subset config can
        # keep `paired: true` (a thermal image with no RGB partner would
        # otherwise be silently dropped, taking the unpaired-FID set with it).
        base = (out_root / "reference" / dataset_name) if is_reference else (out_root / dataset_name)
        rgb_dir = ensure_dir(base / "rgb")
        thermal_dir = ensure_dir(base / "thermal")

        for sequence, (_, row) in enumerate(
            tqdm(chosen.iterrows(), total=len(chosen), desc=dataset_name, leave=False)
        ):
            # The scene group travels in the filename so the exported copy can
            # still be split without leaking near-duplicate frames.
            stem = f"{safe(row['group_key'])}~{sequence:06d}"
            try:
                thermal = resize_short_side(
                    load_thermal(row["thermal_path"], spec.thermal), args.size
                )
                if is_reference:
                    thermal.save(thermal_dir / f"{stem}.jpg", quality=args.quality)
                else:
                    rgb = resize_short_side(load_rgb(row["rgb_path"], spec.rgb), args.size)
                    # Thermal is resampled onto the RGB grid here rather than at
                    # train time, so the expensive alignment step happens once.
                    if thermal.size != rgb.size:
                        thermal = thermal.resize(rgb.size, Image.BICUBIC)
                    rgb.save(rgb_dir / f"{stem}.jpg", quality=args.quality)
                    thermal.save(thermal_dir / f"{stem}.jpg", quality=args.quality)
                written[dataset_name] += 1
            except (OSError, ValueError) as error:
                # A handful of corrupt JPEGs is normal in datasets this size;
                # losing one sample is not worth aborting a 20-minute export.
                skipped[dataset_name] += 1
                if skipped[dataset_name] <= 3:
                    print(f"  skipped {row['thermal_path']}: {error}")

        if is_reference:
            rgb_dir.rmdir()

    info = {
        "source_manifest": str(manifest_path),
        "export_size": args.size,
        "per_dataset_limit": args.per_dataset,
        "written": dict(written),
        "skipped": dict(skipped),
        "viewpoints": {name: viewpoints.get(name, "unknown") for name in written},
    }
    (out_root / "EXPORT_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    total_bytes = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    print("\n" + "=" * 68)
    print(f"Exported {sum(written.values()):,} images to {out_root}")
    print(f"Total size: {total_bytes / 1e6:,.0f} MB")
    for name, count in sorted(written.items()):
        print(f"  {name:<16} {count:>8,}")
    if skipped:
        print(f"Skipped (unreadable): {dict(skipped)}")

    print("\nNext:")
    print("  1. Save Version -> Save & Run All, then publish the output as a Kaggle Dataset.")
    print("  2. In future sessions attach ONLY that dataset (not the originals).")
    print("  3. Rebuild the manifest against it:")
    print("       python scripts/build_manifest.py \\")
    print("           --dataset-specs 'configs/datasets/subset*.yaml' \\")
    print("           --out /kaggle/working/manifest.csv")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
