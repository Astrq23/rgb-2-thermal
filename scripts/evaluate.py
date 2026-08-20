#!/usr/bin/env python
"""Score a trained checkpoint, broken down by source dataset.

    python scripts/evaluate.py --checkpoint outputs/pix2pix_uav/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint .../best.pt --split test --no-unpaired-fid

Reports PSNR/SSIM/LPIPS per dataset plus a global FID, and -- when HIT-UAV is
mounted -- an unpaired FID against real high-altitude UAV thermal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
from torch.utils.data import DataLoader

from rgb2thermal.config import ModelConfig, load_config, load_dataset_specs
from rgb2thermal.data.dataset import PairedThermalDataset, ThermalReferenceDataset, trainable
from rgb2thermal.data.manifest import REFERENCE_SPLIT, load_manifest
from rgb2thermal.engine.evaluator import evaluate
from rgb2thermal.metrics import format_table
from rgb2thermal.models.build import build_generator
from rgb2thermal.utils.checkpoint import load_checkpoint
from rgb2thermal.utils.paths import default_output_dir, ensure_dir
from rgb2thermal.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/pix2pix_uav.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default=None, help="Override eval.split (default: test)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--no-unpaired-fid",
        action="store_true",
        help="Skip the HIT-UAV reference comparison (it is the slow part)",
    )
    parser.add_argument(
        "--max-reference",
        type=int,
        default=2000,
        help="Cap on reference images for unpaired FID (default: 2000)",
    )
    return parser.parse_args()


def model_config_from_checkpoint(checkpoint, fallback):
    """Prefer the architecture the checkpoint was trained with.

    Otherwise any run that used `--set model.ngf=...` fails to load here with an
    opaque state_dict shape error. The YAML still supplies everything else; only
    the architecture is taken from the checkpoint.
    """
    stored = (checkpoint.get("config") or {}).get("model")
    if not stored:
        return fallback
    try:
        merged = ModelConfig(**stored)
    except TypeError:
        return fallback
    if merged != fallback:
        print(f"[checkpoint] using the architecture stored in the checkpoint: {stored}")
    return merged


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.split:
        cfg.eval.split = args.split
    seed_everything(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = load_dataset_specs(cfg.data.dataset_specs)

    manifest_path = Path(args.manifest or cfg.data.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run scripts/build_manifest.py first.")
        return 1
    frame = load_manifest(manifest_path)

    # ---- generator --------------------------------------------------------
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    domain_index = checkpoint.get("domain_index") or {
        name: idx for idx, name in enumerate(sorted({s.name for s in specs}))
    }
    model_cfg = model_config_from_checkpoint(checkpoint, cfg.model)
    generator = build_generator(model_cfg, num_domains=len(domain_index)).to(device)
    generator.load_state_dict(checkpoint["generator"])
    print(f"Loaded generator from {args.checkpoint} (epoch {checkpoint.get('epoch')})")

    # ---- paired eval set --------------------------------------------------
    usable = trainable(frame)
    eval_frame = usable[usable["split"] == cfg.eval.split]
    if eval_frame.empty:
        print(f"No rows with split={cfg.eval.split!r} in the manifest.")
        return 1

    eval_set = PairedThermalDataset(eval_frame, cfg.data, specs, train=False, domain_index=domain_index)
    eval_loader = DataLoader(
        eval_set,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Evaluating {len(eval_set):,} pairs from split={cfg.eval.split}")

    # ---- unpaired reference ----------------------------------------------
    reference_loader = None
    if not args.no_unpaired_fid:
        reference_frame = frame[
            (frame["split"] == REFERENCE_SPLIT)
            & (frame["dataset"] == cfg.eval.fid_reference_dataset)
        ]
        if reference_frame.empty:
            print(
                f"[eval] reference dataset {cfg.eval.fid_reference_dataset!r} not in the "
                "manifest -- skipping unpaired FID"
            )
        else:
            if args.max_reference:
                reference_frame = reference_frame.head(args.max_reference)
            reference_set = ThermalReferenceDataset(reference_frame, cfg.data, specs)
            reference_loader = DataLoader(
                reference_set,
                batch_size=cfg.eval.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
            )
            print(f"Unpaired FID reference: {len(reference_set):,} real thermal images")

    # ---- run --------------------------------------------------------------
    out_dir = ensure_dir(Path(args.out) if args.out else default_output_dir() / cfg.name / "eval")
    results = evaluate(cfg, generator, eval_loader, device, reference_loader, out_dir)

    print("\n" + "=" * 72)
    print(f"PAIRED METRICS  (split={cfg.eval.split}, n={results['n_samples']:,})")
    print("=" * 72)
    print(format_table(results["paired"]))

    if "unpaired_fid" in results:
        print(
            f"\nUnpaired FID vs {cfg.eval.fid_reference_dataset}: "
            f"{results['unpaired_fid']:.3f}   (lower = more like real UAV thermal)"
        )
    if results.get("skipped"):
        print("\nSkipped metrics:")
        for name, reason in results["skipped"].items():
            print(f"  {name}: {reason}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'eval_samples.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
