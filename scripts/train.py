#!/usr/bin/env python
"""Train the RGB -> thermal generator.

    python scripts/train.py --config configs/smoke.yaml          # 2-minute sanity run
    python scripts/train.py --config configs/pix2pix_uav.yaml    # the real thing
    python scripts/train.py --config configs/pix2pix_uav.yaml --resume auto
    python scripts/train.py --config configs/pix2pix_uav.yaml --set train.epochs=60 data.num_workers=4

Builds the manifest automatically if it is missing. ``--resume auto`` picks up
``last.pt`` from the output directory, which is how a run spans more than one
Kaggle session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from rgb2thermal.config import load_config, load_dataset_specs
from rgb2thermal.data.dataset import build_dataloaders
from rgb2thermal.data.manifest import build_manifest, load_manifest, save_manifest, summarize
from rgb2thermal.engine.trainer import Trainer
from rgb2thermal.utils.paths import default_output_dir
from rgb2thermal.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/pix2pix_uav.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides, e.g. train.epochs=5 data.num_workers=4",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path, or 'auto' to continue from the run's last.pt",
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    seed_everything(cfg.train.seed)

    specs = load_dataset_specs(cfg.data.dataset_specs)

    # ---- manifest ---------------------------------------------------------
    manifest_path = Path(args.manifest or cfg.data.manifest)
    if args.rebuild_manifest or not manifest_path.exists():
        print(f"Building manifest -> {manifest_path}")
        frame = build_manifest(
            specs,
            ratios=tuple(cfg.data.split_ratios),  # type: ignore[arg-type]
            seed=cfg.data.split_seed,
            respect_split_hint=cfg.data.respect_split_hint,
        )
        if frame.empty:
            print("\nManifest is empty. Run `python scripts/inspect_datasets.py` to see why.")
            return 1
        save_manifest(frame, manifest_path)
    else:
        print(f"Using existing manifest {manifest_path} (--rebuild-manifest to refresh)")
        frame = load_manifest(manifest_path)

    print()
    print(summarize(frame))
    print()

    # ---- data -------------------------------------------------------------
    train_loader, val_loader, domain_index = build_dataloaders(
        frame, cfg.data, specs, batch_size=cfg.train.batch_size, seed=cfg.train.seed
    )
    print(
        f"train batches/epoch: {len(train_loader):,}   "
        f"val batches: {len(val_loader) if val_loader else 0:,}   "
        f"domains: {domain_index}"
    )

    # ---- train ------------------------------------------------------------
    trainer = Trainer(cfg, train_loader, val_loader, domain_index=domain_index)

    resume_path = args.resume or cfg.train.resume
    if resume_path == "auto":
        candidate = trainer.checkpoint_dir / "last.pt"
        resume_path = str(candidate) if candidate.exists() else ""
        if not resume_path:
            print("[train] --resume auto: no last.pt yet, starting fresh")
    if resume_path:
        trainer.resume(resume_path)

    summary = trainer.fit()

    output_dir = Path(summary.get("output_dir", default_output_dir()))
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nCheckpoints : {output_dir / 'checkpoints'}")
    print(f"Samples     : {output_dir / 'samples'}")
    print(f"Metrics CSV : {output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
