"""Evaluation: per-dataset paired metrics plus unpaired FID against HIT-UAV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import Config
from ..metrics import ThermalMetrics, format_table, to_three_channel, to_uint8
from ..utils.paths import ensure_dir
from ..viz import save_comparison_grid
from .amp import autocast


@torch.no_grad()
def evaluate(
    cfg: Config,
    generator: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    reference_loader: Optional[DataLoader] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Score a trained generator.

    ``reference_loader`` supplies real thermal images with no RGB counterpart
    (HIT-UAV). When present, an extra unpaired FID is reported: the paired
    metrics say how close we are to one specific ground-truth frame, this says
    whether the output looks like real UAV thermal at all.
    """
    generator.eval()
    metrics = ThermalMetrics(metrics=cfg.eval.metrics, device=device)

    generated_for_fid: list[torch.Tensor] = []
    visual_batch: Optional[dict[str, Any]] = None
    seen = 0

    for batch in tqdm(loader, desc="evaluating", leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        real_thermal = batch["thermal"].to(device, non_blocking=True)
        domain = batch["domain"].to(device, non_blocking=True)

        with autocast(cfg.train.amp):
            fake_thermal = generator(rgb, domain)
        fake_thermal = fake_thermal.float()

        metrics.update(fake_thermal, real_thermal, batch["dataset"])

        if reference_loader is not None:
            generated_for_fid.append(fake_thermal.cpu())
        if visual_batch is None:
            visual_batch = {
                "rgb": rgb.cpu(),
                "thermal": real_thermal.cpu(),
                "fake": fake_thermal.cpu(),
                "dataset": list(batch["dataset"]),
            }

        seen += rgb.size(0)
        if cfg.eval.max_samples and seen >= cfg.eval.max_samples:
            break

    results: dict[str, Any] = {"paired": metrics.compute(), "n_samples": seen}
    if metrics.skipped:
        results["skipped"] = metrics.skipped

    if reference_loader is not None and generated_for_fid:
        score = _streaming_unpaired_fid(generated_for_fid, reference_loader, device)
        if score is not None:
            results["unpaired_fid"] = score

    if output_dir is not None and visual_batch is not None:
        output_dir = ensure_dir(output_dir)
        save_comparison_grid(
            visual_batch["rgb"],
            visual_batch["thermal"],
            visual_batch["fake"],
            output_dir / "eval_samples.png",
            max_items=min(cfg.eval.num_visuals, visual_batch["rgb"].shape[0]),
            titles=visual_batch["dataset"],
        )
        (output_dir / "eval_metrics.txt").write_text(
            format_table(results["paired"]), encoding="utf-8"
        )

    return results


@torch.no_grad()
def _streaming_unpaired_fid(
    generated: list[torch.Tensor],
    reference_loader: DataLoader,
    device: str,
) -> Optional[float]:
    """FID between our generated thermal and a real thermal reference set.

    Streams both sides through the metric instead of materialising two large
    tensors, which matters because Kaggle's RAM is shared with the DataLoader
    workers.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return None

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    for batch in tqdm(reference_loader, desc="unpaired FID (real)", leave=False):
        real = batch["thermal"].to(device)
        metric.update(to_uint8(to_three_channel(real)), real=True)

    for chunk in generated:
        metric.update(to_uint8(to_three_channel(chunk.to(device))), real=False)

    try:
        return float(metric.compute())
    except (RuntimeError, ValueError) as exc:
        print(f"[eval] unpaired FID skipped: {exc}")
        return None
