"""Evaluation metrics, reported per source dataset.

A single averaged score would hide the thing we most want to know: whether the
UAV domain actually improved, or whether a gain came from the easier
ground-level FLIR frames. Every paired metric is therefore accumulated per
dataset as well as overall.

PSNR/SSIM measure fidelity to *this* ground-truth frame. LPIPS and FID measure
perceptual and distributional realism, which is closer to what matters for
synthetic thermal: a plausible heat signature in slightly the wrong place is
more useful than a blurry one in exactly the right place.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, Sequence

import torch

PAIRED_METRICS = ("psnr", "ssim", "lpips")


def to_three_channel(thermal: torch.Tensor) -> torch.Tensor:
    """Single-channel thermal -> 3 channels, for ImageNet-pretrained metrics."""
    return thermal.repeat(1, 3, 1, 1) if thermal.shape[1] == 1 else thermal


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """``[-1, 1]`` float -> ``[0, 255]`` uint8, the layout FID expects."""
    return (((images.clamp(-1, 1) + 1) / 2) * 255).round().to(torch.uint8)


class ThermalMetrics:
    """Accumulates paired metrics per dataset plus a global FID.

    Metrics that cannot be constructed (torchmetrics missing, or no internet to
    fetch LPIPS/Inception weights) are skipped with a warning rather than
    aborting the evaluation -- partial numbers beat no numbers.
    """

    def __init__(
        self,
        metrics: Sequence[str] = PAIRED_METRICS + ("fid",),
        device: str = "cpu",
        fid_feature: int = 2048,
    ):
        self.device = device
        self.requested = [m.lower() for m in metrics]
        self.sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: dict[str, int] = defaultdict(int)
        self.skipped: dict[str, str] = {}

        self._psnr = None
        self._ssim = None
        self._lpips = None
        self.fid = None

        try:
            from torchmetrics.functional.image import (
                peak_signal_noise_ratio,
                structural_similarity_index_measure,
            )

            self._psnr = peak_signal_noise_ratio
            self._ssim = structural_similarity_index_measure
        except ImportError as exc:
            self.skipped["psnr/ssim"] = f"torchmetrics unavailable ({exc})"

        if "lpips" in self.requested:
            try:
                from torchmetrics.image.lpip import (
                    LearnedPerceptualImagePatchSimilarity,
                )

                self._lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=False
                ).to(device)
            except Exception as exc:  # noqa: BLE001 - weight download can fail offline
                self.skipped["lpips"] = f"{type(exc).__name__}: {exc}"

        if "fid" in self.requested:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance

                self.fid = FrechetInceptionDistance(feature=fid_feature, normalize=False).to(device)
            except Exception as exc:  # noqa: BLE001
                self.skipped["fid"] = f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------------- updates

    @torch.no_grad()
    def update(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        datasets: Optional[Sequence[str]] = None,
    ) -> None:
        """Accumulate one batch. ``fake``/``real`` are ``(N, 1, H, W)`` in [-1, 1]."""
        fake = fake.clamp(-1, 1).to(self.device)
        real = real.clamp(-1, 1).to(self.device)
        datasets = list(datasets) if datasets is not None else ["all"] * fake.shape[0]

        for index, dataset_name in enumerate(datasets):
            single_fake = fake[index : index + 1]
            single_real = real[index : index + 1]
            values = self._paired_values(single_fake, single_real)
            for key in ("all", dataset_name):
                for metric_name, value in values.items():
                    self.sums[key][metric_name] += value
                self.counts[key] += 1

        if self.fid is not None:
            self.fid.update(to_uint8(to_three_channel(real)), real=True)
            self.fid.update(to_uint8(to_three_channel(fake)), real=False)

    def _paired_values(self, fake: torch.Tensor, real: torch.Tensor) -> dict[str, float]:
        values: dict[str, float] = {}
        # data_range=2.0 because tensors live in [-1, 1], not [0, 1].
        if self._psnr is not None and "psnr" in self.requested:
            values["psnr"] = float(self._psnr(fake, real, data_range=2.0))
        if self._ssim is not None and "ssim" in self.requested:
            values["ssim"] = float(self._ssim(fake, real, data_range=2.0))
        if self._lpips is not None:
            values["lpips"] = float(
                self._lpips(to_three_channel(fake), to_three_channel(real))
            )
        return values

    # --------------------------------------------------------------- readout

    def compute(self) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for key, sums in self.sums.items():
            count = max(1, self.counts[key])
            results[key] = {name: total / count for name, total in sums.items()}
            results[key]["n"] = float(self.counts[key])

        if self.fid is not None:
            try:
                results.setdefault("all", {})["fid"] = float(self.fid.compute())
            except (RuntimeError, ValueError) as exc:
                # FID needs enough samples to estimate a covariance matrix.
                self.skipped["fid"] = f"not enough samples ({exc})"
        return results

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()
        if self.fid is not None:
            self.fid.reset()


@torch.no_grad()
def unpaired_fid(
    generated: torch.Tensor,
    reference: torch.Tensor,
    device: str = "cpu",
    feature: int = 2048,
) -> Optional[float]:
    """FID between generated thermal and a real thermal set with no pairing.

    This is what HIT-UAV is for. PSNR against a matched frame cannot tell us
    whether the output resembles genuine high-altitude UAV thermal imagery;
    comparing distributions can.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return None

    metric = FrechetInceptionDistance(feature=feature, normalize=False).to(device)
    metric.update(to_uint8(to_three_channel(reference.to(device))), real=True)
    metric.update(to_uint8(to_three_channel(generated.to(device))), real=False)
    try:
        return float(metric.compute())
    except (RuntimeError, ValueError):
        return None


def format_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the per-dataset metric table for the console and the notebook."""
    if not results:
        return "(no metrics computed)"

    columns: list[str] = []
    for values in results.values():
        for name in values:
            if name not in columns:
                columns.append(name)
    columns.sort(key=lambda name: (name == "n", name))

    header = f"{'dataset':<16}" + "".join(f"{name:>12}" for name in columns)
    lines = [header, "-" * len(header)]
    for key in sorted(results, key=lambda k: (k != "all", k)):
        row = f"{key:<16}"
        for name in columns:
            value = results[key].get(name)
            if value is None:
                row += f"{'-':>12}"
            elif name == "n":
                row += f"{int(value):>12,}"
            else:
                row += f"{value:>12.4f}"
        lines.append(row)
    return "\n".join(lines)
