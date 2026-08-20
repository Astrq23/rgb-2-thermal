"""Torch Dataset over the unified manifest.

Everything dataset-specific has already been resolved by the adapters and by
``normalize.py``; by the time a sample reaches here it is just "an RGB image and
its thermal counterpart". The only per-dataset knowledge still needed is the
:class:`PreprocessSpec` (border crop / FOV crop), looked up by dataset name.

Images are decoded on the fly straight from ``/kaggle/input`` rather than being
pre-resized into ``/kaggle/working``. That trades a little CPU for staying well
clear of Kaggle's ~20 GB working-directory limit, and it means changing
``image_size`` costs nothing.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from .adapters.base import DatasetSpec
from .normalize import (
    PreprocessSpec,
    load_rgb,
    load_thermal,
    rgb_to_array,
    thermal_to_array,
)
from .sampler import build_weighted_sampler

RESIZE_MODES = ("crop", "stretch")


class PairedThermalDataset(Dataset):
    """RGB -> thermal pairs drawn from the merged manifest."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_cfg: DataConfig,
        specs: Sequence[DatasetSpec],
        train: bool = True,
        domain_index: Optional[dict[str, int]] = None,
    ):
        if data_cfg.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}")

        self.frame = frame.reset_index(drop=True)
        self.cfg = data_cfg
        self.train = train
        self.preprocess = {
            spec.name: PreprocessSpec.from_dict(spec.preprocess) for spec in specs
        }
        # Stable name -> id map, so a checkpoint's domain embedding stays valid
        # even if a dataset is temporarily disabled.
        self.domain_index = domain_index or {
            name: idx for idx, name in enumerate(sorted({s.name for s in specs}))
        }

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def datasets(self) -> list[str]:
        return self.frame["dataset"].tolist()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        spec = self.preprocess.get(row["dataset"], PreprocessSpec())

        rgb = load_rgb(row["rgb_path"], spec.rgb)
        thermal = load_thermal(row["thermal_path"], spec.thermal)

        # After per-modality geometric fixes the two can differ in size (FLIR
        # especially). Put thermal on the RGB grid before any joint transform,
        # otherwise a shared crop box would not describe the same scene region.
        if thermal.size != rgb.size:
            thermal = thermal.resize(rgb.size, Image.BICUBIC)

        rgb, thermal = self._geometric(rgb, thermal)

        rgb_array = rgb_to_array(rgb)
        thermal_array = thermal_to_array(
            thermal,
            mode=self.cfg.thermal_norm,
            low_percentile=self.cfg.low_percentile,
            high_percentile=self.cfg.high_percentile,
        )

        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb_array.transpose(2, 0, 1))),
            "thermal": torch.from_numpy(np.ascontiguousarray(thermal_array))[None],
            "domain": torch.tensor(self.domain_index.get(row["dataset"], 0), dtype=torch.long),
            "dataset": row["dataset"],
            "rgb_path": row["rgb_path"],
        }

    # ------------------------------------------------------------- transforms

    def _geometric(self, rgb: Image.Image, thermal: Image.Image) -> tuple[Image.Image, Image.Image]:
        size = self.cfg.image_size

        if self.cfg.resize_mode == "stretch":
            rgb = rgb.resize((size, size), Image.BICUBIC)
            thermal = thermal.resize((size, size), Image.BICUBIC)
        else:
            load_size = max(self.cfg.load_size, size) if self.train else size
            rgb = _resize_short_side(rgb, load_size)
            thermal = _resize_short_side(thermal, load_size)
            left, top = self._crop_origin(rgb.size, size)
            box = (left, top, left + size, top + size)
            rgb = rgb.crop(box)
            thermal = thermal.crop(box)

        if self.train and self.cfg.hflip and random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)

        return rgb, thermal

    def _crop_origin(self, image_size: tuple[int, int], crop: int) -> tuple[int, int]:
        """Random crop while training, centre crop otherwise (reproducible metrics)."""
        width, height = image_size
        max_left = max(0, width - crop)
        max_top = max(0, height - crop)
        if self.train:
            return random.randint(0, max_left), random.randint(0, max_top)
        return max_left // 2, max_top // 2


def _resize_short_side(image: Image.Image, target: int) -> Image.Image:
    """Scale so the shorter side equals ``target``, preserving aspect ratio."""
    width, height = image.size
    scale = target / min(width, height)
    new_size = (max(target, int(round(width * scale))), max(target, int(round(height * scale))))
    return image.resize(new_size, Image.BICUBIC)


class ThermalReferenceDataset(Dataset):
    """Thermal images with no RGB partner (HIT-UAV).

    Never used for training -- it exists so unpaired FID can ask "does the
    generated output belong to the distribution of real UAV thermal imagery?",
    a question no paired metric can answer.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_cfg: DataConfig,
        specs: Sequence[DatasetSpec],
    ):
        self.frame = frame.reset_index(drop=True)
        self.cfg = data_cfg
        self.preprocess = {
            spec.name: PreprocessSpec.from_dict(spec.preprocess) for spec in specs
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        spec = self.preprocess.get(row["dataset"], PreprocessSpec())

        thermal = load_thermal(row["thermal_path"], spec.thermal)
        size = self.cfg.image_size
        thermal = _resize_short_side(thermal, size)
        width, height = thermal.size
        left, top = (width - size) // 2, (height - size) // 2
        thermal = thermal.crop((left, top, left + size, top + size))

        array = thermal_to_array(
            thermal,
            mode=self.cfg.thermal_norm,
            low_percentile=self.cfg.low_percentile,
            high_percentile=self.cfg.high_percentile,
        )
        return {"thermal": torch.from_numpy(np.ascontiguousarray(array))[None]}


# ------------------------------------------------------------------ factories


def trainable(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for paired training: both modalities present."""
    return frame[(frame["rgb_path"] != "") & (frame["thermal_path"] != "")]


def build_dataloaders(
    frame: pd.DataFrame,
    cfg: DataConfig,
    specs: Sequence[DatasetSpec],
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, Optional[DataLoader], dict[str, int]]:
    """Build the train loader (weighted mixture) and the val loader (plain)."""
    usable = trainable(frame)
    domain_index = {name: idx for idx, name in enumerate(sorted({s.name for s in specs}))}

    train_frame = usable[usable["split"] == "train"]
    val_frame = usable[usable["split"] == "val"]

    if cfg.limit_train:
        train_frame = train_frame.head(cfg.limit_train)
    if cfg.limit_val:
        val_frame = val_frame.head(cfg.limit_val)

    if train_frame.empty:
        raise RuntimeError(
            "No training rows in the manifest. Rebuild it with "
            "`python scripts/build_manifest.py` and check the per-dataset counts."
        )

    train_set = PairedThermalDataset(train_frame, cfg, specs, train=True, domain_index=domain_index)
    sampler = build_weighted_sampler(
        datasets=train_set.datasets,
        weights=_resolve_weights(cfg, specs),
        num_samples=cfg.samples_per_epoch or len(train_set),
        seed=seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader = None
    if not val_frame.empty:
        val_set = PairedThermalDataset(
            val_frame, cfg, specs, train=False, domain_index=domain_index
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, domain_index


def _resolve_weights(cfg: DataConfig, specs: Sequence[DatasetSpec]) -> dict[str, float]:
    """Config weights win; datasets absent from the config keep their own."""
    weights = {spec.name: float(spec.sample_weight) for spec in specs}
    weights.update({k: float(v) for k, v in cfg.sample_weights.items()})
    return weights
