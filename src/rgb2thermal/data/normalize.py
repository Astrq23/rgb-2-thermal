"""Pixel-level normalisation that makes three different sensors comparable.

Deliberately depends only on PIL + numpy (no torch), so the whole normalisation
layer is unit-testable on a laptop with no GPU stack installed.

Two things happen here:

**Geometry.** DroneVehicle ships 840x712 images padded with a 100 px white
border used for boundary annotation; that border must be cropped or the model
spends capacity learning to paint a white frame. FLIR's RGB and thermal cameras
have different fields of view, approximated with a centre crop.

**Intensity.** Thermal files arrive as 3-channel JPEG, 8-bit grayscale PNG, and
everything in between, each with its own camera gain. We flatten to one channel
and apply a per-image percentile stretch. That stretch is what allows three
unrelated thermal cameras to be trained on jointly -- and it is also the reason
this model predicts *thermal appearance*, not absolute temperature. See
README "What the model does and does not learn".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

THERMAL_NORM_MODES = ("percentile", "minmax", "none")


@dataclass
class ModalityPreprocess:
    """Geometric/intensity quirks for one modality of one dataset."""

    #: Pixels to remove from every side (DroneVehicle's annotation border).
    crop_border: int = 0
    #: Fraction of width/height to keep in a centre crop, for FOV matching.
    #: 1.0 disables it.
    fov_crop: float = 1.0
    #: Some IR exports are black-hot rather than white-hot.
    invert: bool = False

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "ModalityPreprocess":
        return cls(**(data or {}))


@dataclass
class PreprocessSpec:
    rgb: ModalityPreprocess = field(default_factory=ModalityPreprocess)
    thermal: ModalityPreprocess = field(default_factory=ModalityPreprocess)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "PreprocessSpec":
        data = data or {}
        return cls(
            rgb=ModalityPreprocess.from_dict(data.get("rgb")),
            thermal=ModalityPreprocess.from_dict(data.get("thermal")),
        )


def crop_border(image: Image.Image, pixels: int) -> Image.Image:
    """Remove ``pixels`` from all four sides."""
    if pixels <= 0:
        return image
    width, height = image.size
    if width <= 2 * pixels or height <= 2 * pixels:
        # Nonsensical config for this image; better to pass it through than to
        # produce a zero-sized crop that fails much later in the DataLoader.
        return image
    return image.crop((pixels, pixels, width - pixels, height - pixels))


def center_crop_fraction(image: Image.Image, fraction: float) -> Image.Image:
    """Centre crop keeping ``fraction`` of each dimension."""
    if fraction >= 1.0 or fraction <= 0.0:
        return image
    width, height = image.size
    new_width = max(1, int(round(width * fraction)))
    new_height = max(1, int(round(height * fraction)))
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height))


def apply_geometry(image: Image.Image, spec: ModalityPreprocess) -> Image.Image:
    return center_crop_fraction(crop_border(image, spec.crop_border), spec.fov_crop)


def load_rgb(path: Path | str, spec: Optional[ModalityPreprocess] = None) -> Image.Image:
    """Load an RGB image, applying that dataset's geometric fixes."""
    spec = spec or ModalityPreprocess()
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    return apply_geometry(image, spec)


def load_thermal(path: Path | str, spec: Optional[ModalityPreprocess] = None) -> Image.Image:
    """Load a thermal image as single-channel ``L``, applying geometric fixes.

    ``convert("L")`` collapses the 3-channel JPEGs that DroneVehicle and LLVIP
    use; for genuinely grayscale files it is a no-op.
    """
    spec = spec or ModalityPreprocess()
    with Image.open(path) as handle:
        image = handle.convert("L")
    image = apply_geometry(image, spec)
    if spec.invert:
        image = Image.eval(image, lambda value: 255 - value)
    return image


def stretch_intensity(
    array: np.ndarray,
    mode: str = "percentile",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Map an intensity array to float32 in ``[0, 1]``.

    ``percentile`` (default) clips to the 1st/99th percentile before scaling.
    This removes the per-camera gain/offset differences that would otherwise
    make a merged multi-sensor dataset incoherent, and it is robust to the few
    saturated pixels that thermal sensors routinely produce.
    """
    if mode not in THERMAL_NORM_MODES:
        raise ValueError(f"thermal_norm must be one of {THERMAL_NORM_MODES}, got {mode!r}")

    array = array.astype(np.float32)
    if mode == "none":
        return np.clip(array / 255.0, 0.0, 1.0)

    if mode == "minmax":
        low, high = float(array.min()), float(array.max())
    else:
        low, high = np.percentile(array, [low_percentile, high_percentile])
        low, high = float(low), float(high)

    # A flat image (all one temperature) has no range to stretch.
    if high - low < 1e-6:
        return np.clip(array / 255.0, 0.0, 1.0)

    return np.clip((array - low) / (high - low), 0.0, 1.0)


def to_signed(array: np.ndarray) -> np.ndarray:
    """``[0, 1]`` -> ``[-1, 1]``, the range a ``tanh`` generator outputs."""
    return array * 2.0 - 1.0


def to_unit(array: np.ndarray) -> np.ndarray:
    """``[-1, 1]`` -> ``[0, 1]``, for saving and for metrics."""
    return np.clip((array + 1.0) / 2.0, 0.0, 1.0)


def thermal_to_array(
    image: Image.Image,
    mode: str = "percentile",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """PIL ``L`` image -> float32 ``HxW`` in ``[-1, 1]``."""
    array = np.asarray(image, dtype=np.uint8)
    return to_signed(stretch_intensity(array, mode, low_percentile, high_percentile))


def rgb_to_array(image: Image.Image) -> np.ndarray:
    """PIL ``RGB`` image -> float32 ``HxWx3`` in ``[-1, 1]``."""
    array = np.asarray(image, dtype=np.uint8).astype(np.float32) / 255.0
    return to_signed(array)
