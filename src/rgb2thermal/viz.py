"""Visual output: comparison grids and alignment overlays.

Looking at the pictures is the fastest bug detector this project has. A grid
that shows RGB / real thermal / generated thermal side by side catches inverted
polarity, misalignment and mode collapse in a glance, long before any metric
moves enough to notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

#: Thermal is single-channel, but people read heat maps far better in a
#: perceptually ordered colour map than in grayscale.
DEFAULT_COLORMAP = "inferno"


def _use_agg() -> None:
    """Force a non-interactive backend before pyplot is imported."""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")


def rgb_tensor_to_array(tensor) -> np.ndarray:
    """``(3, H, W)`` in [-1, 1] -> ``(H, W, 3)`` uint8."""
    array = tensor.detach().float().cpu().numpy()
    array = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    return (array.transpose(1, 2, 0) * 255).astype(np.uint8)


def thermal_tensor_to_array(tensor) -> np.ndarray:
    """``(1, H, W)`` in [-1, 1] -> ``(H, W)`` uint8."""
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    array = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def colorize(gray: np.ndarray, colormap: str = DEFAULT_COLORMAP) -> np.ndarray:
    """Apply a matplotlib colormap to a ``(H, W)`` uint8 array."""
    _use_agg()
    from matplotlib import colormaps

    mapper = colormaps[colormap]
    return (mapper(gray / 255.0)[..., :3] * 255).astype(np.uint8)


def save_comparison_grid(
    rgb: "np.ndarray",
    real_thermal: "np.ndarray",
    fake_thermal: "np.ndarray",
    path: Path | str,
    max_items: int = 8,
    titles: Optional[Sequence[str]] = None,
    colormap: str = DEFAULT_COLORMAP,
) -> Path:
    """Save an ``N x 3`` grid: RGB input, real thermal, generated thermal.

    Accepts torch tensors or numpy arrays shaped ``(N, C, H, W)`` in [-1, 1].
    """
    _use_agg()
    import matplotlib.pyplot as plt

    count = min(max_items, len(rgb))
    figure, axes = plt.subplots(count, 3, figsize=(9, 3 * count), squeeze=False)

    for row in range(count):
        panels = [
            (rgb_tensor_to_array(rgb[row]), "RGB input", None),
            (colorize(thermal_tensor_to_array(real_thermal[row]), colormap), "Thermal (real)", None),
            (colorize(thermal_tensor_to_array(fake_thermal[row]), colormap), "Thermal (generated)", None),
        ]
        for column, (image, label, _) in enumerate(panels):
            axis = axes[row][column]
            axis.imshow(image)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(label, fontsize=10)
        if titles is not None and row < len(titles):
            axes[row][0].set_ylabel(titles[row], fontsize=8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def save_alignment_overlay(
    rgb_image: Image.Image,
    thermal_image: Image.Image,
    path: Path | str,
    title: str = "",
) -> Path:
    """Four-panel alignment check: RGB, thermal, 50/50 blend, thermal edges on RGB.

    The edge panel is the decisive one -- if the thermal edges do not sit on the
    RGB structures, the pair is misregistered and training on it teaches the
    model to blur. This is the check that decides whether FLIR stays enabled.
    """
    _use_agg()
    import matplotlib.pyplot as plt

    thermal_resized = thermal_image.resize(rgb_image.size, Image.BICUBIC)
    rgb_array = np.asarray(rgb_image.convert("RGB"), dtype=np.float32)
    thermal_array = np.asarray(thermal_resized.convert("L"), dtype=np.float32)

    thermal_rgb = colorize(thermal_array.astype(np.uint8)).astype(np.float32)
    blend = (0.5 * rgb_array + 0.5 * thermal_rgb).astype(np.uint8)

    # Cheap gradient magnitude; avoids pulling in scipy/cv2.
    gradient_y, gradient_x = np.gradient(thermal_array)
    edges = np.hypot(gradient_x, gradient_y)
    if edges.max() > 0:
        edges = edges / edges.max()
    overlay = rgb_array.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 255 * edges, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1 - edges), 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1 - edges), 0, 255)

    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, (image, label) in zip(
        axes,
        [
            (rgb_array.astype(np.uint8), "RGB"),
            (thermal_rgb.astype(np.uint8), "Thermal"),
            (blend, "50/50 blend"),
            (overlay.astype(np.uint8), "Thermal edges on RGB"),
        ],
    ):
        axis.imshow(image)
        axis.set_title(label, fontsize=11)
        axis.set_axis_off()

    if title:
        figure.suptitle(title, fontsize=12)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(figure)
    return path


def save_thermal_image(tensor, path: Path | str, colormap: Optional[str] = None) -> Path:
    """Write one generated thermal image, grayscale by default."""
    gray = thermal_tensor_to_array(tensor)
    array = colorize(gray, colormap) if colormap else gray
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path
