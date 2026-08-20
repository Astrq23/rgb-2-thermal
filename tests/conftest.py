"""Synthetic dataset trees that mimic each real Kaggle layout.

The whole point of these fixtures is that the normalisation layer can be tested
end to end without downloading a single byte: every adapter quirk we care about
(DroneVehicle's white border, LLVIP's parallel visible/infrared trees, FLIR's
per-modality filename hashes, HIT-UAV having no RGB at all) is reproduced here
in miniature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

DRONEVEHICLE_SIZE = (840, 712)   # as distributed, border included
BORDER = 100                     # -> 640 x 512 once cropped


def _gradient(width: int, height: int, seed: int) -> np.ndarray:
    """Deterministic non-flat content, so percentile stretching has range."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    noise = rng.integers(0, 40, size=(height, width))
    return np.clip(ramp + noise, 0, 255).astype(np.uint8)


def write_rgb(path: Path, size=(64, 64), seed: int = 0, border: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    channel = _gradient(width, height, seed)
    array = np.stack([channel, channel[::-1], channel[:, ::-1]], axis=-1)
    if border:
        array[:border, :] = 255
        array[-border:, :] = 255
        array[:, :border] = 255
        array[:, -border:] = 255
    Image.fromarray(array).save(path)
    return path


def write_thermal(path: Path, size=(64, 64), seed: int = 0, border: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    array = _gradient(width, height, seed + 1000)
    if border:
        array[:border, :] = 255
        array[-border:, :] = 255
        array[:, :border] = 255
        array[:, -border:] = 255
    Image.fromarray(array).save(path)
    return path


@pytest.fixture
def fake_dronevehicle(tmp_path: Path) -> Path:
    """train/{trainimg,trainimgr} + val/{valimg,valimgr}, white border included."""
    root = tmp_path / "visdrone-dronevehicle"
    for index in range(6):
        name = f"{index:05d}.jpg"
        write_rgb(root / "train" / "trainimg" / name, DRONEVEHICLE_SIZE, index, BORDER)
        write_thermal(root / "train" / "trainimgr" / name, DRONEVEHICLE_SIZE, index, BORDER)
    for index in range(2):
        name = f"{index + 100:05d}.jpg"
        write_rgb(root / "val" / "valimg" / name, DRONEVEHICLE_SIZE, index, BORDER)
        write_thermal(root / "val" / "valimgr" / name, DRONEVEHICLE_SIZE, index, BORDER)

    # An RGB frame whose infrared counterpart is missing: must be dropped, not
    # paired with something else.
    write_rgb(root / "train" / "trainimg" / "99999.jpg", DRONEVEHICLE_SIZE, 42, BORDER)
    # Annotations live alongside the images and must be ignored.
    label_dir = root / "train" / "trainlabel"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "00000.xml").write_text("<annotation/>", encoding="utf-8")
    return root


@pytest.fixture
def fake_llvip(tmp_path: Path) -> Path:
    """visible/{train,test} mirrored by infrared/{train,test}."""
    root = tmp_path / "llvip-dataset" / "LLVIP"
    for index in range(4):
        name = f"01{index:04d}.jpg"
        write_rgb(root / "visible" / "train" / name, (96, 80), index)
        write_thermal(root / "infrared" / "train" / name, (96, 80), index)
    for index in range(2):
        name = f"19{index:04d}.jpg"
        write_rgb(root / "visible" / "test" / name, (96, 80), index + 10)
        write_thermal(root / "infrared" / "test" / name, (96, 80), index + 10)
    return tmp_path / "llvip-dataset"


@pytest.fixture
def fake_flir(tmp_path: Path) -> Path:
    """images_{rgb,thermal}_train/data with a different hash per modality."""
    root = tmp_path / "teledyne-flir-adas-thermal-dataset-v2"
    videos = ["BzZspxAweF8AnKhWK", "QmXhtRbLpN4vJ2wKd"]
    for video_index, video in enumerate(videos):
        for frame in range(3):
            stem = f"video-{video}-frame-{frame:06d}"
            seed = video_index * 10 + frame
            write_rgb(
                root / "images_rgb_train" / "data" / f"{stem}-RGBhash{frame}.jpg", (128, 96), seed
            )
            write_thermal(
                root / "images_thermal_train" / "data" / f"{stem}-THMhash{frame}.jpg", (80, 64), seed
            )
    # Thermal frame with no RGB partner.
    write_thermal(
        root / "images_thermal_train" / "data" / "video-ORPHAN0000000-frame-000009-x.jpg",
        (80, 64),
        99,
    )
    return root


@pytest.fixture
def fake_hituav(tmp_path: Path) -> Path:
    """Thermal only, and no modality token anywhere in the path."""
    root = tmp_path / "hituav-a-highaltitude-infrared-thermal-dataset"
    for split, count in (("train", 4), ("val", 2)):
        for index in range(count):
            write_thermal(root / "normal" / "images" / split / f"{index:04d}.jpg", (64, 48), index)
    return root
