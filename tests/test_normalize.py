"""Pixel normalisation: the border crop and the cross-sensor intensity stretch."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rgb2thermal.data.normalize import (
    ModalityPreprocess,
    PreprocessSpec,
    center_crop_fraction,
    crop_border,
    load_rgb,
    load_thermal,
    rgb_to_array,
    stretch_intensity,
    thermal_to_array,
    to_signed,
    to_unit,
)
from tests.conftest import BORDER, DRONEVEHICLE_SIZE


def test_dronevehicle_border_crop_gives_640x512(fake_dronevehicle):
    """840x712 minus a 100px border on each side is the documented 640x512."""
    path = next((fake_dronevehicle / "train" / "trainimg").glob("*.jpg"))
    image = load_rgb(path, ModalityPreprocess(crop_border=BORDER))
    assert image.size == (640, 512)
    assert DRONEVEHICLE_SIZE == (840, 712)


def test_border_crop_removes_the_white_frame(fake_dronevehicle):
    path = next((fake_dronevehicle / "train" / "trainimgr").glob("*.jpg"))
    before = np.asarray(load_thermal(path, ModalityPreprocess()))
    after = np.asarray(load_thermal(path, ModalityPreprocess(crop_border=BORDER)))
    assert before[0, 0] == 255           # white border present
    assert after.mean() < before.mean()  # and gone afterwards


def test_crop_border_is_a_noop_when_it_would_empty_the_image():
    """A wrong config must not produce a zero-sized crop that fails later."""
    tiny = Image.new("L", (10, 10))
    assert crop_border(tiny, 100).size == (10, 10)


def test_center_crop_fraction():
    image = Image.new("RGB", (100, 200))
    assert center_crop_fraction(image, 0.5).size == (50, 100)
    assert center_crop_fraction(image, 1.0).size == (100, 200)


def test_thermal_is_single_channel(fake_llvip):
    path = next((fake_llvip / "LLVIP" / "infrared" / "train").glob("*.jpg"))
    image = load_thermal(path)
    assert image.mode == "L"
    assert np.asarray(image).ndim == 2


def test_thermal_tensor_range(fake_llvip):
    path = next((fake_llvip / "LLVIP" / "infrared" / "train").glob("*.jpg"))
    array = thermal_to_array(load_thermal(path))
    assert array.dtype == np.float32
    assert array.min() >= -1.0 and array.max() <= 1.0


def test_rgb_tensor_range_and_shape(fake_llvip):
    path = next((fake_llvip / "LLVIP" / "visible" / "train").glob("*.jpg"))
    array = rgb_to_array(load_rgb(path))
    assert array.ndim == 3 and array.shape[2] == 3
    assert array.min() >= -1.0 and array.max() <= 1.0


def test_invert_flips_polarity(tmp_path):
    """Black-hot exports need flipping before they can join white-hot ones."""
    path = tmp_path / "blackhot.png"
    Image.fromarray(np.full((8, 8), 200, dtype=np.uint8)).save(path)

    normal = np.asarray(load_thermal(path, ModalityPreprocess(invert=False)))
    inverted = np.asarray(load_thermal(path, ModalityPreprocess(invert=True)))

    assert normal[0, 0] == 200
    assert inverted[0, 0] == 55


# ------------------------------------------------------- intensity stretching


def test_percentile_stretch_equalises_two_camera_gains():
    """The reason a merged multi-sensor dataset is coherent at all.

    Two sensors imaging the same scene with different gain and offset must land
    on the same normalised values, otherwise the model learns the sensor rather
    than the scene.
    """
    scene = np.linspace(0, 100, 256, dtype=np.float32).reshape(16, 16)
    camera_a = scene                      # low gain, no offset
    camera_b = scene * 1.8 + 40.0         # high gain, offset

    out_a = stretch_intensity(camera_a, "percentile")
    out_b = stretch_intensity(camera_b, "percentile")
    assert np.allclose(out_a, out_b, atol=1e-4)


def test_minmax_stretch_spans_the_full_range():
    array = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    out = stretch_intensity(array, "minmax")
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_none_mode_only_rescales_by_255():
    array = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    out = stretch_intensity(array, "none")
    assert np.allclose(out, array / 255.0)


def test_flat_image_does_not_divide_by_zero():
    """A uniform-temperature frame has no range; it must not produce NaNs."""
    flat = np.full((8, 8), 77, dtype=np.uint8)
    for mode in ("percentile", "minmax", "none"):
        out = stretch_intensity(flat, mode)
        assert np.isfinite(out).all()


def test_percentile_clips_outliers():
    """A few saturated pixels must not compress the whole scene."""
    array = np.full((10, 10), 100, dtype=np.float32)
    array[0, 0] = 0
    array[0, 1] = 255
    out = stretch_intensity(array, "percentile", 1.0, 99.0)
    assert out.min() == 0.0 and out.max() == 1.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="thermal_norm"):
        stretch_intensity(np.zeros((4, 4)), "bogus")


def test_signed_unit_roundtrip():
    values = np.linspace(0, 1, 32, dtype=np.float32)
    assert np.allclose(to_unit(to_signed(values)), values, atol=1e-6)


# ----------------------------------------------------------------- spec parsing


def test_preprocess_spec_from_yaml_shape():
    spec = PreprocessSpec.from_dict({"rgb": {"crop_border": 100}, "thermal": {"invert": True}})
    assert spec.rgb.crop_border == 100
    assert spec.thermal.invert is True
    assert spec.thermal.crop_border == 0


def test_preprocess_spec_defaults_when_absent():
    spec = PreprocessSpec.from_dict(None)
    assert spec.rgb.crop_border == 0
    assert spec.rgb.fov_crop == 1.0
