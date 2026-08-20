"""Adapter discovery against miniature copies of each real dataset layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from rgb2thermal.data.adapters.base import DatasetSpec, GenericPairedAdapter
from rgb2thermal.data.adapters.flir_adas_v2 import FlirAdasV2Adapter
from rgb2thermal.data.adapters.registry import build_adapter


def spec(name: str, root: Path, **kwargs) -> DatasetSpec:
    kwargs.setdefault("exclude_globs", ["**/*label*/**"])
    return DatasetSpec(name=name, search_roots=[str(root)], **kwargs)


# ------------------------------------------------------------- DroneVehicle


def test_dronevehicle_pairs_img_with_imgr(fake_dronevehicle):
    records = GenericPairedAdapter(spec("dronevehicle", fake_dronevehicle)).discover()

    assert len(records) == 8  # 6 train + 2 val; the orphan RGB is excluded
    for record in records:
        assert Path(record.rgb_path).name == Path(record.thermal_path).name
        # RGB must come from an *img* folder, thermal from an *imgr* folder.
        assert Path(record.rgb_path).parent.name.endswith("img")
        assert Path(record.thermal_path).parent.name.endswith("imgr")


def test_unmatched_files_are_dropped(fake_dronevehicle):
    """The orphan RGB frame must vanish, never be paired with a stranger."""
    records = GenericPairedAdapter(spec("dronevehicle", fake_dronevehicle)).discover()
    assert not any("99999" in record.rgb_path for record in records)


def test_split_hints_are_read_from_the_path(fake_dronevehicle):
    records = GenericPairedAdapter(spec("dronevehicle", fake_dronevehicle)).discover()
    hints = {record.split_hint for record in records}
    assert hints == {"train", "val"}


def test_label_folders_are_excluded(fake_dronevehicle):
    records = GenericPairedAdapter(spec("dronevehicle", fake_dronevehicle)).discover()
    # Scoped to the folder name: pytest's tmp_path is named after the test,
    # so the absolute path legitimately contains "label".
    folders = {Path(r.rgb_path).parent.name for r in records}
    folders |= {Path(r.thermal_path).parent.name for r in records}
    assert all("label" not in folder for folder in folders)


# --------------------------------------------------------------------- LLVIP


def test_llvip_pairs_visible_with_infrared(fake_llvip):
    records = GenericPairedAdapter(spec("llvip", fake_llvip)).discover()

    assert len(records) == 6  # 4 train + 2 test
    for record in records:
        assert "visible" in record.rgb_path
        assert "infrared" in record.thermal_path
        assert Path(record.rgb_path).name == Path(record.thermal_path).name


def test_llvip_group_regex_extracts_the_location_id(fake_llvip):
    records = GenericPairedAdapter(
        spec("llvip", fake_llvip, group_regex=r"/(\d{2})\d+\.[A-Za-z]+$")
    ).discover()
    assert {record.group_key for record in records} == {"01", "19"}


def test_group_bucket_fallback_when_regex_misses(fake_llvip):
    """A mirror that renames files must degrade to bucketing, not to crashing."""
    records = GenericPairedAdapter(
        spec("llvip", fake_llvip, group_regex=r"/(NOPE\d+)\.jpg$", group_bucket=2)
    ).discover()
    assert all(record.group_key for record in records)
    assert len({record.group_key for record in records}) > 1


# ---------------------------------------------------------------------- FLIR


def test_flir_pairs_across_different_filename_hashes(fake_flir):
    records = FlirAdasV2Adapter(spec("flir_v2", fake_flir, adapter="flir_adas_v2")).discover()

    assert len(records) == 6  # 2 videos x 3 frames; the orphan thermal drops out
    for record in records:
        assert "images_rgb_train" in record.rgb_path
        assert "images_thermal_train" in record.thermal_path
        # Same frame, different per-modality hash.
        assert Path(record.rgb_path).name != Path(record.thermal_path).name


def test_flir_groups_by_source_video(fake_flir):
    records = FlirAdasV2Adapter(spec("flir_v2", fake_flir, adapter="flir_adas_v2")).discover()
    assert {record.group_key for record in records} == {
        "bzzspxawef8ankhwk",
        "qmxhtrblpn4vj2wkd",
    }


def test_flir_orphan_thermal_is_dropped(fake_flir):
    records = FlirAdasV2Adapter(spec("flir_v2", fake_flir, adapter="flir_adas_v2")).discover()
    assert not any("ORPHAN" in record.thermal_path for record in records)


# ------------------------------------------------------------------- HIT-UAV


def test_hituav_yields_thermal_only_records(fake_hituav):
    records = GenericPairedAdapter(
        spec("hituav", fake_hituav, paired=False, force_modality="thermal")
    ).discover()

    assert len(records) == 6
    assert all(record.rgb_path == "" for record in records)
    assert all(record.thermal_path for record in records)


def test_hituav_without_force_modality_finds_nothing(fake_hituav):
    """Its folders carry no modality token, which is exactly why the flag exists."""
    records = GenericPairedAdapter(spec("hituav", fake_hituav, paired=False)).discover()
    assert records == []


# ------------------------------------------------------------------- general


def test_missing_dataset_is_reported_not_raised(tmp_path):
    adapter = GenericPairedAdapter(spec("ghost", tmp_path / "does-not-exist"))
    assert adapter.is_available() is False
    assert adapter.discover() == []


def test_wildcard_search_root_matches_a_renamed_mirror(fake_llvip):
    """Community mirrors rename slugs; the globs absorb that."""
    pattern = str(fake_llvip.parent / "*llvip*")
    adapter = GenericPairedAdapter(DatasetSpec(name="llvip", search_roots=[pattern]))
    assert adapter.is_available()
    assert len(adapter.discover()) == 6


def test_registry_builds_the_declared_adapter(fake_flir):
    adapter = build_adapter(spec("flir_v2", fake_flir, adapter="flir_adas_v2"))
    assert isinstance(adapter, FlirAdasV2Adapter)


def test_registry_rejects_unknown_adapter(tmp_path):
    with pytest.raises(KeyError, match="Unknown adapter"):
        build_adapter(spec("x", tmp_path, adapter="nope"))


def test_dataset_spec_rejects_typos():
    with pytest.raises(ValueError, match="Unknown key"):
        DatasetSpec.from_dict({"name": "x", "sample_weigth": 0.5})
