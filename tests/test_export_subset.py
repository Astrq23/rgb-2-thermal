"""The subset export must survive a full round trip.

Exporting is only useful if the compact copy behaves exactly like the originals
downstream. The risk is silent metadata loss: if the source dataset name, the
scene grouping or the reference split do not survive, training still runs but
the sampling weights, the per-dataset metrics and the leakage guarantee are all
quietly wrong.

These tests run the real ``scripts/export_subset.py`` and re-discover its output
through the real ``configs/datasets/subset*.yaml`` -- including their regexes,
which is where a mistake would actually hide.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from rgb2thermal.config import load_dataset_specs
from rgb2thermal.data.adapters.base import DatasetSpec
from rgb2thermal.data.adapters.registry import build_adapter
from rgb2thermal.data.manifest import REFERENCE_SPLIT, assign_splits, build_manifest, records_to_frame, save_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SIZE = 64


def source_spec(name: str, root: Path, **kwargs) -> DatasetSpec:
    kwargs.setdefault("exclude_globs", ["**/*label*/**"])
    return DatasetSpec(name=name, search_roots=[str(root)], **kwargs)


def subset_spec(config_name: str, root: Path) -> DatasetSpec:
    """The shipped config, retargeted at the exported folder."""
    spec = load_dataset_specs([f"configs/datasets/{config_name}"])[0]
    spec.search_roots = [str(root)]
    return spec


@pytest.fixture
def exported(fake_dronevehicle, fake_llvip, fake_hituav, tmp_path) -> Path:
    """Manifest of the originals -> run the exporter -> return the output dir."""
    specs = [
        source_spec("dronevehicle", fake_dronevehicle),
        source_spec("llvip", fake_llvip, group_regex=r"/(\d{2})\d+\.[A-Za-z]+$"),
        source_spec("hituav", fake_hituav, paired=False, force_modality="thermal"),
    ]
    frame = build_manifest(specs, verbose=False)
    manifest = save_manifest(frame, tmp_path / "manifest.csv")
    out = tmp_path / "subset"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_subset.py",
            "--manifest", str(manifest),
            "--out", str(out),
            "--size", str(EXPORT_SIZE),
            "--per-dataset", "0",
            "--reference-max", "0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def test_export_writes_both_modalities_and_reference(exported):
    assert (exported / "dronevehicle" / "rgb").is_dir()
    assert (exported / "dronevehicle" / "thermal").is_dir()
    assert (exported / "llvip" / "rgb").is_dir()
    # Thermal-only sources are segregated so subset.yaml can stay `paired: true`.
    assert (exported / "reference" / "hituav" / "thermal").is_dir()
    assert not (exported / "hituav").exists()
    assert (exported / "EXPORT_INFO.json").is_file()


def test_exported_images_are_downscaled(exported):
    """The whole point: the copy is small. DroneVehicle starts at 840x712."""
    image = next((exported / "dronevehicle" / "rgb").glob("*.jpg"))
    with Image.open(image) as handle:
        assert min(handle.size) == EXPORT_SIZE


def test_border_crop_is_baked_in_not_repeated(exported):
    """Geometry is applied once, at export time.

    840x712 minus the 100px border is 640x512, an aspect ratio of 1.25. If the
    exporter had skipped the crop the ratio would be 840/712 = 1.18.
    """
    image = next((exported / "dronevehicle" / "rgb").glob("*.jpg"))
    with Image.open(image) as handle:
        width, height = handle.size
    assert width / height == pytest.approx(640 / 512, abs=0.02)


def test_rgb_and_thermal_are_the_same_size(exported):
    """Alignment resampling moves to export time, out of the training loop."""
    stem = next((exported / "llvip" / "rgb").glob("*.jpg")).name
    with Image.open(exported / "llvip" / "rgb" / stem) as rgb:
        with Image.open(exported / "llvip" / "thermal" / stem) as thermal:
            assert rgb.size == thermal.size


# ----------------------------------------------------------------- round trip


def test_source_dataset_names_survive(exported):
    """Without dataset_from_regex everything would collapse to one name,
    silently breaking sample_weights and the per-dataset metric table."""
    records = build_adapter(subset_spec("subset.yaml", exported)).discover()
    assert {r.dataset for r in records} == {"dronevehicle", "llvip"}


def test_viewpoints_survive(exported):
    records = build_adapter(subset_spec("subset.yaml", exported)).discover()
    viewpoints = {r.dataset: r.viewpoint for r in records}
    assert viewpoints == {"dronevehicle": "uav_high", "llvip": "elevated"}


def test_pair_counts_match_the_originals(exported):
    records = build_adapter(subset_spec("subset.yaml", exported)).discover()
    counts = {}
    for record in records:
        counts[record.dataset] = counts.get(record.dataset, 0) + 1
    assert counts == {"dronevehicle": 8, "llvip": 6}


def test_scene_groups_survive(exported):
    """The leakage guarantee has to cross the export, not restart at random."""
    records = build_adapter(subset_spec("subset.yaml", exported)).discover()
    llvip_groups = {r.group_key for r in records if r.dataset == "llvip"}
    assert llvip_groups == {"01", "19"}


def test_reference_data_is_recovered_and_unpaired(exported):
    records = build_adapter(subset_spec("subset_reference.yaml", exported)).discover()
    assert len(records) == 6
    assert {r.dataset for r in records} == {"hituav"}
    assert all(r.rgb_path == "" for r in records)


def test_reference_never_enters_training_after_round_trip(exported):
    paired = build_adapter(subset_spec("subset.yaml", exported)).discover()
    reference = build_adapter(subset_spec("subset_reference.yaml", exported)).discover()
    frame = assign_splits(records_to_frame(paired + reference))

    hituav = frame[frame["dataset"] == "hituav"]
    assert (hituav["split"] == REFERENCE_SPLIT).all()
    assert REFERENCE_SPLIT not in set(frame[frame["dataset"] != "hituav"]["split"])


def test_subset_and_reference_specs_do_not_overlap(exported):
    """The two globs must partition the export, or samples get counted twice."""
    paired = build_adapter(subset_spec("subset.yaml", exported)).discover()
    reference = build_adapter(subset_spec("subset_reference.yaml", exported)).discover()

    paired_files = {r.thermal_path for r in paired}
    reference_files = {r.thermal_path for r in reference}
    assert not (paired_files & reference_files)
