"""Manifest assembly, and the split guarantee everything else relies on."""

from __future__ import annotations

import pandas as pd
import pytest

from rgb2thermal.data.adapters.base import DatasetSpec, PairRecord
from rgb2thermal.data.manifest import (
    MANIFEST_COLUMNS,
    REFERENCE_SPLIT,
    assign_splits,
    build_manifest,
    discover_records,
    load_manifest,
    records_to_frame,
    save_manifest,
    summarize,
)


def spec(name: str, root, **kwargs) -> DatasetSpec:
    kwargs.setdefault("exclude_globs", ["**/*label*/**"])
    return DatasetSpec(name=name, search_roots=[str(root)], **kwargs)


def make_frame(rows) -> pd.DataFrame:
    return records_to_frame([PairRecord(**row) for row in rows])


# ------------------------------------------------------------------ assembly


def test_merges_datasets_into_one_schema(fake_dronevehicle, fake_llvip, fake_flir):
    specs = [
        spec("dronevehicle", fake_dronevehicle),
        spec("llvip", fake_llvip),
        spec("flir_v2", fake_flir, adapter="flir_adas_v2"),
    ]
    frame = build_manifest(specs, verbose=False)

    assert list(frame.columns) == MANIFEST_COLUMNS
    assert set(frame["dataset"]) == {"dronevehicle", "llvip", "flir_v2"}
    assert len(frame) == 8 + 6 + 6


def test_missing_datasets_are_skipped_not_fatal(fake_llvip, tmp_path):
    """Attaching only some of the four Kaggle datasets is a normal case."""
    specs = [spec("llvip", fake_llvip), spec("ghost", tmp_path / "nope")]
    frame = build_manifest(specs, verbose=False)
    assert set(frame["dataset"]) == {"llvip"}


def test_disabled_dataset_is_ignored(fake_llvip):
    specs = [spec("llvip", fake_llvip, enabled=False)]
    assert discover_records(specs, verbose=False) == []


def test_empty_manifest_has_the_right_columns():
    frame = records_to_frame([])
    assert list(frame.columns) == MANIFEST_COLUMNS
    assert frame.empty


# -------------------------------------------------------------------- splits


def _grouped_rows(n_groups=20, per_group=5):
    return [
        {
            "rgb_path": f"/rgb/{group}_{index}.jpg",
            "thermal_path": f"/ir/{group}_{index}.jpg",
            "dataset": "demo",
            "group_key": f"g{group:03d}",
            "split_hint": "unknown",
        }
        for group in range(n_groups)
        for index in range(per_group)
    ]


def test_no_group_spans_two_splits():
    """The core anti-leakage guarantee: adjacent video frames stay together."""
    frame = assign_splits(make_frame(_grouped_rows()), seed=7)
    spans = frame.groupby("group_key")["split"].nunique()
    assert (spans == 1).all()


def test_all_three_splits_are_populated():
    frame = assign_splits(make_frame(_grouped_rows(n_groups=60)), seed=7)
    assert set(frame["split"]) == {"train", "val", "test"}


def test_split_is_stable_across_runs():
    """Hash-based, not `random` -- a resumed Kaggle session must not reshuffle."""
    rows = _grouped_rows()
    first = assign_splits(make_frame(rows), seed=7)["split"].tolist()
    second = assign_splits(make_frame(rows), seed=7)["split"].tolist()
    assert first == second


def test_seed_changes_the_split():
    rows = _grouped_rows()
    assert (
        assign_splits(make_frame(rows), seed=1)["split"].tolist()
        != assign_splits(make_frame(rows), seed=999)["split"].tolist()
    )


def test_official_split_hints_are_honoured():
    """A dataset's own train/test split beats our heuristic grouping."""
    rows = [
        {
            "rgb_path": f"/rgb/{split}_{i}.jpg",
            "thermal_path": f"/ir/{split}_{i}.jpg",
            "dataset": "llvip",
            "group_key": f"{split}-g{i // 5}",
            "split_hint": split,
        }
        for split in ("train", "test")
        for i in range(40)
    ]
    frame = assign_splits(make_frame(rows), respect_split_hint=True)

    # Every officially-test row stays in test.
    assert (frame[frame["split_hint"] == "test"]["split"] == "test").all()
    # A val set gets carved out of train, since the dataset ships none.
    assert (frame["split"] == "val").any()
    assert frame.groupby("group_key")["split"].nunique().max() == 1


def test_hints_ignored_when_they_barely_cover_the_data():
    """One stray hint must not hijack the split for a whole dataset.

    Asserted by equivalence with the hint-free path rather than by which splits
    happen to appear -- with few groups the hash split can legitimately leave a
    split empty, and that would make the test flaky for the wrong reason.
    """
    rows = _grouped_rows(n_groups=30)
    rows[0]["split_hint"] = "train"  # a single hint proves nothing

    with_hints = assign_splits(make_frame(rows), respect_split_hint=True, seed=3)
    without_hints = assign_splits(make_frame(rows), respect_split_hint=False, seed=3)
    assert with_hints["split"].tolist() == without_hints["split"].tolist()


def test_thermal_only_dataset_goes_to_reference():
    """HIT-UAV must never leak into training."""
    rows = [
        {"rgb_path": "", "thermal_path": f"/ir/{i}.jpg", "dataset": "hituav", "group_key": "g0"}
        for i in range(5)
    ]
    frame = assign_splits(make_frame(rows))
    assert (frame["split"] == REFERENCE_SPLIT).all()


def test_reference_and_paired_coexist(fake_dronevehicle, fake_hituav):
    specs = [
        spec("dronevehicle", fake_dronevehicle),
        spec("hituav", fake_hituav, paired=False, force_modality="thermal"),
    ]
    frame = build_manifest(specs, verbose=False)

    reference = frame[frame["dataset"] == "hituav"]
    paired = frame[frame["dataset"] == "dronevehicle"]
    assert (reference["split"] == REFERENCE_SPLIT).all()
    assert REFERENCE_SPLIT not in set(paired["split"])


# ------------------------------------------------------------------- round-trip


def test_save_load_roundtrip(tmp_path, fake_llvip):
    frame = build_manifest([spec("llvip", fake_llvip)], verbose=False)
    path = save_manifest(frame, tmp_path / "manifest.csv")
    reloaded = load_manifest(path)

    assert len(reloaded) == len(frame)
    assert list(reloaded.columns) == MANIFEST_COLUMNS


def test_load_rejects_a_malformed_manifest(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(path)


def test_empty_rgb_path_survives_the_csv_roundtrip(tmp_path):
    """Reference rows carry an empty rgb_path; pandas must not turn it into NaN."""
    rows = [{"rgb_path": "", "thermal_path": "/ir/0.jpg", "dataset": "hituav", "group_key": "g"}]
    path = save_manifest(assign_splits(make_frame(rows)), tmp_path / "m.csv")
    assert load_manifest(path)["rgb_path"].iloc[0] == ""


# ---------------------------------------------------------------------- report


def test_summary_reports_counts_and_no_leak_warning():
    text = summarize(assign_splits(make_frame(_grouped_rows()), seed=7))
    assert "By dataset x split" in text
    assert "WARNING" not in text


def test_summary_flags_a_leaking_split():
    """The guard has to actually fire, or it is decoration."""
    frame = make_frame(_grouped_rows(n_groups=2))
    frame["split"] = ["train", "val"] * (len(frame) // 2)
    assert "WARNING" in summarize(frame)


def test_summary_explains_an_empty_manifest():
    assert "inspect_datasets" in summarize(records_to_frame([]))
