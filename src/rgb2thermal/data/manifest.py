"""Building the unified manifest -- one table describing every sample we train on.

The manifest is the single contract between "messy Kaggle folders" and
everything downstream. Once it exists, the Dataset, the sampler, the trainer and
the evaluator all speak the same schema and no longer care that DroneVehicle,
LLVIP and FLIR are organised nothing alike.

Columns: rgb_path, thermal_path, dataset, viewpoint, time_of_day, group_key,
split_hint, split.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from .adapters.base import DatasetSpec, PairRecord
from .adapters.registry import build_adapter

MANIFEST_COLUMNS = [
    "rgb_path",
    "thermal_path",
    "dataset",
    "viewpoint",
    "time_of_day",
    "group_key",
    "split_hint",
    "split",
]

#: Split assigned to thermal-only datasets. They are never trained on; they are
#: a reference distribution for unpaired FID.
REFERENCE_SPLIT = "reference"


def discover_records(specs: Sequence[DatasetSpec], verbose: bool = True) -> list[PairRecord]:
    """Run every enabled adapter, skipping datasets that are not mounted.

    A missing dataset is a normal condition (you may attach three of the four on
    Kaggle), so it warns and continues rather than raising.
    """
    records: list[PairRecord] = []
    for spec in specs:
        if not spec.enabled:
            if verbose:
                print(f"[manifest] {spec.name}: disabled in config, skipping")
            continue

        adapter = build_adapter(spec)
        if not adapter.is_available():
            if verbose:
                print(
                    f"[manifest] {spec.name}: NOT FOUND under "
                    f"{spec.search_roots} -- skipping"
                )
            continue

        found = adapter.discover()
        if verbose:
            roots = ", ".join(str(r) for r in adapter.roots())
            if found:
                kind = "pairs" if spec.paired else "thermal-only images"
                print(f"[manifest] {spec.name}: {len(found):,} {kind}  (root: {roots})")
            else:
                print(
                    f"[manifest] {spec.name}: root exists but 0 pairs matched -- "
                    f"the layout differs from the config. Run "
                    f"`python scripts/inspect_datasets.py` and adjust "
                    f"configs/datasets/{spec.name}.yaml  (root: {roots})"
                )
        records.extend(found)
    return records


def records_to_frame(records: Iterable[PairRecord]) -> pd.DataFrame:
    frame = pd.DataFrame([record.to_dict() for record in records])
    if frame.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return frame[MANIFEST_COLUMNS]


def _stable_fraction(*parts: str) -> float:
    """Deterministic float in [0, 1) from strings.

    Python's builtin ``hash`` is salted per process, which would reshuffle the
    split on every run and quietly leak val samples into train across sessions.
    MD5 is not used for security here, only for reproducibility.
    """
    digest = hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest, 16) / float(1 << 128)


def _group_split(
    frame: pd.DataFrame, ratios: tuple[float, float, float], seed: int
) -> pd.Series:
    """Assign splits by ``group_key`` so no scene spans two splits."""
    train_ratio, val_ratio, _ = ratios
    total = sum(ratios)
    train_edge = train_ratio / total
    val_edge = (train_ratio + val_ratio) / total

    def choose(group_key: str) -> str:
        position = _stable_fraction(str(seed), group_key)
        if position < train_edge:
            return "train"
        if position < val_edge:
            return "val"
        return "test"

    return frame["group_key"].map(choose)


def _hint_is_usable(frame: pd.DataFrame) -> bool:
    """Prefer a dataset's official split when it actually covers the data.

    Official test splits were designed to avoid leakage, so they beat our
    heuristic grouping -- but only if they are really present.
    """
    hints = frame["split_hint"]
    known = hints[hints.isin(["train", "val", "test"])]
    return len(known) >= 0.9 * len(frame) and known.nunique() >= 2


def assign_splits(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    respect_split_hint: bool = True,
) -> pd.DataFrame:
    """Assign train/val/test per dataset, never splitting a ``group_key``."""
    if frame.empty:
        return frame

    frame = frame.copy()
    chunks: list[pd.DataFrame] = []

    for dataset_name, subset in frame.groupby("dataset", sort=True):
        subset = subset.copy()

        # Thermal-only sources are reference data, not training data.
        if (subset["rgb_path"] == "").all():
            subset["split"] = REFERENCE_SPLIT
            chunks.append(subset)
            continue

        if respect_split_hint and _hint_is_usable(subset):
            subset["split"] = subset["split_hint"].where(
                subset["split_hint"].isin(["train", "val", "test"]), "train"
            )
            # Many datasets ship only train/test. Carve a val set out of train,
            # still respecting group boundaries.
            if not (subset["split"] == "val").any():
                train_mask = subset["split"] == "train"
                val_share = ratios[1] / (ratios[0] + ratios[1])
                promote = subset.loc[train_mask, "group_key"].map(
                    lambda key: _stable_fraction(str(seed), "val-carve", key) < val_share
                )
                subset.loc[train_mask & promote.reindex(subset.index, fill_value=False), "split"] = "val"
        else:
            subset["split"] = _group_split(subset, ratios, seed)

        chunks.append(subset)

    return pd.concat(chunks).sort_index()


def build_manifest(
    specs: Sequence[DatasetSpec],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    respect_split_hint: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    records = discover_records(specs, verbose=verbose)
    frame = records_to_frame(records)
    return assign_splits(frame, ratios=ratios, seed=seed, respect_split_hint=respect_split_hint)


def save_manifest(frame: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_manifest(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, na_values=[])
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    return frame


def summarize(frame: pd.DataFrame) -> str:
    """Human-readable breakdown, printed by build_manifest.py and the notebook."""
    if frame.empty:
        return (
            "Manifest is EMPTY -- no dataset was found and no pair matched.\n"
            "Run `python scripts/inspect_datasets.py` to see what is actually "
            "mounted under /kaggle/input."
        )

    lines: list[str] = []
    lines.append(f"Total samples: {len(frame):,}")
    lines.append("")
    lines.append("By dataset x split:")
    pivot = pd.crosstab(frame["dataset"], frame["split"], margins=True, margins_name="TOTAL")
    lines.append(pivot.to_string())
    lines.append("")
    lines.append("By viewpoint:")
    lines.append(frame["viewpoint"].value_counts().to_string())
    lines.append("")
    lines.append("By time of day:")
    lines.append(frame["time_of_day"].value_counts().to_string())
    lines.append("")
    lines.append(f"Distinct scene groups: {frame['group_key'].nunique():,}")

    # Leakage guard: the whole point of group-aware splitting.
    lines.append("")
    lines.extend(_leakage_report(frame))
    return "\n".join(lines)


def _leakage_report(frame: pd.DataFrame) -> list[str]:
    """Name the datasets whose scenes straddle a split, and say what to do.

    A group spanning two splits means near-identical video frames sit on both
    sides of the evaluation boundary, which inflates PSNR/SSIM. It has exactly
    one benign cause: honouring a dataset's *official* split when that split
    itself cuts through our scene groups. Reporting per dataset lets you tell
    the two cases apart instead of guessing.
    """
    trainable = frame[frame["split"].isin(["train", "val", "test"])]
    if trainable.empty:
        return ["Split leakage: none (no trainable rows)"]

    offenders: dict[str, int] = {}
    for dataset_name, subset in trainable.groupby("dataset"):
        spans = subset.groupby("group_key")["split"].nunique()
        count = int((spans > 1).sum())
        if count:
            offenders[str(dataset_name)] = count

    if not offenders:
        return ["Split leakage: none -- every scene group sits in exactly one split."]

    detail = ", ".join(f"{name} ({count} group(s))" for name, count in sorted(offenders.items()))
    return [
        f"WARNING: scenes span multiple splits in {detail}",
        "  Expected when a dataset's own train/test split cuts across our scene",
        "  groups and respect_split_hint is on. To force group-disjoint splits",
        "  instead, rebuild with:  --set data.respect_split_hint=false",
    ]
