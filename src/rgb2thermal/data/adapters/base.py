"""Adapter base class: turns a dataset directory into a list of RGB/thermal pairs.

Every dataset we merge is described by a :class:`DatasetSpec` (loaded from
``configs/datasets/*.yaml``). :class:`BaseAdapter` implements a discovery
algorithm generic enough that most datasets need *no* Python subclass at all --
only a YAML file. Subclasses exist purely to encode quirks that cannot be
expressed declaratively.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ...utils.paths import expand_roots, iter_images
from .modality import (
    RGB,
    STRONG_RGB_TOKENS,
    STRONG_THERMAL_TOKENS,
    THERMAL,
    WEAK_RGB_TOKENS,
    WEAK_THERMAL_TOKENS,
    classify_component,
)

SPLIT_TOKENS = ("train", "val", "valid", "validation", "test")
NIGHT_HINTS = ("night", "dark", "nighttime")
DAY_HINTS = ("day", "daytime", "morning", "afternoon")


@dataclass
class PairRecord:
    """One aligned RGB/thermal sample, normalised across all datasets."""

    rgb_path: str
    thermal_path: str
    dataset: str
    viewpoint: str = "unknown"
    time_of_day: str = "unknown"
    group_key: str = ""
    split_hint: str = "unknown"
    split: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSpec:
    """Declarative description of one source dataset."""

    name: str
    search_roots: list[str] = field(default_factory=list)
    adapter: str = "generic"
    enabled: bool = True
    paired: bool = True
    viewpoint: str = "unknown"
    time_of_day: str = "unknown"
    sample_weight: float = 1.0

    # Discovery controls
    include_globs: list[str] = field(default_factory=list)
    exclude_globs: list[str] = field(default_factory=list)
    thermal_tokens: list[str] = field(default_factory=list)
    rgb_tokens: list[str] = field(default_factory=list)

    #: Bypass modality detection entirely and label every file as this modality.
    #: Required for thermal-only sources such as HIT-UAV, whose folders
    #: (``normal/images/train``) carry no modality token at all.
    force_modality: Optional[str] = None

    #: Applied to the filename stem to derive the pairing key. Needed when the
    #: two modalities do not share a filename (FLIR appends a per-modality hash).
    key_regex: Optional[str] = None
    #: Applied to the POSIX relative path to derive a scene/video id for
    #: leakage-free splitting.
    group_regex: Optional[str] = None

    #: Applied to the POSIX path to recover the *source* dataset name, which
    #: then overrides `name` on each record. Exists for the exported subset
    #: (scripts/export_subset.py): one folder holds samples from several
    #: original datasets, and collapsing them all to one name would break both
    #: the per-dataset sampling weights and the per-dataset metric breakdown.
    dataset_from_regex: Optional[str] = None

    #: Optional per-source-dataset viewpoint labels, used together with
    #: `dataset_from_regex` so the subset keeps a truthful viewpoint summary.
    viewpoint_map: dict[str, str] = field(default_factory=dict)
    #: Fallback grouping: bucket this many consecutive frames together.
    group_bucket: int = 100

    #: Per-dataset pixel preprocessing, consumed by data/normalize.py.
    preprocess: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown key(s) in dataset spec {data.get('name', '?')!r}: "
                f"{sorted(unknown)}. Known keys: {sorted(known)}"
            )
        return cls(**data)


class BaseAdapter:
    """Discovers RGB/thermal pairs for one dataset.

    The default :meth:`discover` implementation:

    1. expands ``search_roots`` and walks every image below them;
    2. classifies each file as RGB or thermal from its path components;
    3. rewrites the modality token to a placeholder to build a *pairing key*;
    4. emits a record wherever one key collected both modalities.

    Files that never find a partner are silently dropped -- that is the intended
    failure mode, since it degrades gracefully when a mirror ships extra
    unpaired imagery.
    """

    def __init__(self, spec: DatasetSpec):
        self.spec = spec

    # ------------------------------------------------------------------ setup

    @property
    def thermal_tokens(self) -> Sequence[str]:
        return self.spec.thermal_tokens or STRONG_THERMAL_TOKENS

    @property
    def rgb_tokens(self) -> Sequence[str]:
        return self.spec.rgb_tokens or STRONG_RGB_TOKENS

    def roots(self) -> list[Path]:
        return expand_roots(self.spec.search_roots)

    def is_available(self) -> bool:
        """True when at least one search root exists on this machine."""
        return bool(self.roots())

    # -------------------------------------------------------------- discovery

    def discover(self) -> list[PairRecord]:
        roots = self.roots()
        if not roots:
            return []

        rgb_by_key: dict[str, Path] = {}
        thermal_by_key: dict[str, Path] = {}

        for root in roots:
            for image in iter_images(root):
                rel = image.relative_to(root)
                if not self._keep(rel):
                    continue
                modality, key = self.pairing_key(rel)
                if modality is None:
                    continue
                bucket = rgb_by_key if modality == RGB else thermal_by_key
                # First mirror wins; a second mount of the same data is a
                # duplicate, not a new sample.
                bucket.setdefault(key, image)

        if not self.spec.paired:
            return self._unpaired_records(thermal_by_key)

        return self._paired_records(rgb_by_key, thermal_by_key)

    def _paired_records(
        self, rgb_by_key: dict[str, Path], thermal_by_key: dict[str, Path]
    ) -> list[PairRecord]:
        records: list[PairRecord] = []
        for key in sorted(rgb_by_key.keys() & thermal_by_key.keys()):
            rgb_path = rgb_by_key[key]
            thermal_path = thermal_by_key[key]
            dataset_name = self.dataset_name_for(rgb_path)
            records.append(
                PairRecord(
                    rgb_path=str(rgb_path),
                    thermal_path=str(thermal_path),
                    dataset=dataset_name,
                    viewpoint=self.viewpoint_for(dataset_name),
                    time_of_day=self.time_of_day_for(rgb_path),
                    split_hint=self.split_hint_for(rgb_path),
                )
            )
        self._assign_group_keys(records)
        return records

    def _unpaired_records(self, thermal_by_key: dict[str, Path]) -> list[PairRecord]:
        """Thermal-only datasets (HIT-UAV) become records with an empty rgb_path.

        They cannot train a paired model, but they are a valid *reference*
        distribution for unpaired FID against generated UAV thermal.
        """
        records = [
            PairRecord(
                rgb_path="",
                thermal_path=str(path),
                dataset=self.dataset_name_for(path),
                viewpoint=self.viewpoint_for(self.dataset_name_for(path)),
                time_of_day=self.time_of_day_for(path),
                split_hint=self.split_hint_for(path),
            )
            for _, path in sorted(thermal_by_key.items())
        ]
        self._assign_group_keys(records)
        return records

    # ----------------------------------------------------------------- keying

    def pairing_key(self, rel: Path) -> tuple[Optional[str], str]:
        """Map a dataset-relative path to ``(modality, pairing_key)``.

        Directory components decide the modality, because they are far more
        reliable than filenames -- a random hash in a filename can contain
        ``ir`` by chance, a folder called ``infrared`` cannot be accidental.
        The stem is still normalised so layouts like ``0001_rgb.jpg`` /
        ``0001_ir.jpg`` pair correctly.
        """
        dir_parts = list(rel.parent.parts)
        if dir_parts == ["."]:
            dir_parts = []

        modality: Optional[str] = None
        normalized_dirs: list[str] = []
        for part in dir_parts:
            part_modality, normalized = self._classify(part)
            normalized_dirs.append(normalized)
            if part_modality is not None:
                # Deepest classified directory wins.
                modality = part_modality

        stem_modality, normalized_stem = self._classify(rel.stem)
        if modality is None:
            modality = stem_modality
        if self.spec.force_modality is not None:
            modality = self.spec.force_modality

        if self.spec.key_regex:
            match = re.search(self.spec.key_regex, rel.stem)
            if match is None:
                return None, ""
            # Group 1 if the pattern defines one, else the whole match.
            normalized_stem = match.group(1) if match.groups() else match.group(0)

        key = "/".join([*normalized_dirs, normalized_stem])
        return modality, key

    def _classify(self, component: str) -> tuple[Optional[str], str]:
        return classify_component(
            component,
            thermal_tokens=self.thermal_tokens,
            rgb_tokens=self.rgb_tokens,
            weak_thermal=WEAK_THERMAL_TOKENS,
            weak_rgb=WEAK_RGB_TOKENS,
        )

    def _keep(self, rel: Path) -> bool:
        posix = rel.as_posix()
        if self.spec.include_globs and not any(
            fnmatch.fnmatch(posix, pattern) for pattern in self.spec.include_globs
        ):
            return False
        return not any(fnmatch.fnmatch(posix, pattern) for pattern in self.spec.exclude_globs)

    # --------------------------------------------------------------- metadata

    def _assign_group_keys(self, records: list[PairRecord]) -> None:
        """Attach a scene/video id used to split without frame-level leakage.

        These datasets are sampled video, so neighbouring frames are nearly
        identical. Splitting per image would place near-duplicates in both train
        and val and inflate every metric. When no explicit scene id can be
        parsed we fall back to bucketing runs of consecutive frames, which
        preserves that guarantee approximately but cheaply.
        """
        bucket_size = max(1, self.spec.group_bucket)
        for index, record in enumerate(records):
            source = record.rgb_path or record.thermal_path
            record.group_key = self.group_key_for(Path(source), index, bucket_size)

    def group_key_for(self, path: Path, index: int, bucket_size: int) -> str:
        if self.spec.group_regex:
            match = re.search(self.spec.group_regex, path.as_posix())
            if match is not None:
                return match.group(1) if match.groups() else match.group(0)
        return f"{path.parent.name}#{index // bucket_size:05d}"

    def dataset_name_for(self, path: Path) -> str:
        """Source dataset for one record; usually just the spec name."""
        if self.spec.dataset_from_regex:
            match = re.search(self.spec.dataset_from_regex, path.as_posix())
            if match is not None:
                return match.group(1) if match.groups() else match.group(0)
        return self.spec.name

    def viewpoint_for(self, dataset_name: str) -> str:
        return self.spec.viewpoint_map.get(dataset_name, self.spec.viewpoint)

    def split_hint_for(self, path: Path) -> str:
        lowered = path.as_posix().lower()
        for token in SPLIT_TOKENS:
            if re.search(rf"(?<![a-z]){token}(?![a-z])", lowered):
                return "val" if token.startswith("val") else token
        return "unknown"

    def time_of_day_for(self, path: Path) -> str:
        if self.spec.time_of_day != "unknown":
            return self.spec.time_of_day
        lowered = path.as_posix().lower()
        if any(hint in lowered for hint in NIGHT_HINTS):
            return "night"
        if any(hint in lowered for hint in DAY_HINTS):
            return "day"
        return "unknown"


class GenericPairedAdapter(BaseAdapter):
    """Pure-YAML adapter.

    This is the escape hatch for the one thing we cannot verify from outside
    Kaggle: the exact folder layout of a community mirror. If a mirror is
    organised unexpectedly, adding a YAML file that targets it is enough -- no
    Python change required.
    """
