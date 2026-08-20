"""Adapter for the Teledyne FLIR ADAS Thermal Dataset v2.

FLIR is the one source in this project whose two modalities do **not** share a
filename: both are named ``video-<videoId>-frame-<n>-<perModalityHash>.jpg``,
where only the ``video``/``frame`` portion is common. A plain stem match finds
zero pairs, so we parse ``(video_id, frame_number)`` explicitly. Parsing the
frame number as an *integer* also makes pairing immune to zero-padding
differences between the RGB and thermal exports.

Alignment caveat (important, and why this dataset carries a low sample weight):
the FLIR RGB and thermal cameras have different fields of view and resolutions,
so pairs are only approximately co-registered. ``preprocess.fov_crop`` in
``configs/datasets/flir_adas_v2.yaml`` applies a centre crop that gets them
close, but every pair should be eyeballed via ``scripts/check_alignment.py``
before trusting it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ...utils.paths import iter_images
from .base import BaseAdapter, PairRecord
from .modality import RGB, THERMAL

#: ``video-BzZspxAweF8AnKhWK-frame-000123-Tbvj4CYyCXFsGCrLM``
FRAME_PATTERN = re.compile(r"video-([A-Za-z0-9]+)-frame-(\d+)", re.IGNORECASE)


class FlirAdasV2Adapter(BaseAdapter):
    """Pairs FLIR v2 frames on ``(video_id, frame_number)``."""

    def discover(self) -> list[PairRecord]:
        roots = self.roots()
        if not roots:
            return []

        rgb_by_key: dict[tuple[str, int], Path] = {}
        thermal_by_key: dict[tuple[str, int], Path] = {}

        for root in roots:
            for image in iter_images(root):
                rel = image.relative_to(root)
                if not self._keep(rel):
                    continue
                key = self._frame_key(rel)
                if key is None:
                    continue
                modality = self._modality_from_dirs(rel)
                if modality is None:
                    continue
                bucket = rgb_by_key if modality == RGB else thermal_by_key
                bucket.setdefault(key, image)

        if not rgb_by_key or not thermal_by_key:
            # Mirror does not use the documented naming -- fall back to the
            # generic token/stem matcher rather than returning nothing.
            return super().discover()

        records: list[PairRecord] = []
        for key in sorted(rgb_by_key.keys() & thermal_by_key.keys()):
            rgb_path = rgb_by_key[key]
            records.append(
                PairRecord(
                    rgb_path=str(rgb_path),
                    thermal_path=str(thermal_by_key[key]),
                    dataset=self.spec.name,
                    viewpoint=self.spec.viewpoint,
                    time_of_day=self.time_of_day_for(rgb_path),
                    split_hint=self.split_hint_for(rgb_path),
                    group_key=key[0],  # one group per source video
                )
            )
        return records

    @staticmethod
    def _frame_key(rel: Path) -> Optional[tuple[str, int]]:
        match = FRAME_PATTERN.search(rel.stem)
        if match is None:
            return None
        return match.group(1).lower(), int(match.group(2))

    def _modality_from_dirs(self, rel: Path) -> Optional[str]:
        """Modality from directory names only.

        The trailing hash in a FLIR filename is random and can contain ``ir``,
        so the stem must never be allowed to vote here.
        """
        modality = None
        for part in rel.parent.parts:
            part_modality, _ = self._classify(part)
            if part_modality is not None:
                modality = part_modality
        return modality


__all__ = ["FlirAdasV2Adapter", "FRAME_PATTERN", "RGB", "THERMAL"]
