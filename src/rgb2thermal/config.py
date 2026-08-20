"""Configuration loading: YAML files + `key.path=value` CLI overrides.

Configs compose through an ``extends:`` key so ``smoke.yaml`` can be a five-line
diff against the real training config instead of a copy that drifts out of sync.
"""

from __future__ import annotations

import glob as _glob
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from .data.adapters.base import DatasetSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------- schema


@dataclass
class DataConfig:
    #: Globs pointing at configs/datasets/*.yaml
    dataset_specs: list[str] = field(default_factory=lambda: ["configs/datasets/*.yaml"])
    manifest: str = "manifest.csv"

    image_size: int = 256
    #: Images are resized to this before the random crop down to image_size.
    #: Standard pix2pix jitter; set equal to image_size to disable.
    load_size: int = 286
    #: "crop" preserves aspect ratio (resize short side, then crop).
    #: "stretch" squashes straight to image_size x image_size.
    resize_mode: str = "crop"
    hflip: bool = True

    thermal_norm: str = "percentile"
    low_percentile: float = 1.0
    high_percentile: float = 99.0

    split_ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    respect_split_hint: bool = True
    split_seed: int = 42

    #: Relative sampling probability per dataset. Keys not present default to
    #: the dataset's own `sample_weight`. This is how UAV data is kept dominant.
    sample_weights: dict[str, float] = field(default_factory=dict)
    #: Samples drawn per training epoch when weighted sampling is on.
    #: 0 means "one pass over the training set".
    samples_per_epoch: int = 0

    num_workers: int = 2
    limit_train: int = 0
    limit_val: int = 0


@dataclass
class ModelConfig:
    name: str = "pix2pix"
    ngf: int = 64
    ndf: int = 64
    #: U-Net depth. 8 gives the classic 256x256 generator.
    n_down: int = 8
    norm: str = "batch"
    dropout: float = 0.5
    n_layers_d: int = 3
    #: FiLM-style conditioning on which source dataset a sample came from.
    #: Off by default -- turn it on to let one model mimic several sensors.
    use_domain_embedding: bool = False
    domain_dim: int = 32


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    gan_mode: str = "lsgan"  # "lsgan" | "vanilla"
    amp: bool = True
    #: Epoch at which the LR starts decaying linearly to zero. 0 = half of epochs.
    lr_decay_start: int = 0
    seed: int = 42
    output_dir: str = ""
    resume: str = ""
    log_interval: int = 50
    sample_interval: int = 500
    #: Cap steps per epoch. Useful for smoke runs and for fitting Kaggle's 12h wall.
    max_steps_per_epoch: int = 0
    #: Stop the whole run after this many optimiser steps. 0 = unlimited.
    max_steps: int = 0
    #: Wall-clock budget in hours. The run stops cleanly at the next checkpoint
    #: boundary once this is exceeded. Set it below Kaggle's 12h session ceiling
    #: so the session ends on our terms, with a saved checkpoint, instead of
    #: being killed mid-epoch. 0 = no limit.
    max_hours: float = 0.0


@dataclass
class EvalConfig:
    batch_size: int = 16
    split: str = "test"
    metrics: list[str] = field(default_factory=lambda: ["psnr", "ssim", "lpips", "fid"])
    #: Thermal-only dataset used as the real distribution for unpaired FID.
    fid_reference_dataset: str = "hituav"
    max_samples: int = 0
    num_visuals: int = 8


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    name: str = "run"

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


# -------------------------------------------------------------------- loading


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(path: Path | str) -> Path:
    """Resolve a config path relative to the CWD, then to the repo root."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    from_root = REPO_ROOT / candidate
    if from_root.exists():
        return from_root
    raise FileNotFoundError(f"Config not found: {path} (also tried {from_root})")


def _load_yaml_with_extends(path: Path | str, _seen: Optional[set[Path]] = None) -> dict[str, Any]:
    resolved = _resolve(path).resolve()
    _seen = _seen or set()
    if resolved in _seen:
        raise ValueError(f"Circular `extends` chain at {resolved}")
    _seen.add(resolved)

    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    parent_path = (resolved.parent / parent_ref) if not Path(parent_ref).is_absolute() else Path(parent_ref)
    if not parent_path.exists():
        parent_path = Path(parent_ref)
    parent = _load_yaml_with_extends(parent_path, _seen)
    return deep_merge(parent, data)


def _build_section(section_cls: type, data: dict[str, Any], label: str) -> Any:
    known = {f.name for f in fields(section_cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in config section '{label}': {sorted(unknown)}. "
            f"Known keys: {sorted(known)}"
        )
    return section_cls(**data)


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse ``train.epochs=5`` into ``(["train", "epochs"], 5)``."""
    if "=" not in text:
        raise ValueError(f"Override must look like key.path=value, got {text!r}")
    key, _, raw = text.partition("=")
    return key.strip().split("."), yaml.safe_load(raw)


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    data = dict(data)
    for override in overrides:
        keys, value = parse_override(override)
        cursor = data
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override {override!r} targets a non-mapping node")
        cursor[keys[-1]] = value
    return data


def load_config(path: Path | str, overrides: Sequence[str] = ()) -> Config:
    raw = apply_overrides(_load_yaml_with_extends(path), overrides)
    return Config(
        name=raw.get("name", Path(path).stem),
        data=_build_section(DataConfig, raw.get("data", {}), "data"),
        model=_build_section(ModelConfig, raw.get("model", {}), "model"),
        train=_build_section(TrainConfig, raw.get("train", {}), "train"),
        eval=_build_section(EvalConfig, raw.get("eval", {}), "eval"),
    )


def load_dataset_specs(patterns: Sequence[str]) -> list[DatasetSpec]:
    """Load every ``configs/datasets/*.yaml`` matched by ``patterns``.

    Sorted by filename so manifest ordering (and therefore the split) stays
    reproducible regardless of filesystem iteration order.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = _glob.glob(str(pattern)) or _glob.glob(str(REPO_ROOT / pattern))
        for match in sorted(matches):
            resolved = Path(match).resolve()
            if resolved.suffix.lower() in {".yaml", ".yml"} and resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

    if not paths:
        raise FileNotFoundError(
            f"No dataset spec YAML matched {list(patterns)}. "
            f"Expected files under {CONFIG_DIR / 'datasets'}"
        )

    specs: list[DatasetSpec] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        data.setdefault("name", path.stem)
        specs.append(DatasetSpec.from_dict(data))
    return specs
