"""Adapter registry: maps the ``adapter:`` key in a dataset YAML to a class."""

from __future__ import annotations

from typing import Type

from .base import BaseAdapter, DatasetSpec, GenericPairedAdapter
from .flir_adas_v2 import FlirAdasV2Adapter

ADAPTERS: dict[str, Type[BaseAdapter]] = {
    "generic": GenericPairedAdapter,
    "flir_adas_v2": FlirAdasV2Adapter,
}


def build_adapter(spec: DatasetSpec) -> BaseAdapter:
    try:
        adapter_cls = ADAPTERS[spec.adapter]
    except KeyError as exc:
        raise KeyError(
            f"Unknown adapter {spec.adapter!r} for dataset {spec.name!r}. "
            f"Available: {sorted(ADAPTERS)}"
        ) from exc
    return adapter_cls(spec)


def register_adapter(name: str, adapter_cls: Type[BaseAdapter]) -> None:
    """Register a custom adapter (useful from a notebook cell during debugging)."""
    ADAPTERS[name] = adapter_cls


__all__ = ["ADAPTERS", "build_adapter", "register_adapter"]
