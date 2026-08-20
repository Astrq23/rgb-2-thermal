"""Weighted mixing of the source datasets.

Concatenating the datasets and shuffling would let sheer size decide the mix:
DroneVehicle is roughly twice LLVIP, so the "blend" would be whatever the
download sizes happened to be. Instead each dataset gets an explicit target
probability, and per-sample weights are set to ``w_d / n_d`` so that the chance
of drawing *some* sample from dataset ``d`` equals ``w_d`` regardless of how many
images ``d`` contributes.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def compute_sample_weights(
    datasets: Sequence[str], weights: dict[str, float]
) -> Optional[list[float]]:
    """Per-sample weights realising the requested per-dataset probabilities.

    Returns ``None`` when weighting would be a no-op (a single dataset, or no
    positive weight configured), so the caller can fall back to plain shuffling.
    """
    counts = Counter(datasets)
    active = {name: weights.get(name, 0.0) for name in counts}
    total = sum(value for value in active.values() if value > 0)

    if total <= 0 or len(counts) <= 1:
        return None

    per_dataset = {
        name: (value / total) / counts[name] if value > 0 else 0.0
        for name, value in active.items()
    }
    return [per_dataset[name] for name in datasets]


def build_weighted_sampler(
    datasets: Sequence[str],
    weights: dict[str, float],
    num_samples: int,
    seed: int = 42,
) -> Optional[WeightedRandomSampler]:
    sample_weights = compute_sample_weights(datasets, weights)
    if sample_weights is None:
        return None
    if not any(w > 0 for w in sample_weights):
        return None

    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=max(1, num_samples),
        replacement=True,
        generator=generator,
    )
