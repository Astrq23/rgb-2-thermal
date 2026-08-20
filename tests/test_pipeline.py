"""End-to-end: real config files, real Dataset, real sampler, real training step.

These are the tests that would have caught a broken YAML or a shape mismatch
before burning a Kaggle GPU session on it.
"""

from __future__ import annotations

import pytest
import torch

from rgb2thermal.config import (
    Config,
    apply_overrides,
    deep_merge,
    load_config,
    load_dataset_specs,
    parse_override,
)
from rgb2thermal.data.adapters.base import DatasetSpec
from rgb2thermal.data.dataset import (
    PairedThermalDataset,
    ThermalReferenceDataset,
    build_dataloaders,
    trainable,
)
from rgb2thermal.data.manifest import build_manifest
from rgb2thermal.data.sampler import compute_sample_weights


def spec(name: str, root, **kwargs) -> DatasetSpec:
    kwargs.setdefault("exclude_globs", ["**/*label*/**"])
    return DatasetSpec(name=name, search_roots=[str(root)], **kwargs)


# --------------------------------------------------------------- real configs


def test_shipped_configs_all_load():
    """Every config in the repo must parse -- including through `extends`."""
    for name in ("configs/base.yaml", "configs/pix2pix_uav.yaml", "configs/smoke.yaml"):
        cfg = load_config(name)
        assert isinstance(cfg, Config)
        assert cfg.data.image_size > 0


def test_extends_inherits_then_overrides():
    base = load_config("configs/base.yaml")
    smoke = load_config("configs/smoke.yaml")

    assert smoke.train.max_steps == 50           # smoke's own value
    assert smoke.train.epochs == 1
    assert smoke.model.ngf == base.model.ngf     # inherited through pix2pix_uav
    assert smoke.data.sample_weights             # inherited from pix2pix_uav


def test_shipped_dataset_specs_all_load():
    specs = load_dataset_specs(["configs/datasets/*.yaml"])
    names = {s.name for s in specs}
    assert names == {"dronevehicle", "llvip", "flir_v2", "hituav"}


def test_hituav_is_configured_as_reference_only():
    """A regression guard: it must never become trainable by accident."""
    specs = {s.name: s for s in load_dataset_specs(["configs/datasets/*.yaml"])}
    hituav = specs["hituav"]
    assert hituav.paired is False
    assert hituav.force_modality == "thermal"
    assert hituav.sample_weight == 0.0


def test_dronevehicle_crops_the_documented_border():
    specs = {s.name: s for s in load_dataset_specs(["configs/datasets/*.yaml"])}
    assert specs["dronevehicle"].preprocess["rgb"]["crop_border"] == 100
    assert specs["dronevehicle"].preprocess["thermal"]["crop_border"] == 100


def test_sample_weights_favour_uav_data():
    cfg = load_config("configs/pix2pix_uav.yaml")
    weights = cfg.data.sample_weights
    assert weights["dronevehicle"] > weights["llvip"] > weights["flir_v2"]


def test_cli_overrides():
    assert parse_override("train.epochs=5") == (["train", "epochs"], 5)
    assert parse_override("data.hflip=false") == (["data", "hflip"], False)
    merged = apply_overrides({"train": {"epochs": 40, "lr": 1}}, ["train.epochs=2"])
    assert merged["train"] == {"epochs": 2, "lr": 1}


def test_override_reaches_the_loaded_config():
    cfg = load_config("configs/smoke.yaml", overrides=["train.batch_size=2", "data.image_size=64"])
    assert cfg.train.batch_size == 2
    assert cfg.data.image_size == 64


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"y": 2}})
    assert base == {"a": {"x": 1}}


def test_typo_in_config_is_rejected_loudly(tmp_path):
    """Silently ignoring `epocs: 100` would waste a whole GPU session."""
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  epocs: 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(path)


# -------------------------------------------------------------------- sampler


def test_weights_realise_the_requested_dataset_mix():
    """The point of weighting: proportions come from config, not from file counts."""
    datasets = ["big"] * 900 + ["small"] * 100
    weights = compute_sample_weights(datasets, {"big": 0.5, "small": 0.5})

    big_mass = sum(w for w, d in zip(weights, datasets) if d == "big")
    small_mass = sum(w for w, d in zip(weights, datasets) if d == "small")
    assert big_mass == pytest.approx(small_mass)


def test_zero_weight_dataset_is_never_drawn():
    datasets = ["keep"] * 10 + ["drop"] * 10
    weights = compute_sample_weights(datasets, {"keep": 1.0, "drop": 0.0})
    assert all(w == 0.0 for w, d in zip(weights, datasets) if d == "drop")


def test_single_dataset_needs_no_weighting():
    assert compute_sample_weights(["only"] * 5, {"only": 1.0}) is None


def test_no_positive_weights_falls_back_to_shuffling():
    assert compute_sample_weights(["a", "b"], {"a": 0.0, "b": 0.0}) is None


# -------------------------------------------------------------------- dataset


@pytest.fixture
def small_manifest(fake_dronevehicle, fake_llvip):
    specs = [spec("dronevehicle", fake_dronevehicle), spec("llvip", fake_llvip)]
    return build_manifest(specs, verbose=False), specs


def test_dataset_yields_correctly_shaped_tensors(small_manifest):
    frame, specs = small_manifest
    cfg = load_config(
        "configs/smoke.yaml", overrides=["data.image_size=64", "data.load_size=72"]
    ).data

    dataset = PairedThermalDataset(trainable(frame), cfg, specs, train=True)
    sample = dataset[0]

    assert sample["rgb"].shape == (3, 64, 64)
    assert sample["thermal"].shape == (1, 64, 64)
    assert sample["rgb"].dtype == torch.float32
    assert -1.0 <= float(sample["thermal"].min()) and float(sample["thermal"].max()) <= 1.0


def test_eval_mode_is_deterministic(small_manifest):
    """Metrics are meaningless if the crop moves between runs."""
    frame, specs = small_manifest
    cfg = load_config("configs/smoke.yaml", overrides=["data.image_size=64"]).data

    dataset = PairedThermalDataset(trainable(frame), cfg, specs, train=False)
    assert torch.equal(dataset[0]["rgb"], dataset[0]["rgb"])
    assert torch.equal(dataset[0]["thermal"], dataset[0]["thermal"])


def test_datasets_of_different_native_sizes_are_batched_together(small_manifest):
    """DroneVehicle is 840x712, LLVIP 96x80 -- both must emerge identical."""
    frame, specs = small_manifest
    cfg = load_config("configs/smoke.yaml", overrides=["data.image_size=64"]).data
    usable = trainable(frame)

    shapes = set()
    for name in ("dronevehicle", "llvip"):
        subset = usable[usable["dataset"] == name]
        dataset = PairedThermalDataset(subset, cfg, specs, train=False)
        shapes.add(tuple(dataset[0]["rgb"].shape))
    assert shapes == {(3, 64, 64)}


def test_domain_ids_are_stable_and_dense(small_manifest):
    frame, specs = small_manifest
    cfg = load_config("configs/smoke.yaml").data
    dataset = PairedThermalDataset(trainable(frame), cfg, specs, train=False)
    assert sorted(dataset.domain_index.values()) == list(range(len(dataset.domain_index)))


def test_thermal_reference_dataset_needs_no_rgb(fake_hituav):
    specs = [spec("hituav", fake_hituav, paired=False, force_modality="thermal")]
    frame = build_manifest(specs, verbose=False)
    cfg = load_config("configs/smoke.yaml", overrides=["data.image_size=32"]).data

    dataset = ThermalReferenceDataset(frame, cfg, specs)
    assert len(dataset) == 6
    assert dataset[0]["thermal"].shape == (1, 32, 32)


def test_build_dataloaders_produces_usable_batches(small_manifest):
    frame, specs = small_manifest
    cfg = load_config(
        "configs/smoke.yaml",
        overrides=["data.image_size=64", "data.num_workers=0", "data.samples_per_epoch=8"],
    ).data

    train_loader, _, domain_index = build_dataloaders(frame, cfg, specs, batch_size=2)
    batch = next(iter(train_loader))

    assert batch["rgb"].shape == (2, 3, 64, 64)
    assert batch["thermal"].shape == (2, 1, 64, 64)
    assert batch["domain"].max() < len(domain_index)


def test_build_dataloaders_explains_an_empty_train_split(fake_llvip):
    specs = [spec("llvip", fake_llvip)]
    frame = build_manifest(specs, verbose=False)
    frame["split"] = "test"
    cfg = load_config("configs/smoke.yaml").data
    with pytest.raises(RuntimeError, match="No training rows"):
        build_dataloaders(frame, cfg, specs, batch_size=2)


# ------------------------------------------------------------------ full step


def test_one_training_step_runs_and_checkpoints(small_manifest, tmp_path):
    """The smoke run, in-process: data -> both nets -> losses -> checkpoint -> resume."""
    from rgb2thermal.engine.trainer import Trainer

    frame, specs = small_manifest
    cfg = load_config(
        "configs/smoke.yaml",
        overrides=[
            "data.image_size=64",
            "data.load_size=64",
            "data.num_workers=0",
            "data.samples_per_epoch=4",
            "model.ngf=8",
            "model.ndf=8",
            "model.n_down=6",
            "train.batch_size=2",
            "train.epochs=1",
            "train.max_steps=2",
            "train.amp=false",
            f"train.output_dir={tmp_path.as_posix()}",
        ],
    )

    train_loader, val_loader, domain_index = build_dataloaders(
        frame, cfg.data, specs, batch_size=cfg.train.batch_size
    )
    trainer = Trainer(cfg, train_loader, val_loader, domain_index, device="cpu")
    summary = trainer.fit()

    assert summary["global_step"] >= 1
    checkpoint = trainer.checkpoint_dir / "last.pt"
    assert checkpoint.exists()
    assert (trainer.output_dir / "metrics.csv").exists()

    # And a fresh trainer can pick the run back up -- the Kaggle 12h story.
    resumed = Trainer(cfg, train_loader, val_loader, domain_index, device="cpu")
    resumed.resume(checkpoint)
    assert resumed.global_step == summary["global_step"]
