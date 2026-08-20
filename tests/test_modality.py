"""Modality detection: the piece every adapter depends on."""

from __future__ import annotations

import pytest

from rgb2thermal.data.adapters.modality import PLACEHOLDER, RGB, THERMAL, classify_component


@pytest.mark.parametrize(
    "component,expected",
    [
        # Exact folder names.
        ("visible", RGB),
        ("infrared", THERMAL),
        ("RGB", RGB),
        ("thermal", THERMAL),
        ("ir", THERMAL),
        # DroneVehicle: the token is a suffix, and `imgr` must beat `img`.
        ("trainimg", RGB),
        ("trainimgr", THERMAL),
        ("valimgr", THERMAL),
        # FLIR: strong token wins over the weak `img` that also matches.
        ("images_rgb_train", RGB),
        ("images_thermal_train", THERMAL),
        ("video_thermal_test", THERMAL),
        # Nothing to see here.
        ("train", None),
        ("data", None),
        ("annotations", None),
    ],
)
def test_classify_component(component, expected):
    modality, _ = classify_component(component)
    assert modality == expected


@pytest.mark.parametrize("component", ["first", "stairs", "director", "wireframe"])
def test_weak_tokens_do_not_fire_mid_word(component):
    """`ir` buried inside a longer word is a coincidence, not a modality."""
    modality, _ = classify_component(component)
    assert modality is None


def test_known_limitation_trailing_weak_token():
    """A component *ending* in a weak token is accepted, and that is deliberate.

    `trainimg` (real, RGB) and `chair` (accidental) are lexically identical
    cases: a weak token at the end of a longer word. Rejecting one rejects the
    other, and DroneVehicle needs `trainimg` to work.

    The cost is bounded by design: a misclassified file simply never finds a
    partner and is dropped from the manifest, so a stray folder cannot corrupt
    a pair -- see test_adapters.py::test_unmatched_files_are_dropped.
    """
    assert classify_component("chair")[0] == THERMAL
    assert classify_component("trainimg")[0] == RGB


def test_token_override_resolves_a_false_positive():
    """The documented escape hatch: narrow the tokens in the dataset YAML."""
    modality, _ = classify_component(
        "chair", thermal_tokens=["infrared", "thermal"], rgb_tokens=["rgb"], weak_thermal=[]
    )
    assert modality is None


def test_pairs_collapse_to_the_same_key():
    """The whole pairing scheme rests on this: differing only in modality
    means normalising to an identical string."""
    _, rgb_key = classify_component("trainimg")
    _, thermal_key = classify_component("trainimgr")
    assert rgb_key == thermal_key == f"train{PLACEHOLDER}"

    _, rgb_key = classify_component("images_rgb_train")
    _, thermal_key = classify_component("images_thermal_train")
    assert rgb_key == thermal_key == f"images_{PLACEHOLDER}_train"


def test_custom_tokens_override_defaults():
    """A mirror with unusual names is a config change, not a code change."""
    modality, _ = classify_component(
        "warmcam", thermal_tokens=["warmcam"], rgb_tokens=["coldcam"]
    )
    assert modality == THERMAL
