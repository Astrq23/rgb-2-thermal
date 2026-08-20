"""Deciding which modality a file belongs to, from its path alone.

The four Kaggle datasets we merge each name their RGB and thermal folders
differently (``trainimg``/``trainimgr``, ``visible``/``infrared``,
``images_rgb_train``/``images_thermal_train``, ...). Rather than hard-coding
each layout, we classify every path component by looking for modality tokens,
then rewrite the matched span to a placeholder. Two files that differ *only* in
modality therefore collapse to the same key, and pairing becomes a dict lookup.

Tokens come in two tiers. Strong tokens (``thermal``, ``rgb``, ``infrared``,
``imgr``, ...) are unambiguous. Weak tokens (``img``, ``ir``, ``vi``) are short
enough to appear by accident, so they are only consulted when no strong token
matched anywhere in the component. Without that split, ``images_rgb_train``
would match the weak ``img`` at position 0 and never reach the ``rgb`` that
actually carries the meaning.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

#: Substituted for the matched modality token when building a pairing key.
PLACEHOLDER = "<M>"

THERMAL = "thermal"
RGB = "rgb"

# Ordered longest-first at match time, not here.
# Every token here must be long/distinctive enough that a boundary-anchored
# match is meaningful. `ther` was deliberately left out: it matches `other`.
STRONG_THERMAL_TOKENS: tuple[str, ...] = (
    "thermal_8_bit",
    "thermal_16_bit",
    "infrared",
    "thermal",
    "lwir",
    "imgr",
    "tir",
)
STRONG_RGB_TOKENS: tuple[str, ...] = (
    "rgb_8_bit",
    "visible",
    "colour",
    "color",
    "rgb",
    "vis",
)
# Short and accident-prone (`ir` matches `air`, `img` matches `images`), so they
# only get consulted after every strong token has failed. A misfire here is not
# fatal anyway: a wrongly-labelled file simply fails to find a partner and is
# dropped from the manifest rather than corrupting a pair.
WEAK_THERMAL_TOKENS: tuple[str, ...] = ("ir",)
WEAK_RGB_TOKENS: tuple[str, ...] = ("img",)


def _flatten(text: str) -> str:
    """Lowercase and drop separators, so ``Thermal-8-Bit`` == ``thermal_8_bit``."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """True when the matched span is not buried inside a longer word.

    A match is accepted at the start or end of the component, or when both
    neighbours are non-alphanumeric. This is what keeps ``ir`` from firing on
    ``first`` while still allowing it on ``ir_images`` and ``trainimgr``.
    """
    if start == 0 or end == len(text):
        return True
    return not text[start - 1].isalnum() and not text[end].isalnum()


def _search(text: str, tokens: Sequence[str]) -> Optional[tuple[int, int, str]]:
    """Longest boundary-valid token match in ``text``, or None."""
    best: Optional[tuple[int, int, str]] = None
    for token in tokens:
        for match in re.finditer(re.escape(token.lower()), text):
            if not _boundary_ok(text, match.start(), match.end()):
                continue
            if best is None or (match.end() - match.start()) > (best[1] - best[0]):
                best = (match.start(), match.end(), token)
            break  # first occurrence of this token is enough
    return best


def classify_component(
    component: str,
    thermal_tokens: Sequence[str] = STRONG_THERMAL_TOKENS,
    rgb_tokens: Sequence[str] = STRONG_RGB_TOKENS,
    weak_thermal: Sequence[str] = WEAK_THERMAL_TOKENS,
    weak_rgb: Sequence[str] = WEAK_RGB_TOKENS,
) -> tuple[Optional[str], str]:
    """Classify one path component.

    Returns ``(modality, normalized)`` where modality is ``"rgb"``,
    ``"thermal"`` or ``None``, and ``normalized`` has the modality token
    replaced by :data:`PLACEHOLDER` (or is the lowercased component unchanged
    when nothing matched).
    """
    lowered = component.lower()
    flat = _flatten(component)

    # Pass 1 - the whole component *is* a token. Most reliable signal.
    for modality, tokens in (
        (THERMAL, list(thermal_tokens) + list(weak_thermal)),
        (RGB, list(rgb_tokens) + list(weak_rgb)),
    ):
        for token in tokens:
            if flat == _flatten(token):
                return modality, PLACEHOLDER

    # Pass 2 - strong tokens as substrings; longest match across both wins.
    thermal_hit = _search(lowered, sorted(thermal_tokens, key=len, reverse=True))
    rgb_hit = _search(lowered, sorted(rgb_tokens, key=len, reverse=True))
    winner = _pick(thermal_hit, rgb_hit)
    if winner is not None:
        modality, hit = winner
        return modality, lowered[: hit[0]] + PLACEHOLDER + lowered[hit[1] :]

    # Pass 3 - weak tokens only, now that strong ones are ruled out.
    thermal_hit = _search(lowered, sorted(weak_thermal, key=len, reverse=True))
    rgb_hit = _search(lowered, sorted(weak_rgb, key=len, reverse=True))
    winner = _pick(thermal_hit, rgb_hit)
    if winner is not None:
        modality, hit = winner
        return modality, lowered[: hit[0]] + PLACEHOLDER + lowered[hit[1] :]

    return None, lowered


def _pick(
    thermal_hit: Optional[tuple[int, int, str]],
    rgb_hit: Optional[tuple[int, int, str]],
) -> Optional[tuple[str, tuple[int, int, str]]]:
    """Choose the longer of the two matches; thermal wins exact ties."""
    if thermal_hit is None and rgb_hit is None:
        return None
    if rgb_hit is None:
        return THERMAL, thermal_hit  # type: ignore[return-value]
    if thermal_hit is None:
        return RGB, rgb_hit
    t_len = thermal_hit[1] - thermal_hit[0]
    r_len = rgb_hit[1] - rgb_hit[0]
    return (THERMAL, thermal_hit) if t_len >= r_len else (RGB, rgb_hit)
