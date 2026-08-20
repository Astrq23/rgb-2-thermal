"""Model factory.

Everything the trainer knows about architecture goes through here, so adding a
plain-U-Net baseline later means adding a branch in this file plus a YAML --
never touching engine/trainer.py.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..config import ModelConfig
from .patchgan import PatchGANDiscriminator
from .unet_generator import UNetGenerator

RGB_CHANNELS = 3
THERMAL_CHANNELS = 1


def build_generator(cfg: ModelConfig, num_domains: int = 0) -> nn.Module:
    if cfg.name not in {"pix2pix", "unet"}:
        raise ValueError(f"Unknown model {cfg.name!r}; expected 'pix2pix' or 'unet'")
    return UNetGenerator(
        in_channels=RGB_CHANNELS,
        out_channels=THERMAL_CHANNELS,
        ngf=cfg.ngf,
        n_down=cfg.n_down,
        norm=cfg.norm,
        dropout=cfg.dropout,
        num_domains=num_domains if cfg.use_domain_embedding else 0,
        domain_dim=cfg.domain_dim,
    )


def build_discriminator(cfg: ModelConfig) -> Optional[nn.Module]:
    """``None`` for the plain-U-Net baseline, which trains on L1 alone."""
    if cfg.name == "unet":
        return None
    return PatchGANDiscriminator(
        in_channels=RGB_CHANNELS + THERMAL_CHANNELS,
        ndf=cfg.ndf,
        n_layers=cfg.n_layers_d,
        norm=cfg.norm,
    )


def build_models(cfg: ModelConfig, num_domains: int = 0) -> tuple[nn.Module, Optional[nn.Module]]:
    return build_generator(cfg, num_domains), build_discriminator(cfg)
