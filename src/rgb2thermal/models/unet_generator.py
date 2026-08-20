"""U-Net generator: RGB (3 channels) -> thermal (1 channel).

Written as an explicit encoder/decoder pair rather than the recursive
``UnetSkipConnectionBlock`` of the reference pix2pix implementation. The
recursion is elegant but makes the bottleneck unreachable, and reaching the
bottleneck is exactly what the optional dataset conditioning needs.

Skip connections are what matter most here: thermal edges land on the same
pixels as RGB edges (buildings, vehicles, roads), so the decoder needs the
encoder's high-resolution detail even though the *intensities* are unrelated.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm {kind!r}; expected batch|instance|none")


class FiLM(nn.Module):
    """Per-dataset feature modulation at the bottleneck.

    Merging several thermal cameras means the same RGB scene has several
    plausible thermal renderings (different gain, palette, sensor response).
    Given a dataset id, this predicts a scale and shift for the bottleneck
    features, letting one generator represent all of them instead of averaging
    them into mush. Optional -- ``model.use_domain_embedding``.
    """

    def __init__(self, num_domains: int, embed_dim: int, channels: int):
        super().__init__()
        self.embedding = nn.Embedding(num_domains, embed_dim)
        self.to_affine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, channels * 2),
        )
        # Start as identity so enabling conditioning does not disturb a run.
        nn.init.zeros_(self.to_affine[-1].weight)
        nn.init.zeros_(self.to_affine[-1].bias)

    def forward(self, features: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_affine(self.embedding(domain)).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return features * (1.0 + gamma) + beta


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        ngf: int = 64,
        n_down: int = 8,
        norm: str = "batch",
        dropout: float = 0.5,
        num_domains: int = 0,
        domain_dim: int = 32,
    ):
        super().__init__()
        if n_down < 2:
            raise ValueError("n_down must be >= 2")

        # 64, 128, 256, 512, 512, ... capped at ngf * 8.
        widths = [min(ngf * 2**i, ngf * 8) for i in range(n_down)]
        self.widths = widths
        self.n_down = n_down

        encoder: list[nn.Module] = []
        previous = in_channels
        for level, width in enumerate(widths):
            layers: list[nn.Module] = []
            if level > 0:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            use_norm = 0 < level < n_down - 1
            layers.append(nn.Conv2d(previous, width, 4, 2, 1, bias=not use_norm))
            if use_norm:
                layers.append(make_norm(norm, width))
            encoder.append(nn.Sequential(*layers))
            previous = width
        self.encoder = nn.ModuleList(encoder)

        self.film: Optional[FiLM] = None
        if num_domains > 0:
            self.film = FiLM(num_domains, domain_dim, widths[-1])

        # Decoder level i consumes level i's features (doubled by the skip
        # concat everywhere except the bottleneck) and produces level i-1's.
        decoder: list[nn.Module] = []
        for level in range(n_down - 1, 0, -1):
            in_width = widths[level] if level == n_down - 1 else widths[level] * 2
            out_width = widths[level - 1]
            layers = [
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_width, out_width, 4, 2, 1, bias=False),
                make_norm(norm, out_width),
            ]
            # Dropout in the three innermost blocks only, as in pix2pix: it is
            # the generator's only source of stochasticity.
            if dropout > 0 and level >= n_down - 3:
                layers.append(nn.Dropout(dropout))
            decoder.append(nn.Sequential(*layers))
        self.decoder = nn.ModuleList(decoder)

        self.to_image = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(widths[0] * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

        self.apply(init_weights)

    def forward(self, rgb: torch.Tensor, domain: Optional[torch.Tensor] = None) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        features = rgb
        for block in self.encoder:
            features = block(features)
            skips.append(features)

        if self.film is not None and domain is not None:
            features = self.film(features, domain)

        for offset, block in enumerate(self.decoder):
            features = block(features)
            skip = skips[self.n_down - 2 - offset]
            features = torch.cat([features, skip], dim=1)

        return self.to_image(features)


def init_weights(module: nn.Module) -> None:
    """DCGAN-style init (normal, 0.02) -- the pix2pix default."""
    name = module.__class__.__name__
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif "BatchNorm" in name and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)
