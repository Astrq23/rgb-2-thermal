"""PatchGAN discriminator.

Classifies overlapping ~70x70 patches rather than the whole image. For thermal
synthesis that is the right granularity: realism here is a local property
(does this rooftop have a plausible heat signature?), and a patch critic gives
far denser gradient than a single global verdict.

It is *conditional*: RGB and thermal are concatenated on the channel axis, so
the discriminator judges whether a thermal image is plausible **for this
particular RGB input**, not merely whether it looks thermal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet_generator import init_weights, make_norm


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,  # 3 (RGB) + 1 (thermal)
        ndf: int = 64,
        n_layers: int = 3,
        norm: str = "batch",
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        width = ndf
        for layer in range(1, n_layers):
            previous, width = width, min(ndf * 2**layer, ndf * 8)
            layers += [
                nn.Conv2d(previous, width, 4, 2, 1, bias=False),
                make_norm(norm, width),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final stride-1 pair keeps the receptive field at 70x70 without
        # shrinking the output map further.
        previous, width = width, min(ndf * 2**n_layers, ndf * 8)
        layers += [
            nn.Conv2d(previous, width, 4, 1, 1, bias=False),
            make_norm(norm, width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, 1, 4, 1, 1),
        ]

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([rgb, thermal], dim=1))
