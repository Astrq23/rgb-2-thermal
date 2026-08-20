"""Losses for conditional image translation.

The L1 term does the heavy lifting -- it fixes structure and overall intensity.
The adversarial term is what stops the result being the blurry conditional mean
that L1 alone converges to; it supplies the high-frequency thermal texture. The
default weighting (lambda_l1 = 100) is the pix2pix ratio and is a sensible
starting point rather than something tuned for this data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GAN_MODES = ("lsgan", "vanilla")


class GANLoss(nn.Module):
    """Adversarial loss with a matching-shaped target built on demand.

    ``lsgan`` (least squares) is the default: it gives smoother gradients and is
    markedly less prone to the discriminator running away early, which matters
    when several datasets of differing difficulty are mixed in one batch.
    """

    def __init__(self, mode: str = "lsgan", real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        if mode not in GAN_MODES:
            raise ValueError(f"gan_mode must be one of {GAN_MODES}, got {mode!r}")
        self.mode = mode
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = (self.real_label if target_is_real else self.fake_label).expand_as(prediction)
        if self.mode == "lsgan":
            return F.mse_loss(prediction, target)
        return F.binary_cross_entropy_with_logits(prediction, target)


def generator_loss(
    fake_logits: torch.Tensor,
    fake_thermal: torch.Tensor,
    real_thermal: torch.Tensor,
    gan_loss: GANLoss,
    lambda_l1: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generator objective: fool the critic, and stay close to the ground truth."""
    adversarial = gan_loss(fake_logits, True)
    reconstruction = F.l1_loss(fake_thermal, real_thermal)
    total = adversarial + lambda_l1 * reconstruction
    return total, {
        "g_total": float(total.detach()),
        "g_gan": float(adversarial.detach()),
        "g_l1": float(reconstruction.detach()),
    }


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    gan_loss: GANLoss,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Critic objective, halved so it does not outpace the generator."""
    real_term = gan_loss(real_logits, True)
    fake_term = gan_loss(fake_logits, False)
    total = 0.5 * (real_term + fake_term)
    return total, {
        "d_total": float(total.detach()),
        "d_real": float(real_term.detach()),
        "d_fake": float(fake_term.detach()),
    }
