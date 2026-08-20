"""Shape and wiring checks for the generator, the critic and the losses."""

from __future__ import annotations

import pytest
import torch

from rgb2thermal.config import ModelConfig
from rgb2thermal.losses import GANLoss, discriminator_loss, generator_loss
from rgb2thermal.models.build import build_discriminator, build_generator, build_models
from rgb2thermal.models.patchgan import PatchGANDiscriminator
from rgb2thermal.models.unet_generator import UNetGenerator


def test_generator_maps_rgb_to_single_channel_thermal():
    generator = UNetGenerator(n_down=8)
    output = generator(torch.randn(2, 3, 256, 256))
    assert output.shape == (2, 1, 256, 256)


def test_generator_output_is_bounded_by_tanh():
    """Must match the [-1, 1] range the data pipeline normalises into."""
    output = UNetGenerator(n_down=6)(torch.randn(2, 3, 64, 64))
    assert output.min() >= -1.0 and output.max() <= 1.0


@pytest.mark.parametrize("size,n_down", [(64, 6), (128, 7), (256, 8)])
def test_generator_handles_configured_depths(size, n_down):
    output = UNetGenerator(n_down=n_down, ngf=8)(torch.randn(1, 3, size, size))
    assert output.shape == (1, 1, size, size)


def test_generator_rejects_a_degenerate_depth():
    with pytest.raises(ValueError, match="n_down"):
        UNetGenerator(n_down=1)


def test_discriminator_returns_a_patch_map_not_a_scalar():
    """PatchGAN's value is the dense verdict; a 1x1 output would mean it collapsed."""
    critic = PatchGANDiscriminator()
    logits = critic(torch.randn(2, 3, 256, 256), torch.randn(2, 1, 256, 256))
    assert logits.shape[0] == 2 and logits.shape[1] == 1
    assert logits.shape[2] > 1 and logits.shape[3] > 1


def test_discriminator_sees_both_modalities():
    """It is conditional: swapping the RGB must change the verdict."""
    torch.manual_seed(0)
    critic = PatchGANDiscriminator().eval()
    thermal = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        first = critic(torch.randn(1, 3, 64, 64), thermal)
        second = critic(torch.randn(1, 3, 64, 64), thermal)
    assert not torch.allclose(first, second)


# ------------------------------------------------------------ domain embedding


def test_domain_embedding_starts_as_identity():
    """Enabling conditioning must not perturb an otherwise identical model."""
    torch.manual_seed(0)
    conditioned = UNetGenerator(n_down=6, ngf=8, num_domains=3, dropout=0.0).eval()
    rgb = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        as_domain_0 = conditioned(rgb, torch.zeros(2, dtype=torch.long))
        as_domain_2 = conditioned(rgb, torch.full((2,), 2, dtype=torch.long))
    # FiLM is zero-initialised, so every domain behaves the same at step 0.
    assert torch.allclose(as_domain_0, as_domain_2)


def test_domain_embedding_can_diverge_once_trained():
    generator = UNetGenerator(n_down=6, ngf=8, num_domains=3, dropout=0.0).eval()
    with torch.no_grad():
        for parameter in generator.film.to_affine[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)
        rgb = torch.randn(2, 3, 64, 64)
        first = generator(rgb, torch.zeros(2, dtype=torch.long))
        second = generator(rgb, torch.full((2,), 2, dtype=torch.long))
    assert not torch.allclose(first, second)


def test_generator_ignores_domain_when_conditioning_is_off():
    generator = UNetGenerator(n_down=6, ngf=8, num_domains=0).eval()
    assert generator.film is None
    with torch.no_grad():
        output = generator(torch.randn(1, 3, 64, 64), torch.tensor([5]))
    assert output.shape == (1, 1, 64, 64)


# -------------------------------------------------------------------- factory


def test_build_models_from_config():
    cfg = ModelConfig(name="pix2pix", ngf=8, ndf=8, n_down=6)
    generator, critic = build_models(cfg, num_domains=2)
    assert isinstance(generator, UNetGenerator)
    assert isinstance(critic, PatchGANDiscriminator)


def test_unet_baseline_has_no_discriminator():
    """The L1-only baseline shares the trainer; it just skips the critic."""
    assert build_discriminator(ModelConfig(name="unet")) is None


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_generator(ModelConfig(name="stylegan"))


def test_domain_embedding_only_built_when_enabled():
    cfg = ModelConfig(ngf=8, n_down=6, use_domain_embedding=False)
    assert build_generator(cfg, num_domains=4).film is None
    cfg.use_domain_embedding = True
    assert build_generator(cfg, num_domains=4).film is not None


# --------------------------------------------------------------------- losses


@pytest.mark.parametrize("mode", ["lsgan", "vanilla"])
def test_gan_loss_prefers_the_matching_target(mode):
    loss = GANLoss(mode)
    confident_real = torch.full((2, 1, 8, 8), 5.0 if mode == "vanilla" else 1.0)
    assert loss(confident_real, True) < loss(confident_real, False)


def test_gan_loss_rejects_unknown_mode():
    with pytest.raises(ValueError, match="gan_mode"):
        GANLoss("wgan")


def test_generator_loss_is_dominated_by_l1():
    """lambda_l1=100 is what stops the adversarial term producing noise."""
    logits = torch.zeros(2, 1, 8, 8)
    fake = torch.zeros(2, 1, 16, 16)
    real = torch.ones(2, 1, 16, 16)
    total, stats = generator_loss(logits, fake, real, GANLoss("lsgan"), lambda_l1=100.0)

    assert stats["g_l1"] == pytest.approx(1.0)
    assert total.item() == pytest.approx(stats["g_gan"] + 100.0)


def test_discriminator_loss_is_halved():
    loss = GANLoss("lsgan")
    real_logits = torch.ones(2, 1, 4, 4)
    fake_logits = torch.zeros(2, 1, 4, 4)
    total, stats = discriminator_loss(real_logits, fake_logits, loss)
    assert total.item() == pytest.approx(0.5 * (stats["d_real"] + stats["d_fake"]))


def test_losses_backpropagate():
    generator = UNetGenerator(n_down=6, ngf=8)
    critic = PatchGANDiscriminator(ndf=8)
    rgb = torch.randn(2, 3, 64, 64)
    real = torch.randn(2, 1, 64, 64)

    fake = generator(rgb)
    total, _ = generator_loss(critic(rgb, fake), fake, real, GANLoss("lsgan"), 100.0)
    total.backward()

    grads = [p.grad for p in generator.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
