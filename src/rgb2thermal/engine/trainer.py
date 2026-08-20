"""Training loop for conditional RGB -> thermal translation.

Shaped around one operational constraint: a Kaggle session dies at 12 hours,
without warning and without mercy. So the loop checkpoints every epoch, writes
metrics to CSV as it goes, and can resume into the middle of a run with the RNG
state intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import Config
from ..losses import GANLoss, discriminator_loss, generator_loss
from ..models.build import build_models
from ..utils.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from ..utils.logging import CSVLogger, Stopwatch
from ..utils.paths import default_output_dir, ensure_dir
from ..viz import save_comparison_grid
from .amp import autocast, make_scaler


class Trainer:
    def __init__(
        self,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        domain_index: Optional[dict[str, int]] = None,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.domain_index = domain_index or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        output_root = Path(cfg.train.output_dir) if cfg.train.output_dir else default_output_dir()
        self.output_dir = ensure_dir(output_root / cfg.name)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.sample_dir = ensure_dir(self.output_dir / "samples")

        self.generator, self.discriminator = build_models(cfg.model, len(self.domain_index))
        self.generator.to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)

        self.gan_loss = GANLoss(cfg.train.gan_mode).to(self.device)

        betas = (cfg.train.beta1, cfg.train.beta2)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=cfg.train.lr, betas=betas)
        self.optimizer_d = (
            torch.optim.Adam(self.discriminator.parameters(), lr=cfg.train.lr, betas=betas)
            if self.discriminator is not None
            else None
        )

        self.scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g, self._lr_lambda)
        self.scheduler_d = (
            torch.optim.lr_scheduler.LambdaLR(self.optimizer_d, self._lr_lambda)
            if self.optimizer_d is not None
            else None
        )

        self.scaler_g = make_scaler(cfg.train.amp)
        self.scaler_d = make_scaler(cfg.train.amp)

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_l1 = float("inf")
        self.stopwatch = Stopwatch()
        self.logger: Optional[CSVLogger] = None

    # ------------------------------------------------------------- scheduling

    def _lr_lambda(self, epoch: int) -> float:
        """Constant LR, then a linear ramp to zero -- the pix2pix schedule."""
        total = self.cfg.train.epochs
        start = self.cfg.train.lr_decay_start or max(1, total // 2)
        if epoch < start:
            return 1.0
        return max(0.0, 1.0 - (epoch - start) / max(1, total - start))

    # ------------------------------------------------------------- checkpoints

    def state_dict(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_l1": self.best_val_l1,
            "generator": self.generator.state_dict(),
            "discriminator": (
                self.discriminator.state_dict() if self.discriminator is not None else None
            ),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": (
                self.optimizer_d.state_dict() if self.optimizer_d is not None else None
            ),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": (
                self.scheduler_d.state_dict() if self.scheduler_d is not None else None
            ),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "config": self.cfg.to_dict(),
            "domain_index": self.domain_index,
        }

    def resume(self, path: Path | str) -> None:
        checkpoint = load_checkpoint(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        if self.discriminator is not None and checkpoint.get("discriminator"):
            self.discriminator.load_state_dict(checkpoint["discriminator"])

        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        if self.optimizer_d is not None and checkpoint.get("optimizer_d"):
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        self.scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        if self.scheduler_d is not None and checkpoint.get("scheduler_d"):
            self.scheduler_d.load_state_dict(checkpoint["scheduler_d"])

        self.scaler_g.load_state_dict(checkpoint["scaler_g"])
        self.scaler_d.load_state_dict(checkpoint["scaler_d"])

        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_l1 = float(checkpoint.get("best_val_l1", float("inf")))
        restore_rng(checkpoint.get("rng"))

        print(
            f"[trainer] resumed from {path} at epoch {self.start_epoch}, "
            f"step {self.global_step:,} (best val L1 {self.best_val_l1:.4f})"
        )

    # -------------------------------------------------------------- main loop

    def fit(self) -> dict[str, Any]:
        self.logger = CSVLogger(self.output_dir / "metrics.csv", resume=self.start_epoch > 0)
        cfg = self.cfg.train

        if self.start_epoch >= cfg.epochs:
            print(f"[trainer] nothing to do: already at epoch {self.start_epoch}/{cfg.epochs}")
            return {"epochs_run": 0}

        print(
            f"[trainer] device={self.device}  epochs={self.start_epoch}..{cfg.epochs - 1}  "
            f"batch={cfg.batch_size}  amp={cfg.amp}  output={self.output_dir}"
        )

        stopped_early = False
        for epoch in range(self.start_epoch, cfg.epochs):
            train_stats = self._train_epoch(epoch)
            val_stats = self._validate(epoch) if self.val_loader is not None else {}

            self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

            save_checkpoint(self.checkpoint_dir / "last.pt", **self.state_dict(epoch))

            val_l1 = val_stats.get("val_l1")
            if val_l1 is not None and val_l1 < self.best_val_l1:
                self.best_val_l1 = val_l1
                save_checkpoint(self.checkpoint_dir / "best.pt", **self.state_dict(epoch))
                print(f"[trainer] new best val L1: {val_l1:.4f}")

            self.logger.log(
                epoch=epoch,
                step=self.global_step,
                lr=self.optimizer_g.param_groups[0]["lr"],
                elapsed=self.stopwatch.format(),
                **train_stats,
                **val_stats,
            )

            if cfg.max_steps and self.global_step >= cfg.max_steps:
                print(f"[trainer] reached max_steps={cfg.max_steps}, stopping")
                stopped_early = True
                break

        return {
            "epochs_run": epoch - self.start_epoch + 1,
            "global_step": self.global_step,
            "best_val_l1": self.best_val_l1,
            "output_dir": str(self.output_dir),
            "stopped_early": stopped_early,
        }

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        cfg = self.cfg.train
        self.generator.train()
        if self.discriminator is not None:
            self.discriminator.train()

        totals: dict[str, float] = {}
        seen = 0
        max_steps = cfg.max_steps_per_epoch or len(self.train_loader)
        progress = tqdm(
            self.train_loader,
            total=min(max_steps, len(self.train_loader)),
            desc=f"epoch {epoch}",
            leave=False,
        )

        for step, batch in enumerate(progress):
            if step >= max_steps:
                break

            stats = self._train_step(batch)
            self.global_step += 1
            seen += 1
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value

            if self.global_step % max(1, cfg.log_interval) == 0:
                progress.set_postfix(
                    {k: f"{v:.3f}" for k, v in stats.items() if k in ("g_l1", "g_gan", "d_total")}
                )

            if cfg.sample_interval and self.global_step % cfg.sample_interval == 0:
                self._dump_samples(batch, tag=f"step{self.global_step:07d}")

            if cfg.max_steps and self.global_step >= cfg.max_steps:
                break

        progress.close()
        return {f"train_{k}": v / max(1, seen) for k, v in totals.items()}

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        cfg = self.cfg.train
        rgb = batch["rgb"].to(self.device, non_blocking=True)
        real_thermal = batch["thermal"].to(self.device, non_blocking=True)
        domain = batch["domain"].to(self.device, non_blocking=True)

        stats: dict[str, float] = {}

        with autocast(cfg.amp):
            fake_thermal = self.generator(rgb, domain)

        # ---- discriminator ------------------------------------------------
        if self.discriminator is not None and self.optimizer_d is not None:
            self.optimizer_d.zero_grad(set_to_none=True)
            with autocast(cfg.amp):
                real_logits = self.discriminator(rgb, real_thermal)
                # detach: the generator must not be updated by the critic's step.
                fake_logits = self.discriminator(rgb, fake_thermal.detach())
                loss_d, stats_d = discriminator_loss(real_logits, fake_logits, self.gan_loss)
            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            stats.update(stats_d)

        # ---- generator ----------------------------------------------------
        self.optimizer_g.zero_grad(set_to_none=True)
        with autocast(cfg.amp):
            if self.discriminator is not None:
                fake_logits_for_g = self.discriminator(rgb, fake_thermal)
                loss_g, stats_g = generator_loss(
                    fake_logits_for_g, fake_thermal, real_thermal, self.gan_loss, cfg.lambda_l1
                )
            else:
                # Plain U-Net baseline: reconstruction only.
                reconstruction = F.l1_loss(fake_thermal, real_thermal)
                loss_g = cfg.lambda_l1 * reconstruction
                stats_g = {"g_total": float(loss_g.detach()), "g_l1": float(reconstruction.detach())}
        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()
        stats.update(stats_g)

        return stats

    # ------------------------------------------------------------ validation

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.generator.eval()
        total_l1 = 0.0
        seen = 0
        first_batch: Optional[dict[str, Any]] = None

        for batch in self.val_loader:
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            real_thermal = batch["thermal"].to(self.device, non_blocking=True)
            domain = batch["domain"].to(self.device, non_blocking=True)

            with autocast(self.cfg.train.amp):
                fake_thermal = self.generator(rgb, domain)

            total_l1 += float(F.l1_loss(fake_thermal.float(), real_thermal).detach()) * rgb.size(0)
            seen += rgb.size(0)
            if first_batch is None:
                first_batch = batch

        if first_batch is not None:
            self._dump_samples(first_batch, tag=f"val_epoch{epoch:03d}")

        return {"val_l1": total_l1 / max(1, seen), "val_n": float(seen)}

    @torch.no_grad()
    def _dump_samples(self, batch: dict[str, Any], tag: str) -> None:
        """Write a RGB / real / generated grid. Cheap, and the best sanity check."""
        was_training = self.generator.training
        self.generator.eval()

        count = min(self.cfg.eval.num_visuals, batch["rgb"].size(0))
        rgb = batch["rgb"][:count].to(self.device)
        real_thermal = batch["thermal"][:count].to(self.device)
        domain = batch["domain"][:count].to(self.device)

        with autocast(self.cfg.train.amp):
            fake_thermal = self.generator(rgb, domain)

        save_comparison_grid(
            rgb.float().cpu(),
            real_thermal.float().cpu(),
            fake_thermal.float().cpu(),
            self.sample_dir / f"{tag}.png",
            max_items=count,
            titles=list(batch.get("dataset", []))[:count],
        )

        if was_training:
            self.generator.train()
