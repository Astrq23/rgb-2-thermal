#!/usr/bin/env python
"""Generate thermal images from arbitrary RGB inputs.

    python scripts/predict.py --checkpoint outputs/pix2pix_uav/checkpoints/best.pt --input photo.jpg
    python scripts/predict.py --checkpoint .../best.pt --input my_uav_frames/ --out preds/ --colormap inferno

Takes a file or a directory. Output is single-channel grayscale by default;
``--colormap`` writes a colourised version instead, and ``--side-by-side``
writes the input next to the prediction, which is the more useful artefact when
showing results to someone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import torch
from PIL import Image

from rgb2thermal.config import ModelConfig, load_config, load_dataset_specs
from rgb2thermal.data.normalize import rgb_to_array
from rgb2thermal.models.build import build_generator
from rgb2thermal.utils.checkpoint import load_checkpoint
from rgb2thermal.utils.paths import IMAGE_EXTS, ensure_dir
from rgb2thermal.viz import colorize, thermal_tensor_to_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--out", default="predictions")
    parser.add_argument("--config", default="configs/pix2pix_uav.yaml")
    parser.add_argument("--colormap", default=None, help="e.g. inferno; omit for grayscale")
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument(
        "--domain",
        default=None,
        help="Dataset name to condition on, when the model was trained with use_domain_embedding",
    )
    parser.add_argument(
        "--keep-size",
        action="store_true",
        help="Resize the prediction back to the input's original resolution",
    )
    return parser.parse_args()


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"No images found under {path}")
    return files


def prepare(image: Image.Image, size: int) -> torch.Tensor:
    """Short-side resize then centre crop -- matches evaluation preprocessing."""
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize(
        (max(size, int(round(width * scale))), max(size, int(round(height * scale)))),
        Image.BICUBIC,
    )
    left = (resized.size[0] - size) // 2
    top = (resized.size[1] - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    array = rgb_to_array(cropped).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(array))[None]


def model_config_from_checkpoint(checkpoint, fallback):
    """Prefer the architecture the checkpoint was trained with.

    Otherwise any run that used `--set model.ngf=...` fails to load here with an
    opaque state_dict shape error. The YAML still supplies everything else; only
    the architecture is taken from the checkpoint.
    """
    stored = (checkpoint.get("config") or {}).get("model")
    if not stored:
        return fallback
    try:
        merged = ModelConfig(**stored)
    except TypeError:
        return fallback
    if merged != fallback:
        print(f"[checkpoint] using the architecture stored in the checkpoint: {stored}")
    return merged


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    specs = load_dataset_specs(cfg.data.dataset_specs)
    domain_index = checkpoint.get("domain_index") or {
        name: idx for idx, name in enumerate(sorted({s.name for s in specs}))
    }

    model_cfg = model_config_from_checkpoint(checkpoint, cfg.model)
    generator = build_generator(model_cfg, num_domains=len(domain_index)).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    domain_id = domain_index.get(args.domain, 0) if args.domain else 0
    if args.domain and args.domain not in domain_index:
        print(f"[predict] unknown --domain {args.domain!r}; known: {sorted(domain_index)}")

    inputs = collect_inputs(Path(args.input))
    out_dir = ensure_dir(args.out)
    print(f"Generating thermal for {len(inputs)} image(s) on {device} -> {out_dir}")

    for path in inputs:
        with Image.open(path) as handle:
            original = handle.convert("RGB")

        tensor = prepare(original, cfg.data.image_size).to(device)
        domain = torch.tensor([domain_id], dtype=torch.long, device=device)

        with torch.no_grad():
            prediction = generator(tensor, domain)

        gray = thermal_tensor_to_array(prediction[0].cpu())
        array = colorize(gray, args.colormap) if args.colormap else gray
        result = Image.fromarray(array)

        if args.keep_size:
            result = result.resize(original.size, Image.BICUBIC)

        if args.side_by_side:
            reference = original.resize(result.size, Image.BICUBIC).convert("RGB")
            canvas = Image.new("RGB", (result.size[0] * 2, result.size[1]))
            canvas.paste(reference, (0, 0))
            canvas.paste(result.convert("RGB"), (result.size[0], 0))
            result = canvas

        destination = Path(out_dir) / f"{path.stem}_thermal.png"
        result.save(destination)
        print(f"  {path.name} -> {destination.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
