#!/usr/bin/env python
"""Generate the Kaggle notebooks from source.

The notebooks are the user-facing entry point, but a .ipynb is an awkward thing
to keep correct by hand -- JSON escaping, cell ids, metadata. Generating them
means the code inside them is written as ordinary Python here and cannot drift
into invalid JSON.

    python scripts/build_notebooks.py

Re-run after editing any cell text below.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

#: The published repository the Kaggle notebooks clone from.
REPO_URL = "https://github.com/Astrq23/rgb-2-thermal.git"
REPO_NAME = "rgb-2-thermal"

KERNEL_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "GPU",
}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    stripped = text.strip("\n")
    return [line + "\n" for line in stripped.split("\n")[:-1]] + [stripped.split("\n")[-1]]


def write_notebook(name: str, cells: list[dict]) -> Path:
    notebook = {
        "cells": cells,
        "metadata": KERNEL_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------- shared cells

SETUP_CELL = f'''
# --- Pull the project code from GitHub -------------------------------------
# Requires "Internet" to be ON in the notebook settings panel on the right.
REPO_URL = "{REPO_URL}"
BRANCH   = "main"

import os, subprocess, sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/{REPO_NAME}")

if REPO_DIR.exists():
    # Re-running the notebook: fast-forward instead of re-cloning.
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{{BRANCH}}"], check=True)
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True
    )

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))

print("repo:", REPO_DIR)
print("commit:", subprocess.run(
    ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True).stdout.strip())
'''

INSTALL_CELL = '''
# Editable install so `import rgb2thermal` works everywhere, including inside
# DataLoader worker processes. --no-deps keeps Kaggle's preinstalled torch.
!pip install -e . --no-deps -q

import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
'''


# ---------------------------------------------------------------- notebook 00


def notebook_explore() -> list[dict]:
    return [
        markdown(
            """
# 00 · Explore the datasets

**Run this first.** It answers one question: *do the adapters actually understand
the folder layout of the Kaggle mirrors you attached?*

Community mirrors sometimes reorganise the upstream structure. When they do,
every later step fails with "0 pairs" or an empty manifest, and the fix belongs
here — in `configs/datasets/*.yaml`, not in Python.

### Before running

1. **Add data** (right panel) → attach the datasets you want:
   - `brendanalvey/visdrone-dronevehicle` — DroneVehicle, the core UAV source
   - `monishshrivastava1/llvip-dataset` — LLVIP, night-time, tightly aligned
   - `samdazel/teledyne-flir-adas-thermal-dataset-v2` — FLIR ADAS, ground level
   - `pandrii000/hituav-a-highaltitude-infrared-thermal-dataset` — HIT-UAV, evaluation only
2. **Settings** → Accelerator: **GPU**, Internet: **ON**

Attaching a subset is fine; missing datasets are skipped, not fatal.
"""
        ),
        code(SETUP_CELL),
        code(INSTALL_CELL),
        markdown(
            """
## What is actually mounted?

The tree below is ground truth. Compare it against the `search_roots` in each
`configs/datasets/*.yaml`.
"""
        ),
        code("!python scripts/inspect_datasets.py --depth 4"),
        markdown(
            """
### Reading the probe output

| Status | Meaning | What to do |
|---|---|---|
| `OK — N pairs` | The adapter found the data | Nothing |
| `NOT MOUNTED` | No search root matched | Attach the dataset, or add its path to `search_roots` |
| `root exists but 0 pairs matched` | Folder layout differs from expectations | Compare the tree above with the YAML and fix the globs |

For the last case, the usual fixes are all config-only:

- the RGB/thermal folders use unexpected names → add them to `thermal_tokens` / `rgb_tokens`
- the two modalities do not share a filename → set `key_regex`
- there is genuinely no modality folder → set `force_modality` (as HIT-UAV does)
"""
        ),
        markdown("## Build the unified manifest"),
        code("!python scripts/build_manifest.py --out /kaggle/working/manifest.csv"),
        markdown(
            """
Check the printed table before going further:

- **Counts** — DroneVehicle should land near 28k pairs, LLVIP near 15k. Anything
  in the low hundreds means the adapter is only seeing part of the tree.
- **Split leakage** — a warning naming a dataset means its scenes appear in more
  than one split. That is expected when its own official split is honoured; if
  you would rather have strictly disjoint scenes, rebuild with
  `--set data.respect_split_hint=false`.
"""
        ),
        code(
            '''
import pandas as pd

manifest = pd.read_csv("/kaggle/working/manifest.csv", keep_default_na=False)
display(pd.crosstab(manifest["dataset"], manifest["split"], margins=True))
display(pd.crosstab(manifest["viewpoint"], manifest["time_of_day"], margins=True))
'''
        ),
        markdown(
            """
## Alignment check — the step people skip and regret

Paired training assumes the RGB and thermal frames show the same scene at the
same instant. DroneVehicle and LLVIP are hardware-aligned. **FLIR is not**: its
two cameras have different fields of view, and `preprocess.rgb.fov_crop` in
`configs/datasets/flir_v2.yaml` is an estimate, not a calibration.

A misaligned pair does not raise an error. It teaches the generator to blur, and
the loss curve looks perfectly healthy the whole time.
"""
        ),
        code(
            '''
!python scripts/check_alignment.py --manifest /kaggle/working/manifest.csv --n 3

from pathlib import Path
from IPython.display import Image as ShowImage, display

for path in sorted(Path("/kaggle/working/outputs/alignment").glob("*.png")):
    print(path.name)
    display(ShowImage(filename=str(path), width=1100))
'''
        ),
        markdown(
            """
## Optional but recommended: export a compact subset

The source datasets are tens of gigabytes, and most of that is resolution
training throws away immediately — DroneVehicle ships 840x712 frames that get
cropped and resized to 256x256. Every session pays the attach cost for the full
resolution, then decodes it again on every epoch.

Run the export **once**. It bakes in each dataset's geometry (border crop, FOV
crop), resizes to 288px and writes a folder typically **20–50x smaller**.
"""
        ),
        code(
            '''
!python scripts/export_subset.py \\
    --manifest /kaggle/working/manifest.csv \\
    --out /kaggle/working/subset \\
    --per-dataset 8000
'''
        ),
        markdown(
            """
Then, to make it reusable:

1. **Save Version → Save & Run All** and wait for it to finish.
2. From the finished run, publish `/kaggle/working/subset` as a **new Kaggle
   Dataset** (Notebook Output → Create Dataset). Name it `rgb-thermal-subset`
   so the default `search_roots` find it.
3. In every later session attach **only that dataset**, never the originals as
   well — attaching both would put each sample into the manifest twice. Then:

   ```
   !python scripts/build_manifest.py \\
       --dataset-specs 'configs/datasets/subset*.yaml' \\
       --out /kaggle/working/manifest.csv
   ```

The subset keeps the source dataset names, the sampling weights, the
per-dataset metric breakdown and the leakage-free scene groups, because all of
that travels in the exported path. Thermal-only data (HIT-UAV) goes to
`reference/` and still serves unpaired FID.

Sessions then start in seconds, and the DataLoader stops being the bottleneck.
"""
        ),
        markdown(
            """
**Read the 4th panel** (thermal edges in red over RGB):

- Edges land on buildings, vehicles, road markings → aligned, keep the dataset.
- Edges float away from structures → misaligned. Try a different crop:

  ```
  !python scripts/check_alignment.py --dataset flir_v2 --fov-crop 0.55 --n 3
  ```

  Sweep a few values, then write the best one into
  `configs/datasets/flir_v2.yaml`. If nothing lines up, set `enabled: false` —
  less data beats data that teaches blurring.

Next: **01_kaggle_train.ipynb**.
"""
        ),
    ]


# ---------------------------------------------------------------- notebook 01


def notebook_train() -> list[dict]:
    return [
        markdown(
            """
# 01 · Train RGB → thermal (Pix2Pix, UAV-weighted)

Code lives on GitHub and is cloned in below; this notebook only orchestrates.
Edit code in the repo and re-run the setup cell — never paste code into cells,
or the run stops being reproducible.

**Settings → Accelerator: GPU · Internet: ON**

### The 12-hour problem

A Kaggle session is killed at 12 hours with no warning. Training checkpoints
after every epoch, so a run simply continues in the next session — see the last
section for how.
"""
        ),
        code(SETUP_CELL),
        code(INSTALL_CELL),
        markdown("## 1 · Manifest\n\nSkip to section 2 if you already ran notebook 00 in this session."),
        code(
            '''
import subprocess
from pathlib import Path

MANIFEST = Path("/kaggle/working/manifest.csv")

if MANIFEST.exists():
    print(f"reusing {MANIFEST}")
else:
    subprocess.run(
        ["python", "scripts/build_manifest.py", "--out", str(MANIFEST)], check=True
    )
'''
        ),
        markdown(
            """
## 2 · Smoke run (~2 minutes)

Fifty steps at a small resolution. It exercises the manifest, the DataLoader,
both networks, the loss, AMP, checkpointing and image dumping.

Two minutes here has repeatedly been worth more than the ten hours it protects:
a config typo that surfaces at hour nine costs a whole session.
"""
        ),
        code(
            '''
!python scripts/train.py \\
    --config configs/smoke.yaml \\
    --manifest /kaggle/working/manifest.csv
'''
        ),
        code(
            '''
from pathlib import Path
from IPython.display import Image as ShowImage, display

samples = sorted(Path("/kaggle/working/outputs/smoke/samples").glob("*.png"))
if samples:
    display(ShowImage(filename=str(samples[-1]), width=760))
else:
    print("No sample grid was written — check the smoke run output above.")
'''
        ),
        markdown(
            """
The output will be noise — fifty steps is nothing. What matters is that the grid
rendered at all, with three columns and the correct number of rows.

## 3 · Full training

Defaults (`configs/pix2pix_uav.yaml`): 40 epochs × 20,000 samples at 256×256,
batch 16, mixed precision. On a P100 that is roughly 8–10 hours — deliberately
inside one session.

The dataset mixture is weighted, not concatenated: DroneVehicle 0.60,
LLVIP 0.30, FLIR 0.10. UAV imagery stays dominant no matter how the download
sizes compare.

**Adjust before starting if needed:**

| Situation | Change |
|---|---|
| T4 ×2 instead of P100 | `--set train.batch_size=24 data.num_workers=4` |
| Out of memory | `--set train.batch_size=8` |
| Want a first result quickly | `--set train.epochs=10` |
| Session start is slow | Export a compact subset (notebook 00, last section) |

`train.max_hours=11` above is the safety margin: the run stops itself cleanly
and checkpoints at 11 hours instead of being killed mid-epoch at 12. After the
first epoch the trainer prints the **measured throughput (img/s)** and an
**ETA** for the remaining epochs, so you can tell within minutes whether the
run fits a session rather than guessing.
| FLIR failed its alignment check | set `enabled: false` in its YAML, then rebuild the manifest |
"""
        ),
        code(
            '''
!python scripts/train.py \\
    --config configs/pix2pix_uav.yaml \\
    --manifest /kaggle/working/manifest.csv \\
    --resume auto \\
    --set train.max_hours=11
'''
        ),
        markdown("## 4 · Training curves"),
        code(
            '''
import pandas as pd
import matplotlib.pyplot as plt

metrics = pd.read_csv("/kaggle/working/outputs/pix2pix_uav/metrics.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(metrics["epoch"], metrics["train_g_l1"], label="train")
if "val_l1" in metrics:
    axes[0].plot(metrics["epoch"], metrics["val_l1"], label="val")
axes[0].set_title("L1 reconstruction"); axes[0].set_xlabel("epoch"); axes[0].legend()

axes[1].plot(metrics["epoch"], metrics["train_g_gan"], label="generator")
axes[1].plot(metrics["epoch"], metrics["train_d_total"], label="discriminator")
axes[1].set_title("Adversarial"); axes[1].set_xlabel("epoch"); axes[1].legend()

axes[2].plot(metrics["epoch"], metrics["lr"])
axes[2].set_title("Learning rate"); axes[2].set_xlabel("epoch")

for axis in axes:
    axis.grid(alpha=0.3)
plt.tight_layout(); plt.show()

metrics.tail(10)
'''
        ),
        markdown(
            """
**What healthy training looks like**

- `train_g_l1` falls steadily and `val_l1` tracks it. A growing gap is overfitting.
- The adversarial losses oscillate around a rough equilibrium. That is normal.
- `train_d_total` collapsing to ~0 means the discriminator won; the generator
  stops receiving useful gradient. Lower `train.lr`, or raise `train.lambda_l1`.
- `train_g_l1` flat from the very start usually means the data is wrong, not the
  model — go back to the alignment check.
"""
        ),
        markdown("## 5 · Generated samples"),
        code(
            '''
from pathlib import Path
from IPython.display import Image as ShowImage, display

samples = sorted(Path("/kaggle/working/outputs/pix2pix_uav/samples").glob("val_epoch*.png"))
for path in samples[-3:]:
    print(path.name)
    display(ShowImage(filename=str(path), width=760))
'''
        ),
        markdown(
            """
Judge these by **thermal plausibility**, not by pixel match: vehicles and people
should be brighter than road and vegetation, engine bays hotter than bodywork,
buildings retaining heat after dark. Getting that ordering right matters more
than matching any individual frame.

## 6 · Evaluate
"""
        ),
        code(
            '''
!python scripts/evaluate.py \\
    --checkpoint /kaggle/working/outputs/pix2pix_uav/checkpoints/best.pt \\
    --config configs/pix2pix_uav.yaml \\
    --manifest /kaggle/working/manifest.csv \\
    --split test
'''
        ),
        markdown(
            """
Metrics are reported **per source dataset** on purpose. A single average would
hide the thing that matters here — whether the UAV domain improved, or whether a
gain came from the easier ground-level FLIR frames.

- **PSNR / SSIM** — fidelity to that exact ground-truth frame.
- **LPIPS** — perceptual distance. Usually the most informative of the three.
- **FID** — distributional realism across the whole set.
- **Unpaired FID vs HIT-UAV** — the one metric that asks whether the output looks
  like *real high-altitude UAV thermal* rather than merely matching one frame.

## 7 · Continuing in the next session

Kaggle kills the session at 12 hours. To carry on:

1. **Save Version** → *Save & Run All*, and wait for it to finish.
2. In a new notebook: **Add data → Notebook Output** → select this run.
3. Copy the checkpoint across and resume:

```python
!mkdir -p /kaggle/working/outputs/pix2pix_uav/checkpoints
!cp /kaggle/input/<your-notebook-output>/outputs/pix2pix_uav/checkpoints/last.pt \\
    /kaggle/working/outputs/pix2pix_uav/checkpoints/

!python scripts/train.py --config configs/pix2pix_uav.yaml \\
    --manifest /kaggle/working/manifest.csv --resume auto
```

The checkpoint carries both networks, both optimisers, both schedulers, the AMP
scaler, the epoch counter and the RNG state, so the loss curve continues without
a visible seam.

Also download `checkpoints/best.pt` before the session ends — Kaggle's working
directory does not survive it.
"""
        ),
    ]


# ---------------------------------------------------------------- notebook 02


def notebook_inference() -> list[dict]:
    return [
        markdown(
            """
# 02 · Inference — RGB → thermal on your own images

Run a trained checkpoint over arbitrary imagery: your own UAV frames, or a
held-out slice of the datasets.

**Point `CHECKPOINT` at a real file first** — either a Notebook Output from
training, or an uploaded Kaggle Dataset containing `best.pt`.
"""
        ),
        code(SETUP_CELL),
        code(INSTALL_CELL),
        code(
            '''
from pathlib import Path

# EDIT THESE TWO
CHECKPOINT = "/kaggle/input/<your-notebook-output>/outputs/pix2pix_uav/checkpoints/best.pt"
INPUT_DIR  = "/kaggle/input/<your-images>"

assert Path(CHECKPOINT).exists(), (
    f"No checkpoint at {CHECKPOINT}.\\n"
    "Add data -> Notebook Output -> pick your training run, then fix the path."
)
print("checkpoint OK:", CHECKPOINT)
'''
        ),
        markdown("## Generate"),
        code(
            '''
!python scripts/predict.py \\
    --checkpoint {CHECKPOINT} \\
    --input {INPUT_DIR} \\
    --out /kaggle/working/predictions \\
    --colormap inferno \\
    --side-by-side
'''
        ),
        code(
            '''
from pathlib import Path
from IPython.display import Image as ShowImage, display

for path in sorted(Path("/kaggle/working/predictions").glob("*.png"))[:8]:
    print(path.name)
    display(ShowImage(filename=str(path), width=900))
'''
        ),
        markdown(
            """
## Options

| Flag | Effect |
|---|---|
| *(none)* | Single-channel grayscale — use this if the output feeds another model |
| `--colormap inferno` | Colourised for human viewing |
| `--side-by-side` | Input next to prediction |
| `--keep-size` | Upscale the 256×256 output back to the input resolution |
| `--domain dronevehicle` | Pick which sensor style to imitate (only if trained with `model.use_domain_embedding: true`) |

## What this model does and does not give you

It predicts **thermal appearance** — the spatial pattern of relative heat.

It does **not** predict temperature. The training pipeline applies a per-image
percentile stretch to every thermal frame, which is exactly what makes three
different thermal cameras trainable as one dataset, and which necessarily
discards absolute radiometric values. A pixel of 200 means "hot relative to this
frame", never "44 °C".

For UAV work that is usually the right trade: augmenting detection or
segmentation training sets, previewing what a thermal payload would see,
prototyping a pipeline before the hardware arrives. It is not a substitute for a
radiometric sensor, and nothing downstream should treat it as one.
"""
        ),
    ]


def main() -> int:
    for name, cells in (
        ("00_explore_datasets.ipynb", notebook_explore()),
        ("01_kaggle_train.ipynb", notebook_train()),
        ("02_kaggle_inference.ipynb", notebook_inference()),
    ):
        path = write_notebook(name, cells)
        # NOTEBOOK_DIR is monkeypatched to a tmp dir by the regeneration test,
        # so do not assume the output lives inside the repo.
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
