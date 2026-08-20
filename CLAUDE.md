# CLAUDE.md

Working agreements for this repository. Read before making any change.

---

## Git workflow — not negotiable

**Never push to `main`. Never push directly to `dev`.**

Every change goes:

```
feat/<topic>  or  fix/<topic>        <- branch off dev, do the work here
        |
        v  merge request
      dev                            <- integration branch
        |
        v  merge request
      main                           <- stable; what the Kaggle notebooks clone
```

Concretely, for every task:

```bash
git fetch origin
git checkout -b feat/<topic> origin/dev     # or fix/<topic>
# ... work, commit ...
git push -u origin feat/<topic>
```

Then open a merge request into `dev` and stop. Do not merge it yourself, and do
not fast-forward `main` "because it is only a fast-forward" — that decision
belongs to the repository owner.

After pushing a branch, hand over the comparison URL so the MR can be opened in
one click:

```
https://github.com/Astrq23/rgb-2-thermal/compare/dev...<branch>
```

`gh` is not installed on this machine, so the MR cannot be opened from the CLI.
Push the branch and give the link.

**Kaggle consequence.** The notebooks clone `main` (`BRANCH` in
`scripts/build_notebooks.py`). Work merged only as far as `dev` will *not* reach
a Kaggle session until `dev` is merged into `main`. To test unreleased work, set
`BRANCH = "dev"` in the notebook cell rather than pushing to `main`.

---

## What this is

RGB → thermal image translation for UAV imagery. Pix2Pix trained on four Kaggle
datasets normalised into one schema, designed to run inside a single Kaggle
session. User-facing docs are `README.md` and `DATASETS.md`.

---

## Commands

```bash
pytest -q                                   # full suite; needs no dataset and no GPU
python scripts/build_notebooks.py           # regenerate notebooks after editing cells
python scripts/inspect_datasets.py          # first thing to run on Kaggle
python scripts/build_manifest.py
python scripts/check_alignment.py --dataset flir_v2
python scripts/train.py --config configs/smoke.yaml
```

---

## Where to change what

The data path is four layers; changing the wrong one is the usual mistake.

| Layer | File | Change it when |
|---|---|---|
| Modality detection | `data/adapters/modality.py` | A folder-naming scheme is unrecognised **everywhere** |
| Pair discovery | `configs/datasets/*.yaml` | One dataset's layout differs — **prefer this** |
| Adapter code | `data/adapters/*.py` | A quirk cannot be expressed declaratively (only FLIR so far) |
| Pixel handling | `data/normalize.py` | Geometry or intensity handling changes |

**A mirror with an unexpected layout is a YAML change, not a Python change.**
`thermal_tokens`, `rgb_tokens`, `key_regex`, `force_modality` and
`dataset_from_regex` exist precisely so no new adapter class is needed.

Model architecture is reached only through `models/build.py`; the trainer never
learns about a new architecture directly.

---

## Conventions

- **Language.** Code, docstrings, comments, commit messages and CLI output in
  English. `README.md` and `DATASETS.md` in Vietnamese, since the owner reads
  them most.
- **Notebooks are generated.** Edit `scripts/build_notebooks.py`, never the
  `.ipynb`. A test asserts the committed notebooks match the generator.
- **Configs are strict.** An unknown key raises rather than being ignored — a
  silent `epocs: 100` would waste a GPU session.
- **Tests must not need data.** `tests/conftest.py` builds miniature copies of
  each real layout with PIL. Keep it that way; a test that needs a download is
  a test nobody runs.

---

## Traps this repository has already fallen into

**`.gitignore` rules must be root-anchored.** An unanchored `data/` also matched
`src/rgb2thermal/data/` and silently excluded the entire data layer from a push.
The full suite stayed green because tests read the working tree while the
notebooks clone from GitHub. `tests/test_packaging.py` guards this now — do not
weaken it, and write `/data/`, never `data/`.

**Backslash continuations do not survive shell heredocs.** Writing a code block
containing `\` + newline through a bash heredoc into Python collapses the line.
Use the `Write` tool for such files, or build the string with `chr(92)`.

**Regexes in YAML belong in single quotes.** `'...\d+\.'` keeps the backslash
literal; `"...\d+\."` is a YAML escape error.

**Metric and weight breakdowns depend on the `dataset` column.** Anything that
rewrites paths (`scripts/export_subset.py`) must preserve the source dataset
name and the scene `group_key`, or the sampling weights and the per-dataset
metrics go quietly wrong while training still appears to work.

**Splits are grouped, never per-image.** These datasets are sampled video;
neighbouring frames are near-duplicates. Any new split logic must keep a
`group_key` inside exactly one split.
