import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "stage2_flow_matching.ipynb"
REPO_URL = "https://github.com/Asaf21S/flow-matching-as-a-layer.git"


def as_source(text: str) -> list[str]:
    """Split a block of text into the line list an ipynb cell expects."""
    lines = text.strip("\n").split("\n")
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def markdown(text: str) -> dict:
    """Build a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": as_source(text)}


def code(text: str) -> dict:
    """Build an unexecuted code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": as_source(text),
    }


def build_cells() -> list[dict]:
    """Assemble every cell of the Stage 2 notebook, in order."""
    return [
        markdown(
            """
# Stage 2 — Flow Matching as a Layer in the CLIP Classifier

Train an unconditional velocity field `v(x, t)` that transports a training image embedding
at t=0 towards its class text embedding at t=1 along the straight conditional-OT path. At
inference, integrate an **unlabelled** test embedding and run the same cosine 1-NN against
the text prototypes as the Stage 1 baseline.

The field never takes a label as input, which is what makes it applicable at test time. It
is zero-initialised at its output layer, so t=0 reproduces the Stage 1 zero-shot number
exactly and the run asserts this.

Only the cached `clip_rn50` features are needed from Stage 1; cell 4 re-creates them if they
are missing, so this notebook runs standalone.
"""
        ),
        markdown("## 1. Clone the repo"),
        code(
            f"""
!git clone {REPO_URL}
%cd flow-matching-as-a-layer
!git checkout asaf/stage2
"""
        ),
        code(
            """
# Re-running a later session? Pull instead of cloning.
# %cd /content/flow-matching-as-a-layer
# !git pull --ff-only
"""
        ),
        markdown(
            """
## 2. Install dependencies

Adds `flow_matching` (Meta's reference probability paths and ODE solvers) and its
`torchdiffeq` backend on top of the Stage 1 requirements.
"""
        ),
        code("!pip install -q -r requirements-colab.txt"),
        markdown(
            """
## 3. Session paths and imports

Keep `USE_DRIVE = True` so the CLIP features are reused across sessions and nothing has to
be re-encoded.
"""
        ),
        code(
            """
import os
import sys
from pathlib import Path

USE_DRIVE = True
DRIVE_ROOT = Path("/content/drive/MyDrive/fmlayer")

REPO_ROOT = Path("/content/flow-matching-as-a-layer")
DATA_ROOT = Path("/content/data")

if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    FEATURE_ROOT = DRIVE_ROOT / "features"
    RESULTS_ROOT = DRIVE_ROOT / "results"
else:
    FEATURE_ROOT = Path("/content/features")
    RESULTS_ROOT = Path("/content/results")

for path in (DATA_ROOT, FEATURE_ROOT, RESULTS_ROOT):
    path.mkdir(parents=True, exist_ok=True)

os.environ["FMLAYER_DATA_ROOT"] = str(DATA_ROOT)
os.environ["FMLAYER_FEATURE_ROOT"] = str(FEATURE_ROOT)
os.environ["FMLAYER_RESULTS_ROOT"] = str(RESULTS_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("features ->", FEATURE_ROOT)
print("results  ->", RESULTS_ROOT)
"""
        ),
        markdown(
            """
### 3b. (Optional) Keep the datasets on Drive too

Datasets default to session-local scratch and are re-downloaded every session. Symlinking
them onto Drive makes the ~3.4 GB download a one-off.
"""
        ),
        code(
            """
import shutil

if USE_DRIVE:
    DRIVE_DATA = DRIVE_ROOT / "data"
    DRIVE_DATA.mkdir(parents=True, exist_ok=True)

    if DATA_ROOT.is_symlink():
        DATA_ROOT.unlink()
    elif DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.symlink_to(DRIVE_DATA)

    os.environ["FMLAYER_DATA_ROOT"] = str(DATA_ROOT)
    print("datasets persist on Drive ->", DRIVE_DATA)
else:
    print("USE_DRIVE is False; datasets stay in session-local scratch.")
"""
        ),
        markdown(
            """
## 4. Check the CLIP features, and rebuild them if missing

Stage 2 touches only the `clip_rn50` caches: `train`, `val`, `test` and the text prototypes
per dataset.
"""
        ),
        code(
            """
from src.fmlayer.features.cache import feature_path

missing = []
for dataset in ("dtd", "aircraft"):
    for split in ("train", "val", "test", None):
        path = feature_path("clip_rn50", dataset, split)
        print(f"{'ok     ' if path.is_file() else 'MISSING'} {path.name}")
        if not path.is_file():
            missing.append((dataset, split))

print(f"\\n{len(missing)} missing cache(s).")
"""
        ),
        code(
            """
# Only runs when something is missing. Downloads the datasets (~3.4 GB) and the CLIP
# checkpoint, then extracts just the CLIP branch.
if missing:
    from src.fmlayer.data.prepare import prepare_datasets
    from src.fmlayer.features.extract import extract_all

    prepare_datasets()
    extract_all(cells=[("clip_rn50", "dtd"), ("clip_rn50", "aircraft")])
    print("CLIP features rebuilt.")
else:
    print("Nothing to do.")
"""
        ),
        markdown(
            """
## 5. Download the datasets

Needed only for the reverse-flow retrieval figure, which decodes real images. Skip it if
cell 4 already downloaded them, or if you do not want that figure.
"""
        ),
        code(
            """
from src.fmlayer.data.prepare import prepare_datasets

report = prepare_datasets()
"""
        ),
        markdown(
            """
## 6. Train the flow-matching layer

Two datasets x 3 seeds, on the full training split. There is no K axis here, because the
CLIP branch had a single baseline run rather than a K sweep.

The objective is the flow-matching regression plus a cross-entropy on the single-step
estimate of the t=1 endpoint. Pure flow matching regresses onto `E[x_1 | x_t]`, which for a
target set of only C prototypes is their posterior-weighted barycentre — the worst possible
place for a cosine 1-NN, since every embedding lands near the centre of the prototype cloud.
The cross-entropy term makes the layer optimise the metric it is actually scored on.

Validation sweeps the whole time grid, not just t=1, and selects the stopping time as well
as the epoch.
"""
        ),
        code(
            """
from src.fmlayer.train.train_flow_clip import run_all_flow_clip, summarize_flow_clip

flow_results = run_all_flow_clip()
summary = summarize_flow_clip(flow_results)
"""
        ),
        code(
            """
# A single run, e.g. the seed the figures default to.
from src.fmlayer.train.train_flow_clip import run_flow_clip

result = run_flow_clip("dtd", seed=0)
print(result["accuracy_t0"], "->", result["accuracy_at_best_time"], "at t =", result["best_time"])
"""
        ),
        markdown(
            """
### Ablations

The knobs that matter, all exposed on `run_flow_clip` and `run_all_flow_clip`:

- `ce_weight=0.0` — pure flow matching, no classification term. This is the configuration
  whose accuracy *fell* below the baseline; keep it as the ablation.
- `target_noise=0.0` — target stays a set of C atoms instead of C small clouds.
- `renormalize=False` — integrate in the ambient space instead of on the unit sphere.
"""
        ),
        code(
            """
pure_flow = run_flow_clip("dtd", seed=0, ce_weight=0.0, record=False, verbose=False)
no_noise = run_flow_clip("dtd", seed=0, target_noise=0.0, record=False, verbose=False)
flat = run_flow_clip("dtd", seed=0, renormalize=False, record=False, verbose=False)

for name, run in [("full", result), ("ce_weight=0", pure_flow), ("no target noise", no_noise), ("no renormalize", flat)]:
    print(f"{name:<18} t=0 {run['accuracy_t0']:.4f}  best {run['accuracy_at_best_time']:.4f} at t={run['best_time']:.1f}  t=1 {run['accuracy_t1']:.4f}")
"""
        ),
        markdown(
            """
## 7. Accuracy as a function of t

How classification changes as the layer moves each embedding from its original position
(t=0, the baseline) towards the class text embeddings.

Reference levels on every panel:

- the **zero-shot baseline**, which the curve must start on at t=0;
- the **constant-shift ablation**, which translates every embedding by the mean training
  displacement. Note this is not a no-op: with unit-norm prototypes the shifted score is
  `(z + m) . t_c = z . t_c + m . t_c`, and the bias `m . t_c` differs per class, so a shared
  translation reorders the similarities. It is a lower bound to clear, not a neutral line;
- the **validation-selected t**, the only point on the curve that is not chosen using test
  data.

The three Euler step counts should lie on top of each other; visible separation means the
sweep is under-resolved.
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_combined_accuracy_vs_t

plot_combined_accuracy_vs_t()
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_accuracy_vs_t

plot_accuracy_vs_t("aircraft", seed=0)
"""
        ),
        markdown(
            """
## 8. Trajectory snapshots in 2D

The PCA basis is fitted once on the t=0 frame and reused for every later frame, so the
panels share one coordinate system and the motion is real rather than a re-projection.

Watch for contraction: if the class clouds merge into one blob by t=1, the transport has
destroyed the separation the cosine 1-NN depends on, and the accuracy curve will say so.
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_trajectory_embeddings

plot_trajectory_embeddings("dtd", seed=0)
plot_trajectory_embeddings("aircraft", seed=0)
"""
        ),
        markdown(
            """
## 9. Reverse flow, rendered by retrieval

Flow matching is time-symmetric, so the same field integrated from t=1 down to t=0 carries
each class *text* embedding back into image-embedding space. Each intermediate point is
rendered as the nearest real training image in cosine similarity — no decoder, and nothing
leaves the CLIP RN50 space the field was trained in.

Read it as "what does this point in embedding space look like", not as generation: the
forward map is many-to-one, so the reverse path from exactly `t_c` is a single deterministic
trajectory towards an average class image. Real generation would need an unCLIP-style
decoder, which is conditioned on CLIP ViT-L/14 rather than RN50.

Needs the datasets (cell 4 or 5).
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_reverse_retrieval

plot_reverse_retrieval("dtd", seed=0)
"""
        ),
        markdown("## 10. Report: table and all figures"),
        code(
            """
from src.fmlayer.report_stage2 import make_stage2_report

report = make_stage2_report()
"""
        ),
        code(
            """
# Skip the image-decoding panels when the datasets were not downloaded this session.
# report = make_stage2_report(with_retrieval=False)

from src.fmlayer.report_stage2 import flow_table, print_flow_table

table = flow_table()
print_flow_table(table)
"""
        ),
        markdown(
            """
The aggregated table lands in `<results>/flow_accuracy_table.csv`. Rows also go to the
shared `runs.csv` under `method = "fm_clip"` with the integration time in the `t` column;
Stage 1 rows carry `t = "none"` and are unaffected.
"""
        ),
    ]


def build_notebook() -> dict:
    """Assemble the full notebook document."""
    return {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path | None = None) -> Path:
    """Write the notebook to disk.

    Args:
        path: Destination ``.ipynb``; defaults to :data:`NOTEBOOK_PATH`.

    Returns:
        The path that was written.
    """
    path = path if path is not None else NOTEBOOK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_notebook(), indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    written = write_notebook()
    print(f"Wrote {written.relative_to(REPO_ROOT)}")

