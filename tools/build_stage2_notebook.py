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

Train an unconditional velocity field `v(z, t)` that transports a training image embedding
at t=0 towards its frozen class text prototype at t=1 along the straight conditional-OT
path. At inference, run T Euler steps of size 1/T on an **unlabelled** test embedding and
apply the same cosine 1-NN against the text prototypes as the Stage 1 baseline.

The field never takes a label as input, which is what makes it applicable at test time. It
is a small MLP — two 512-wide hidden layers, SiLU, the scalar t simply concatenated to the
input — and its output layer is zero-initialised, so an untrained field is the identity map
and t=0 reproduces the Stage 1 zero-shot number exactly. Each diagnostic run asserts this.

Two objectives are compared at T = 4 and T = 12, across K in {5, 10, full} and three seeds:
the standard flow-matching velocity regression, and rolled-out training that backpropagates
through the whole T-step Euler loop and supervises only the endpoint.

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

The grid follows the brief: two datasets x K in {5, 10, full} x seeds {0, 1, 2}, reusing the
Stage 1 K-shot subsets so the comparison is like-for-like, and two objectives.

- **Standard FM** regresses the velocity at a random point on the straight path,
  `L = ||v(z_t, t) - (p_y - z_i)||^2`. It does not depend on T, so one field is trained per
  (dataset, K, seed) and scored at both T = 4 and T = 12.
- **Rolled-out FM** runs the same T-step Euler loop used at inference and supervises only
  where it lands, `L = ||z_T - p_y||^2`, backpropagating through all T calls. T is baked in,
  so it needs one field per T.

Architecture, optimiser, epochs and batch size are identical across the two. Both are scored
the same way: transport the test feature with T Euler steps of size 1/T, then cosine 1-NN on
the endpoint `z_T`.

54 trainings in total, all on cached features, so each is seconds to a couple of minutes.
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
# A single run, e.g. the configuration the figures default to.
from src.fmlayer.train.train_flow_clip import ROLLED, STANDARD, run_flow_clip

standard = run_flow_clip("dtd", STANDARD, k="full", seed=0)
rolled = run_flow_clip("dtd", ROLLED, k="full", seed=0, steps=12)

print("baseline", standard["baseline_accuracy"])
print("standard", standard["accuracy_by_steps"])
print("rolled  ", rolled["accuracy_by_steps"])
"""
        ),
        markdown(
            """
### Extensions (not part of the brief)

`run_flow_clip` and `run_all_flow_clip` keep two knobs that are **off by default**, so the
headline numbers are the assignment's bare regression. They exist for the discussion section
only:

- `ce_weight > 0` adds a cross-entropy on the single-step estimate of the t=1 endpoint, i.e.
  it optimises the metric the layer is scored on rather than the velocity.
- `target_noise > 0` turns the t=1 end into a small cloud around each prototype instead of
  one of only C atoms.

The sphere-projection variant has been removed entirely: the only Euler rule in the codebase
is now the brief's `z[k+1] = z[k] + (1/T) v(z[k], k/T)`.
"""
        ),
        code(
            """
extended = run_flow_clip(
    "dtd", STANDARD, k="full", seed=0, ce_weight=1.0, record=False, verbose=False
)

for name, run in [("brief (L_FM)", standard), ("+ endpoint CE", extended)]:
    scores = "  ".join(f"T={t} {a:.4f}" for t, a in run["accuracy_by_steps"].items())
    print(f"{name:<16} base {run['baseline_accuracy']:.4f}   {scores}")
"""
        ),
        markdown(
            """
## 7. Deliverable 1 — accuracy versus K

Every variant against the Stage 1 prototype baseline, with standard-deviation bars over the
three seeds. Our baseline is zero-shot CLIP, which never sees the training subset, so it is a
**flat line** and `dAcc(K) = Acc_FM(K) - Acc_zeroshot` is measured against a constant. For
the linear-probe branch it would rise with K instead.
"""
        ),
        code(
            """
from src.fmlayer.report_stage2 import flow_table, print_flow_table
from src.fmlayer.viz.flow_clip import plot_accuracy_vs_k

table = flow_table()
print_flow_table(table)
plot_accuracy_vs_k(table)
"""
        ),
        markdown(
            """
## 8. Deliverable 2 — training-loss curves

Standard and rolled-out training are shown in **separate panels** on purpose: one is a
velocity MSE against `p_y - z_i`, the other an endpoint MSE against `p_y`. They are not on
the same scale, so overlaying them would be meaningless.
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_training_curves

plot_training_curves("dtd")
plot_training_curves("aircraft")
"""
        ),
        markdown(
            """
## 9. Deliverable 3 — feature space before and after the layer

Original features, after standard FM, and after rolled-out FM. The PCA is fitted **once**
over all three feature sets and the prototypes together and then applied to each, so the
panels share one coordinate system and the motion between them is real.

Watch for contraction: if the class clouds collapse onto the prototype barycentre, the
cosine 1-NN loses the separation it depends on, which is exactly what the accuracy table
should then show.
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_feature_comparison

plot_feature_comparison("dtd")
plot_feature_comparison("aircraft")
"""
        ),
        markdown(
            """
## 10. Deliverable 4 — flow trajectories for individual examples

A handful of test features traced through the T Euler steps: circle = original feature,
every intermediate state marked, square = transported endpoint, star = the class prototype.
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_flow_trajectories

plot_flow_trajectories("dtd", ROLLED)
plot_flow_trajectories("dtd", STANDARD)
"""
        ),
        markdown(
            """
## 11. Diagnostic — accuracy along t

Not a deliverable: the brief scores the endpoint `z_T`. This finely-integrated sweep exists
to show *whether* accuracy peaks before t=1, which is what motivates rolled-out training. It
needs a run recorded with `with_curve=True`; that run also asserts that t=0 reproduces the
zero-shot baseline, since the field's output layer is zero-initialised.
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_accuracy_vs_t

run_flow_clip("dtd", STANDARD, k="full", seed=0, with_curve=True, record=False)
plot_accuracy_vs_t("dtd", STANDARD)
"""
        ),
        markdown(
            """
## 12. Optional — reverse flow, rendered by retrieval

Flow matching is time-symmetric, so the same field integrated from t=1 down to t=0 carries
each class *text* embedding back into image-embedding space. Each intermediate point is
rendered as the nearest real training image in cosine similarity — no decoder, and nothing
leaves the CLIP RN50 space the field was trained in.

Read it as "what does this point in embedding space look like", not as generation: the
forward map is many-to-one, so the reverse path from exactly `p_c` is a single deterministic
trajectory towards an average class image. Real generation would need an unCLIP-style
decoder, which is conditioned on CLIP ViT-L/14 rather than RN50.

Needs the datasets (cell 4 or 5).
"""
        ),
        code(
            """
from src.fmlayer.viz.flow_clip import plot_reverse_retrieval

plot_reverse_retrieval("dtd", ROLLED)
"""
        ),
        markdown("## 13. Report: table and all figures in one call"),
        code(
            """
from src.fmlayer.report_stage2 import make_stage2_report

# Pass with_retrieval=True to add the image-decoding panels (needs the datasets).
report = make_stage2_report(with_retrieval=False)
"""
        ),
        markdown(
            """
The aggregated table lands in `<results>/flow_accuracy_table.csv`. Rows also go to the
shared `runs.csv` under `method = "fm_clip_standard"` or `"fm_clip_rolled"`, with T in the
`steps` column; Stage 1 rows carry `steps = "none"` and are unaffected.
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

