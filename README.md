# flow-matching-as-a-layer
Implementation of generative Flow Matching (FM) as a structural layer within image classification networks.

## Stage 1 — Classification baselines

Plan: [`docs/PLAN_stage1.md`](docs/PLAN_stage1.md) · Spec: [`docs/cv_project_stage_1.pdf`](docs/cv_project_stage_1.pdf)

| | |
|---|---|
| Datasets | DTD (partition 1, 47 classes), FGVC-Aircraft (variant, 100 classes) |
| Encoders | ResNet-18 (both datasets), DINOv2 ViT-S/14 (DTD), CLIP RN50 (zero-shot branch) |
| Baselines | Linear probe (K ∈ {5, 10, full} × 3 seeds) + zero-shot CLIP |

## How this project is run

**All compute happens on Google Colab.** The repository is cloned into a Colab session,
the datasets are downloaded per session into `/content/data`, and every training,
evaluation and visualization step runs there. Datasets, cached features and results are
never committed.

All logic lives in `src/`, so the notebook cells stay short. Ready-to-paste cells are in
[`docs/colab_cells.md`](docs/colab_cells.md).

### Environments

| File | Where | Contains |
|---|---|---|
| `requirements-colab.txt` | Google Colab | only `open_clip_torch`; torch & friends are preinstalled |
| `requirements-dev.txt` | your laptop | `pypdf` — **no torch, no datasets** |

There is deliberately no `requirements.txt`: the full stack (torch, torchvision, numpy,
scikit-learn, matplotlib, pandas) is whatever Colab provides, and pinning it locally would
only invite IDE warnings for packages this machine should not have.

Installs the lightweight local authoring dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### Step 2 — download and verify the datasets (on Colab)

Downloads both datasets and asserts the official split sizes and class counts:

```python
from src.fmlayer.data.prepare import prepare_datasets

report = prepare_datasets()            # or prepare_datasets(["dtd"])
```

The data root resolves to `$FMLAYER_DATA_ROOT` → `/content/data` on Colab → `<repo>/data`.
The returned report holds the per-split summaries, class names and CLIP prompts.

## Layout

```
src/fmlayer/       all logic, imported as src.fmlayer.*
docs/              plan, spec, Colab cells
tools/             local helper scripts
```
