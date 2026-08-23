# flow-matching-as-a-layer

Implementation of generative Flow Matching (FM) as a structural layer within image
classification networks.

The project is built in stages. **Stage 1** establishes the frozen-encoder classification
baselines and contains no flow matching; **Stage 2** adds the flow-matching layer itself and is
compared against those baselines.

Results write-ups: [`docs/stage1.md`](docs/stage1.md) · [`docs/stage2.md`](docs/stage2.md).

---

## Stage 1 — Classification baselines

A reproducible classification pipeline on frozen pretrained encoders. Features are extracted
once and cached; every classifier then trains and evaluates on those cached vectors.

| | |
|---|---|
| **Datasets** | DTD (official partition 1, 47 classes) · FGVC-Aircraft (variant level, 100 classes) |
| **Encoders** | ResNet-18 (ImageNet-1K, 512-d) · DINOv2 ViT-S/14 (384-d) · CLIP RN50 (1024-d) |
| **Baseline 1** | Linear probe — 3 encoder/dataset cells × K ∈ {5, 10, full} × 3 seeds = **27 runs** |
| **Baseline 2** | Zero-shot CLIP with text prototypes — **2 runs**, no labelled training images |
| **Metric** | Top-1 accuracy on the complete official test split |

Protocol rules enforced in code: encoders stay frozen, official splits are never merged,
validation is used only for model selection, the test split only for the final number, and
K-shot subsets are balanced per class and shared across encoders.

---

## Stage 2 — Flow matching as a layer

A small velocity field `v(z, t)` transports a frozen CLIP image embedding onto the frozen CLIP
**text** prototype of its class. At inference the field is applied to an unlabelled embedding
for T Euler steps, and the transported embedding is classified with the same cosine 1-NN rule
as the Stage 1 zero-shot baseline. The field never sees a label, which is what makes it
applicable at test time.

| | |
|---|---|
| **Objective 1** | Standard FM — `L = ‖v(z_t, t) − (p_y − z_i)‖²` on the straight conditional-OT path |
| **Objective 2** | Rolled-out FM — `L = ‖z_T − p_y‖²`, backpropagated through all T Euler steps |
| **Inference** | `z_{k+1} = z_k + (1/T)·v(z_k, k/T)`, T ∈ {4, 12}, then cosine 1-NN on `z_T` |
| **Field** | MLP 1025 → 512 → 512 → 1024, SiLU, scalar t concatenated, zero-initialised output |
| **Protocol** | K ∈ {5, 10, full} × 3 seeds, reusing the Stage 1 subsets — **54 trained fields** |

Headline: +26.6 points on DTD (0.4005 → 0.6668) and +17.5 points on FGVC-Aircraft
(0.1545 → 0.3299) over the zero-shot baseline, with the encoder and the prototypes both frozen.
Both come from rolled-out FM at T = 4, K = full.

---

## What is implemented

```
src/fmlayer/
  data/
    specs.py        Dataset registry: class counts, official split sizes, prompt templates,
                    protocol kwargs (DTD partition=1, Aircraft annotation_level=variant),
                    and data-root resolution.
    datasets.py     torchvision wrappers, label/class-name access, and verify_split(), which
                    asserts a materialised split matches the protocol.
    class_names.py  Minimal, documented class-name normalisation and CLIP prompt building.
    prepare.py      Entry point: download both datasets and verify all six splits.
    fewshot.py      Balanced K-shot sampling with seeds {0,1,2}; indices persisted as .npy so
                    every encoder trains on exactly the same images.

  encoders/
    base.py         Encoder ABC. Freezes the module (eval + requires_grad_(False)), binds it to
                    its own eval preprocessing, validates the output width.
    resnet18.py     torchvision IMAGENET1K_V1 with fc -> Identity, 512-d penultimate features.
    dinov2.py       ViT-S/14 from torch.hub, final class token, official 224 eval transform.
    clip_rn50.py    open_clip RN50/openai; image tower plus embed_texts() for prompts.
    registry.py     build_encoder(), encoder/dataset pairing, the linear-probe cells.
    check.py        Entry point: smoke-tests shapes, dtype and that zero parameters are trainable.

  features/
    cache.py        .npz cache paths, save/load, and a config hash covering encoder, dataset,
                    split, embed dim, item count and the stringified transform, so stale caches
                    are detected automatically.
    extract.py      Entry point: one forward pass per split (shuffle=False keeps features aligned
                    with labels), asserts counts against the official split sizes, and caches
                    15 image files plus 2 CLIP text-prototype files. Each encoder is loaded once.

  models/
    prototypes.py   L2 normalisation, image prototypes = normalize(mean(normalize(z))),
                    cosine similarity and nearest-prototype classification.
    linear_probe.py nn.Linear(D, C), i.e. s = Wz + b, seeded initialisation.
    zeroshot.py     Entry point: the zero-shot CLIP baseline.
    flow_matching_clip.py
                    Stage 2 velocity field (2 x 512 SiLU MLP, scalar t concatenated, zero-init
                    output so the untrained field is the identity), the conditional-OT path and
                    the optional endpoint cross-entropy helpers.
    flow_ode.py     The brief's Euler rule as a differentiable rollout (shared by rolled-out
                    training and inference), plus an arbitrary-time-grid integrator used for the
                    diagnostic sweep and the reverse flow.
    retrieval.py    Cosine nearest-neighbour lookup, used to render reverse-flow states.

  train/
    train_linear.py Entry point: AdamW (lr 1e-3, wd 1e-4), batch 64, up to 200 epochs, checkpoint
                    selected by highest validation accuracy, then evaluated on the test split.
                    Saves per-epoch histories and aggregates mean +/- std per setting.
    train_flow_clip.py
                    Entry point: trains the FM layer under either objective. AdamW (lr 1e-3
                    cosine-annealed to 1e-5, wd 1e-4), batch 256, 1000 epochs, checkpoint
                    selected by validation accuracy of the T-step rollout. Sweeps K x seeds x T
                    and records one row per T.
    evaluate.py     Top-1 accuracy, per-class accuracy, row-normalised confusion matrix.

  viz/
    accuracy.py     Accuracy vs K with error bars; zero-shot drawn as a horizontal reference line.
    curves.py       Train/validation loss and validation accuracy, marking both the selected
                    epoch and the minimum-validation-loss epoch.
    confusion.py    Row-normalised confusion matrices for the probe and for zero-shot.
    embeddings.py   PCA / t-SNE of test features with their prototypes, projection fitted jointly,
                    fixed classes and colours so encoders are directly comparable.
    flow_clip.py    Stage 2 figures: accuracy vs K per variant, training curves, the three-way
                    feature comparison under one joint PCA, per-example flow trajectories, the
                    accuracy-along-t diagnostic and the reverse-flow retrieval grid.
    figures.py      Figure directory resolution and saving.

  utils/
    results.py      Tidy runs.csv. Re-running a run replaces its row instead of duplicating it.
    seeding.py      Seeds Python, NumPy and torch.

  report.py         Entry point: accuracy table (ordered K = 5, 10, full) plus every figure.
  report_stage2.py  Entry point: Stage 2 table grouped by (dataset, objective, T, K) with mean,
                    std and ΔAcc against the baseline, plus every Stage 2 figure.
```

Training runs on cached feature vectors held entirely in GPU memory, with batches cut by a
seeded `randperm` rather than a `DataLoader`, so each of the 27 probe runs takes seconds. The
54 Stage 2 fields train on the same cached vectors and take seconds to a couple of minutes each.

---

## Running it

All compute happens on **Google Colab**. The repository is cloned into a session, datasets are
downloaded per session, and every step runs there. Datasets, cached features and results are
never committed.

- Stage 1: `notebooks/classifiers_baselines.ipynb` — downloads the datasets, extracts features,
  trains and evaluates the linear probes and zero-shot baseline, and generates every figure.
- Stage 2: `notebooks/stage2_flow_matching.ipynb` — needs only the cached `clip_rn50` features
  and rebuilds them if they are missing, so it runs standalone. Regenerate it from
  `tools/build_stage2_notebook.py` rather than editing the `.ipynb` by hand.

---

## Outputs

```
<results>/runs.csv                  one row per run: method, dataset, encoder, K, seed, steps, accuracy
<results>/accuracy_table.csv        mean +/- std per method/dataset/encoder/K
<results>/flow_accuracy_table.csv   Stage 2: mean, std and ΔAcc per dataset/objective/T/K
<results>/curves/*.json             per-epoch train loss, val loss, val accuracy
<results>/flow_curves/*.json        Stage 2 per-run history, accuracies per T and diagnostic sweep
<results>/flow_ckpt/*.pt            trained velocity fields, so figures need no retraining
<results>/figures/*.png             accuracy vs K, loss curves, confusion matrices, embeddings
<features>/<encoder>/*.npz          cached frozen-encoder features
<features>/subsets/*.npy            persisted K-shot indices
```

---

## Conventions

- Imports are absolute from the repository root: `from src.fmlayer.data.specs import get_spec`.
- No CLI scripts and no YAML configs: settings are Python constants next to the code that uses
  them, and each pipeline step exposes one entry-point function to call from a notebook cell.
