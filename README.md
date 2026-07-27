# flow-matching-as-a-layer

Implementation of generative Flow Matching (FM) as a structural layer within image
classification networks.

The project is built in stages. **Stage 1, implemented here, contains no flow matching**: it
establishes the frozen-encoder classification baselines that later stages are compared against.

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

  train/
    train_linear.py Entry point: AdamW (lr 1e-3, wd 1e-4), batch 64, up to 200 epochs, checkpoint
                    selected by highest validation accuracy, then evaluated on the test split.
                    Saves per-epoch histories and aggregates mean +/- std per setting.
    evaluate.py     Top-1 accuracy, per-class accuracy, row-normalised confusion matrix.

  viz/
    accuracy.py     Accuracy vs K with error bars; zero-shot drawn as a horizontal reference line.
    curves.py       Train/validation loss and validation accuracy, marking both the selected
                    epoch and the minimum-validation-loss epoch.
    confusion.py    Row-normalised confusion matrices for the probe and for zero-shot.
    embeddings.py   PCA / t-SNE of test features with their prototypes, projection fitted jointly,
                    fixed classes and colours so encoders are directly comparable.
    figures.py      Figure directory resolution and saving.

  utils/
    results.py      Tidy runs.csv. Re-running a run replaces its row instead of duplicating it.
    seeding.py      Seeds Python, NumPy and torch.

  report.py         Entry point: accuracy table (ordered K = 5, 10, full) plus every figure.
```

Training runs on cached feature vectors held entirely in GPU memory, with batches cut by a
seeded `randperm` rather than a `DataLoader`, so each of the 27 probe runs takes seconds.

---

## Running it

All compute happens on **Google Colab**. The repository is cloned into a session, datasets are
downloaded per session, and every step runs there. Datasets, cached features and results are
never committed.

the notebook for stage 1 is `notebooks/classifiers_baseline.ipynb`. It contains all the code to download datasets, extract features, train and evaluate linear probes, and generate figures.

---

## Outputs

```
<results>/runs.csv              one row per run: method, dataset, encoder, K, seed, accuracy
<results>/accuracy_table.csv    mean +/- std per method/dataset/encoder/K
<results>/curves/*.json         per-epoch train loss, val loss, val accuracy
<results>/figures/*.png         accuracy vs K, loss curves, confusion matrices, embeddings
<features>/<encoder>/*.npz      cached frozen-encoder features
<features>/subsets/*.npy        persisted K-shot indices
```

---

## Conventions

- Imports are absolute from the repository root: `from src.fmlayer.data.specs import get_spec`.
- No CLI scripts and no YAML configs: settings are Python constants next to the code that uses
  them, and each pipeline step exposes one entry-point function to call from a notebook cell.
