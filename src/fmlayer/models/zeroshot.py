from pathlib import Path

import numpy as np

from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.prototypes import classify, image_prototypes
from src.fmlayer.train.evaluate import top1_accuracy
from src.fmlayer.utils.results import record_run

METHOD = "clip_zeroshot"
TEST_SPLIT = "test"


def run_zeroshot(
    dataset: str,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    record: bool = True,
) -> dict:
    """Classify the test split with CLIP text prototypes and score top-1 accuracy.

    Uses no labelled training images, so there is exactly one result per dataset.

    Args:
        dataset: Dataset key, ``"dtd"`` or ``"aircraft"``.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        record: Append the result to ``runs.csv``.

    Returns:
        The accuracy, the predictions and the ground-truth labels.
    """
    spec = get_spec(dataset)
    encoder = ClipRN50Encoder.NAME

    features, labels, _ = load_split(encoder, dataset, TEST_SPLIT, feature_root)
    prototypes, _, text_meta = load_split(encoder, dataset, None, feature_root)

    if len(prototypes) != spec.num_classes:
        raise ValueError(
            f"{dataset}: {len(prototypes)} text prototypes for {spec.num_classes} classes"
        )

    predictions = classify(features, prototypes)
    accuracy = top1_accuracy(predictions, labels)

    print(
        f"[{spec.display_name}] zero-shot CLIP RN50: {accuracy:.4f} top-1 "
        f"on {len(labels)} test images"
    )
    print(f"    prompt example: {text_meta['prompts'][0]!r}")

    if record:
        record_run(
            {
                "method": METHOD,
                "dataset": dataset,
                "encoder": encoder,
                "k": "none",
                "seed": "none",
                "split": TEST_SPLIT,
                "accuracy": accuracy,
                "num_items": len(labels),
            },
            results_root,
        )

    return {
        "dataset": dataset,
        "accuracy": accuracy,
        "predictions": predictions,
        "labels": labels,
        "prompts": text_meta["prompts"],
    }


def run_zeroshot_all(
    datasets: list[str] | None = None,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    record: bool = True,
) -> dict:
    """Run the zero-shot baseline on several datasets.

    Args:
        datasets: Dataset keys; defaults to every Stage 1 dataset.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        record: Append the results to ``runs.csv``.

    Returns:
        One result per dataset, keyed by dataset key.
    """
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)
    return {
        dataset: run_zeroshot(dataset, feature_root, results_root, record)
        for dataset in datasets
    }


def image_prototype_accuracy(
    dataset: str,
    encoder: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    feature_root: Path | None = None,
) -> float:
    """Score image-derived prototypes on the test split.

    Not a Stage 1 baseline under Option B; kept for the feature visualisations and
    as a sanity check against the zero-shot numbers.

    Args:
        dataset: Dataset key.
        encoder: Encoder key whose cached test features are used.
        train_features: Features of the selected training subset.
        train_labels: Labels of the selected training subset.
        feature_root: Feature cache directory; defaults to the resolved feature root.

    Returns:
        Top-1 accuracy on the full test split.
    """
    spec = get_spec(dataset)
    prototypes = image_prototypes(train_features, train_labels, spec.num_classes)
    features, labels, _ = load_split(encoder, dataset, TEST_SPLIT, feature_root)
    return top1_accuracy(classify(features, prototypes), labels)
