import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.fmlayer.data.fewshot import K_VALUES, SEEDS, load_train_subset
from src.fmlayer.data.specs import get_spec
from src.fmlayer.encoders.base import default_device
from src.fmlayer.encoders.registry import LINEAR_PROBE_CELLS
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.linear_probe import build_linear_probe
from src.fmlayer.utils.results import default_results_root, record_run
from src.fmlayer.utils.seeding import set_seed

METHOD = "linear_probe"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
CURVES_DIRNAME = "curves"


def to_tensors(
    features: np.ndarray, labels: np.ndarray, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Move a cached feature/label pair onto the compute device.

    Args:
        features: Feature array of shape ``(num_items, dim)``.
        labels: Integer labels of shape ``(num_items,)``.
        device: Target device.

    Returns:
        The features as float32 and the labels as int64, both on the device.
    """
    return (
        torch.from_numpy(np.ascontiguousarray(features)).float().to(device),
        torch.from_numpy(np.ascontiguousarray(labels)).long().to(device),
    )


@torch.no_grad()
def evaluate_probe(
    model: nn.Linear, features: Tensor, labels: Tensor, criterion: nn.Module
) -> tuple[float, float]:
    """Score a probe on a whole split in one forward pass.

    Args:
        model: The linear classifier.
        features: Features of the split.
        labels: Labels of the split.
        criterion: Loss used for reporting.

    Returns:
        The mean loss and the top-1 accuracy.
    """
    model.eval()
    logits = model(features)
    loss = criterion(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def train_probe(
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    num_classes: int,
    seed: int,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[nn.Linear, list[dict], int]:
    """Train a linear probe and keep the checkpoint with the best validation accuracy.

    Args:
        train_features: Features of the selected training subset.
        train_labels: Labels of the selected training subset.
        val_features: Features of the full validation split.
        val_labels: Labels of the full validation split.
        num_classes: Number of output classes.
        seed: Seed controlling initialisation and batch shuffling.
        device: Device to train on.
        max_epochs: Maximum number of epochs.
        batch_size: Examples per optimisation step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.

    Returns:
        The model restored to its best checkpoint, the per-epoch history and the best epoch.
    """
    model = build_linear_probe(train_features.shape[1], num_classes, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_train = len(train_features)

    history = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(num_train, generator=generator).to(device)
        epoch_loss = 0.0

        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_features[batch]), train_labels[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        train_loss = epoch_loss / num_train
        val_loss, val_accuracy = evaluate_probe(model, val_features, val_labels, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model, history, best_epoch


def curves_path(
    encoder: str, dataset: str, k: int | str, seed: int, results_root: Path | None = None
) -> Path:
    """Build the path of a saved training-curve file.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        Path of the JSON history file.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / CURVES_DIRNAME / f"{encoder}_{dataset}_k{k}_seed{seed}.json"


def run_linear_probe(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    record: bool = True,
    verbose: bool = True,
) -> dict:
    """Train and evaluate one linear probe on cached features.

    For the 5-shot and 10-shot settings the seed selects the training subset; for the
    full setting the subset is fixed and the seed only varies the classifier initialisation.

    Args:
        encoder: Encoder key supplying the cached features.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        subset_root: Subset index directory; defaults to the resolved subset root.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Maximum number of epochs.
        record: Append the result to ``runs.csv``.
        verbose: Print a one-line summary.

    Returns:
        The test accuracy, the best validation accuracy, the best epoch, the per-epoch
        history and the test-set predictions.
    """
    device = device if device is not None else default_device()
    num_classes = get_spec(dataset).num_classes
    set_seed(seed)

    train_features, train_labels, indices = load_train_subset(
        encoder, dataset, k, seed, feature_root, subset_root
    )
    val_features, val_labels, _ = load_split(encoder, dataset, "val", feature_root)
    test_features, test_labels, _ = load_split(encoder, dataset, "test", feature_root)

    train_x, train_y = to_tensors(train_features, train_labels, device)
    val_x, val_y = to_tensors(val_features, val_labels, device)
    test_x, test_y = to_tensors(test_features, test_labels, device)

    model, history, best_epoch = train_probe(
        train_x, train_y, val_x, val_y, num_classes, seed, device, max_epochs
    )

    criterion = nn.CrossEntropyLoss()
    _, test_accuracy = evaluate_probe(model, test_x, test_y, criterion)
    best_val_accuracy = history[best_epoch - 1]["val_accuracy"]

    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1).cpu().numpy()

    path = curves_path(encoder, dataset, k, seed, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "encoder": encoder,
                "dataset": dataset,
                "k": k,
                "seed": seed,
                "num_train": len(indices),
                "best_epoch": best_epoch,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if record:
        record_run(
            {
                "method": METHOD,
                "dataset": dataset,
                "encoder": encoder,
                "k": k,
                "seed": seed,
                "split": "test",
                "accuracy": test_accuracy,
                "num_items": len(test_labels),
            },
            results_root,
        )

    if verbose:
        print(
            f"    {encoder:<14} {dataset:<9} k={str(k):<4} seed={seed}  "
            f"train={len(indices):>5}  val={best_val_accuracy:.4f}@{best_epoch:<3}  "
            f"test={test_accuracy:.4f}"
        )

    return {
        "encoder": encoder,
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "test_accuracy": test_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "num_train": len(indices),
        "history": history,
        "predictions": predictions,
        "labels": test_labels,
    }


def run_all_linear_probes(
    cells: tuple[tuple[str, str], ...] | None = None,
    k_values: tuple[int | str, ...] = K_VALUES,
    seeds: tuple[int, ...] = SEEDS,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    record: bool = True,
) -> dict:
    """Run the full linear-probe grid: 3 cells x 3 training sizes x 3 seeds.

    Args:
        cells: ``(encoder, dataset)`` pairs; defaults to the Stage 1 probe cells.
        k_values: Training-set sizes to sweep.
        seeds: Seeds per training-set size.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        subset_root: Subset index directory; defaults to the resolved subset root.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Maximum number of epochs.
        record: Append the results to ``runs.csv``.

    Returns:
        One result per run, keyed by ``"encoder/dataset/k/seed"``.
    """
    cells = cells if cells is not None else LINEAR_PROBE_CELLS
    device = device if device is not None else default_device()
    print(f"Device: {device}")

    jobs = [
        (encoder, dataset, k, seed)
        for encoder, dataset in cells
        for k in k_values
        for seed in seeds
    ]

    results = {}
    for encoder, dataset, k, seed in tqdm(jobs, desc="linear probes"):
        result = run_linear_probe(
            encoder,
            dataset,
            k,
            seed,
            feature_root,
            subset_root,
            results_root,
            device,
            max_epochs,
            record,
        )
        results[f"{encoder}/{dataset}/{k}/{seed}"] = result

    print(f"\n{len(results)} linear-probe runs complete.")
    return results


def summarize_linear_probes(results: dict) -> dict:
    """Aggregate the run grid into mean and standard deviation per setting.

    Args:
        results: Output of :func:`run_all_linear_probes`.

    Returns:
        Mean accuracy, standard deviation and run count, keyed by ``"encoder/dataset/k"``.
    """
    grouped: dict[str, list[float]] = {}
    for result in results.values():
        key = f"{result['encoder']}/{result['dataset']}/{result['k']}"
        grouped.setdefault(key, []).append(result["test_accuracy"])

    summary = {}
    for key, accuracies in sorted(grouped.items()):
        values = np.asarray(accuracies)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "runs": len(values),
        }
        print(f"{key:<30} {values.mean():.4f} +/- {values.std(ddof=0):.4f}  (n={len(values)})")
    return summary
