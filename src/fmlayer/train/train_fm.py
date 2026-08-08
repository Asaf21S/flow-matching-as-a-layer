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
from src.fmlayer.models.flow_matching import compute_cfm_loss
from src.fmlayer.models.stage3 import Stage3, build_stage3
from src.fmlayer.train.train_linear import summarize_linear_probes, to_tensors
from src.fmlayer.utils.results import default_results_root, record_run
from src.fmlayer.utils.seeding import set_seed

METHOD = "stage3"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 64
MAX_EPOCHS = 200
CFM_EPOCHS = 50
CURVES_DIRNAME = "curves_fm"


@torch.no_grad()
def evaluate_fm_probe(
    model: Stage3, features: Tensor, labels: Tensor, criterion: nn.Module
) -> tuple[float, float]:
    """Score a Stage3 model on a split in one forward pass.

    Args:
        model: Stage3 composite model.
        features: Features of the split.
        labels: Labels of the split.
        criterion: Loss used for reporting.

    Returns:
        Mean loss and top-1 accuracy.
    """
    model.eval()
    logits = model(features)
    loss = criterion(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def train_fm_probe(
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    num_classes: int,
    seed: int,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    cfm_epochs: int = CFM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[Stage3, list[dict], int]:
    """Train a Flow Matching Layer + Linear Probe model with checkpointing on validation accuracy.

    First Stage: Train vector field with Conditional Flow Matching (CFM) loss.
    Second Stage: Train downstream classifier layer on flow-transformed representations.

    Args:
        train_features: Training subset features.
        train_labels: Training subset labels.
        val_features: Validation split features.
        val_labels: Validation split labels.
        num_classes: Number of target classes.
        seed: Random seed.
        device: Torch compute device.
        max_epochs: Total epochs for classifier training.
        cfm_epochs: Epochs for CFM vector field pretraining.
        batch_size: Batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.

    Returns:
        Trained Stage3 model, epoch history, best epoch.
    """
    embed_dim = train_features.shape[1]
    model = build_stage3(
        embed_dim=embed_dim,
        num_classes=num_classes,
        seed=seed,
        device=device,
    )

    num_train = len(train_features)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # -------------------------------------------------------------
    # Stage 1: Pretrain Flow Matching Layer with Class-Centroid CFM Loss
    # -------------------------------------------------------------
    centroids = torch.zeros(num_classes, embed_dim, device=device)
    for c in range(num_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            c_mean = train_features[mask].mean(dim=0)
            centroids[c] = nn.functional.normalize(c_mean, p=2, dim=0)
        else:
            centroids[c] = train_features.mean(dim=0)

    cfm_optimizer = torch.optim.AdamW(
        model.fm_layer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    model.fm_layer.train()

    for _ in range(cfm_epochs):
        order = torch.randperm(num_train, generator=generator).to(device)
        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            x0 = train_features[batch]
            x1 = centroids[train_labels[batch]]
            cfm_optimizer.zero_grad(set_to_none=True)
            loss = compute_cfm_loss(model.fm_layer.vector_field, x1=x1, x0=x0)
            loss.backward()
            cfm_optimizer.step()

    # -------------------------------------------------------------
    # Stage 2: Train Classifier Probe on Flow-Transformed Features
    # -------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

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
            logits = model(train_features[batch])
            loss = criterion(logits, train_labels[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        train_loss = epoch_loss / num_train
        val_loss, val_accuracy = evaluate_fm_probe(model, val_features, val_labels, criterion)

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


def curves_path_fm(
    encoder: str, dataset: str, k: int | str, seed: int, results_root: Path | None = None
) -> Path:
    """Build path for FM probe curves file."""
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / CURVES_DIRNAME / f"{encoder}_{dataset}_k{k}_seed{seed}.json"


def run_stage3(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    cfm_epochs: int = CFM_EPOCHS,
    record: bool = True,
    verbose: bool = True,
) -> dict:
    """Train and evaluate one Stage3 run."""
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

    model, history, best_epoch = train_fm_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        num_classes,
        seed,
        device,
        max_epochs=max_epochs,
        cfm_epochs=cfm_epochs,
    )

    criterion = nn.CrossEntropyLoss()
    _, test_accuracy = evaluate_fm_probe(model, test_x, test_y, criterion)
    best_val_accuracy = history[best_epoch - 1]["val_accuracy"]

    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1).cpu().numpy()

    path = curves_path_fm(encoder, dataset, k, seed, results_root)
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
            f"    [FM Layer] {encoder:<14} {dataset:<9} k={str(k):<4} seed={seed}  "
            f"train={len(indices):>5}  val={best_val_accuracy:.4f}@{best_epoch:<3}  "
            f"test={test_accuracy:.4f}"
        )

    return {
        "encoder": encoder,
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "model": model,
        "test_accuracy": test_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "num_train": len(indices),
        "history": history,
        "predictions": predictions,
        "labels": test_labels,
    }


def run_all_stage3(
    cells: tuple[tuple[str, str], ...] | None = None,
    k_values: tuple[int | str, ...] = K_VALUES,
    seeds: tuple[int, ...] = SEEDS,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    cfm_epochs: int = CFM_EPOCHS,
    record: bool = True,
) -> dict:
    """Run full FM Linear Probe grid: 3 cells x 3 training sizes x 3 seeds."""
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
    for encoder, dataset, k, seed in tqdm(jobs, desc="fm linear probes"):
        result = run_stage3(
            encoder,
            dataset,
            k,
            seed,
            feature_root,
            subset_root,
            results_root,
            device,
            max_epochs,
            cfm_epochs,
            record,
        )
        results[f"{encoder}/{dataset}/{k}/{seed}"] = result

    print(f"\n{len(results)} FM linear-probe runs complete.")
    return results
