from pathlib import Path

import torch
from torch import Tensor, nn

from src.fmlayer.models.linear_probe import build_linear_probe
from src.fmlayer.models.probe_bank import ProbeBank
from src.fmlayer.train.train_linear import MAX_EPOCHS as PROBE_EPOCHS
from src.fmlayer.train.train_linear import train_probe
from src.fmlayer.utils.results import default_results_root

PROBES_DIRNAME = "probes_stage3"

# One probe per (encoder, dataset, k, seed) is shared by every flow configuration.
_PROBE_CACHE: dict[tuple, nn.Linear] = {}


def freeze(probe: nn.Linear) -> nn.Linear:
    """Put a probe into frozen evaluation mode."""
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    probe.eval()
    return probe


def probe_path(
    encoder: str, dataset: str, k: int | str, seed: int, suffix: str, results_root: Path | None = None
) -> Path:
    """Build the checkpoint path of one cached probe."""
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / PROBES_DIRNAME / f"{encoder}_{dataset}_k{k}_seed{seed}{suffix}.pt"


def get_probe(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    num_classes: int,
    device: torch.device,
    results_root: Path | None = None,
) -> nn.Linear:
    """Return the frozen Stage 1 probe for one cell, training it only once.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        train_features: Features of the K-shot training subset.
        train_labels: Labels of the K-shot training subset.
        val_features: Features of the validation split.
        val_labels: Labels of the validation split.
        num_classes: Number of classes.
        device: Device to train on.
        results_root: Results directory holding the probe cache.

    Returns:
        The frozen probe, identical to the Stage 1 baseline for the same arguments.
    """
    key = (encoder, dataset, str(k), seed, str(device))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    path = probe_path(encoder, dataset, k, seed, "", results_root)
    probe = build_linear_probe(train_features.shape[1], num_classes, seed, device)
    if path.is_file():
        probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    else:
        probe, _, _ = train_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_classes,
            seed,
            device,
            max_epochs=PROBE_EPOCHS,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), path)

    _PROBE_CACHE[key] = freeze(probe)
    return _PROBE_CACHE[key]


def load_cached_probe(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    embed_dim: int,
    num_classes: int,
    device: torch.device,
    results_root: Path | None = None,
) -> nn.Linear | None:
    """Load a probe from the disk cache without touching the features.

    Used when rebuilding results from checkpoints, where the training data is not needed.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        embed_dim: Feature dimension.
        num_classes: Number of classes.
        device: Device to load onto.
        results_root: Results directory holding the probe cache.

    Returns:
        The frozen probe, or ``None`` when it has not been cached yet.
    """
    key = (encoder, dataset, str(k), seed, str(device))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    path = probe_path(encoder, dataset, k, seed, "", results_root)
    if not path.is_file():
        return None

    probe = build_linear_probe(embed_dim, num_classes, seed, device)
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    _PROBE_CACHE[key] = freeze(probe)
    return _PROBE_CACHE[key]


def assign_folds(labels: Tensor, num_folds: int, seed: int) -> Tensor:
    """Split samples into class-stratified folds.

    Args:
        labels: Training labels.
        num_folds: Number of folds.
        seed: Seed controlling the shuffle within each class.

    Returns:
        Fold index per sample, shape ``(num_items,)``.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    folds = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
    for class_id in labels.unique():
        members = torch.nonzero(labels == class_id, as_tuple=True)[0]
        shuffled = members[torch.randperm(len(members), generator=generator).to(labels.device)]
        folds[shuffled] = torch.arange(len(shuffled), device=labels.device) % num_folds
    return folds


def get_probe_bank(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    num_classes: int,
    device: torch.device,
    num_folds: int,
    results_root: Path | None = None,
    trainable: bool = False,
) -> tuple[ProbeBank, Tensor | None]:
    """Build the probe bank the flow trains against.

    With ``num_folds == 1`` the bank wraps the evaluation probe itself. With more folds
    each sample is scored by a probe fitted without it, so the classification objectives
    see genuine errors instead of a memorised training set.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        train_features: Features of the K-shot training subset.
        train_labels: Labels of the K-shot training subset.
        val_features: Features of the validation split.
        val_labels: Labels of the validation split.
        num_classes: Number of classes.
        device: Device to train on.
        num_folds: Number of cross-fitting folds; ``1`` disables cross-fitting.
        results_root: Results directory holding the probe cache.

    Returns:
        The bank and the per-sample fold index, which is ``None`` without cross-fitting.
    """
    probe = get_probe(
        encoder, dataset, k, seed, train_features, train_labels,
        val_features, val_labels, num_classes, device, results_root,
    )
    if num_folds <= 1:
        return ProbeBank([probe], trainable), None

    folds = assign_folds(train_labels, num_folds, seed)
    path = probe_path(encoder, dataset, k, seed, f"_x{num_folds}", results_root)
    probes = [build_linear_probe(train_features.shape[1], num_classes, seed, device) for _ in range(num_folds)]

    if path.is_file():
        states = torch.load(path, map_location=device, weights_only=True)
        for fold_probe, state in zip(probes, states):
            fold_probe.load_state_dict(state)
    else:
        probes = []
        for fold in range(num_folds):
            keep = folds != fold
            fold_probe, _, _ = train_probe(
                train_features[keep],
                train_labels[keep],
                val_features,
                val_labels,
                num_classes,
                seed,
                device,
                max_epochs=PROBE_EPOCHS,
            )
            probes.append(fold_probe)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save([fold_probe.state_dict() for fold_probe in probes], path)

    return ProbeBank([freeze(fold_probe) for fold_probe in probes], trainable), folds
