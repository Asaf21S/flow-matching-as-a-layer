import torch
from torch import Tensor, nn

from src.fmlayer.models.probe_bank import ProbeBank

CENTROIDS = "centroids"
PROBE_WEIGHTS = "probe_weights"
MARGIN = "margin"
NO_TARGET = "none"
TARGET_TYPES = (CENTROIDS, PROBE_WEIGHTS, MARGIN)

# Distance the margin target pushes a point past the decision boundary, as a
# fraction of the mean feature norm.
DEFAULT_MARGIN_RATIO = 0.10
EPSILON = 1e-8


def class_centroids(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    """Mean training feature of each class, falling back to the global mean.

    Args:
        features: Training features, shape ``(num_items, dim)``.
        labels: Training labels, shape ``(num_items,)``.
        num_classes: Number of classes.

    Returns:
        Centroid table of shape ``(num_classes, dim)``.
    """
    sums = torch.zeros(num_classes, features.shape[1], device=features.device)
    sums.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=num_classes).unsqueeze(1).float()
    global_mean = features.mean(dim=0, keepdim=True)
    return torch.where(counts > 0, sums / counts.clamp_min(1.0), global_mean)


def probe_weight_targets(probe: nn.Linear, features: Tensor) -> Tensor:
    """Probe weight directions rescaled to the mean feature norm.

    Args:
        probe: The frozen evaluation probe supplying the directions.
        features: Training features used to set the target norm.

    Returns:
        Target table of shape ``(num_classes, dim)``.
    """
    directions = torch.nn.functional.normalize(probe.weight.detach(), p=2, dim=1)
    return directions * features.norm(p=2, dim=1).mean()


class ClassTargets:
    """Per-class target table, looked up by label."""

    def __init__(self, table: Tensor):
        self.table = table

    def __call__(self, source: Tensor, labels: Tensor, folds: Tensor | None = None) -> Tensor:
        """Return the target of each sample's class."""
        return self.table[labels]


class MarginTargets:
    """Smallest move that puts a point the required distance past the boundary.

    The target is built against the runner-up class, so a point that is already
    correct by more than ``margin_distance`` is its own target and the flow is asked
    to leave it alone. Only points the probe gets wrong, or gets right by too little,
    carry a non-zero velocity.
    """

    def __init__(self, bank: ProbeBank, margin_distance: float):
        self.bank = bank
        self.margin_distance = margin_distance

    def __call__(self, source: Tensor, labels: Tensor, folds: Tensor | None = None) -> Tensor:
        """Return the margin-corrected position of each sample."""
        with torch.no_grad():
            logits = self.bank.logits(source, folds)
            blocked = torch.nn.functional.one_hot(labels, logits.shape[1]).bool()
            runner_up = logits.masked_fill(blocked, float("-inf")).argmax(dim=1)

            true_weight, true_bias = self.bank.rows(labels, folds)
            rival_weight, rival_bias = self.bank.rows(runner_up, folds)

            direction = true_weight - rival_weight
            norm = direction.norm(p=2, dim=1).clamp_min(EPSILON)
            gap = (source * direction).sum(dim=1) + (true_bias - rival_bias)
            # Signed distance to the {true vs runner-up} boundary.
            shortfall = (self.margin_distance - gap / norm).clamp_min(0.0)
            return source + (shortfall / norm).unsqueeze(1) * direction


def build_target_provider(
    target_type: str,
    features: Tensor,
    labels: Tensor,
    bank: ProbeBank,
    probe: nn.Linear,
    num_classes: int,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
) -> ClassTargets | MarginTargets:
    """Build the target provider named by ``target_type``.

    Args:
        target_type: One of :data:`TARGET_TYPES`.
        features: Training features the targets are derived from.
        labels: Training labels.
        bank: Probe bank the margin target is measured against, per sample.
        probe: The single evaluation probe, used for class-level targets.
        num_classes: Number of classes.
        margin_ratio: Margin distance as a fraction of the mean feature norm.

    Returns:
        A callable mapping ``(source, labels, folds)`` to per-sample targets.
    """
    if target_type == CENTROIDS:
        return ClassTargets(class_centroids(features, labels, num_classes))
    if target_type == PROBE_WEIGHTS:
        return ClassTargets(probe_weight_targets(probe, features))
    if target_type == MARGIN:
        distance = margin_ratio * features.norm(p=2, dim=1).mean().item()
        return MarginTargets(bank, distance)
    raise ValueError(f"Unknown target_type {target_type!r}. Available: {sorted(TARGET_TYPES)}")
