import torch
from torch import nn

from src.fmlayer.models.flow_ode import rollout
from src.fmlayer.models.probe_bank import ProbeBank
from src.fmlayer.models.targets import MARGIN, MarginTargets, build_target_provider, class_centroids
from src.fmlayer.models.velocity_field import build_velocity_field, feature_statistics
from src.fmlayer.train.probes import assign_folds

TOLERANCE = 1e-4


def make_probe(num_classes: int, embed_dim: int, seed: int, device: torch.device) -> nn.Linear:
    """Build a small randomly initialised probe for the checks."""
    torch.manual_seed(seed)
    return nn.Linear(embed_dim, num_classes).to(device)


def check_probe_bank(device: torch.device) -> None:
    """A one-probe bank must match the probe, and folds must select the right probe."""
    features = torch.randn(32, 16, device=device)
    probes = [make_probe(5, 16, seed, device) for seed in range(3)]

    single = ProbeBank([probes[0]])
    assert torch.allclose(single.logits(features), probes[0](features), atol=TOLERANCE)

    bank = ProbeBank(probes)
    folds = torch.randint(0, 3, (32,), device=device)
    selected = bank.logits(features, folds)
    for index in range(3):
        mask = folds == index
        assert torch.allclose(selected[mask], probes[index](features)[mask], atol=TOLERANCE)

    labels = torch.randint(0, 5, (32,), device=device)
    weight, bias = bank.rows(labels, folds)
    for index in range(3):
        mask = folds == index
        assert torch.allclose(weight[mask], probes[index].weight[labels[mask]], atol=TOLERANCE)
        assert torch.allclose(bias[mask], probes[index].bias[labels[mask]], atol=TOLERANCE)
    print("  probe bank            OK")


def check_margin_targets(device: torch.device) -> None:
    """Confident points must be left alone; the rest must land exactly on the margin."""
    features = torch.randn(256, 16, device=device)
    labels = torch.randint(0, 5, (256,), device=device)
    bank = ProbeBank([make_probe(5, 16, 0, device)])
    distance = 0.5
    targets = MarginTargets(bank, distance)(features, labels)

    logits = bank.logits(features)
    blocked = nn.functional.one_hot(labels, 5).bool()
    runner_up = logits.masked_fill(blocked, float("-inf")).argmax(dim=1)
    true_weight, true_bias = bank.rows(labels)
    rival_weight, rival_bias = bank.rows(runner_up)
    direction = true_weight - rival_weight
    norm = direction.norm(dim=1)

    def signed_distance(points):
        return ((points * direction).sum(dim=1) + true_bias - rival_bias) / norm

    before, after = signed_distance(features), signed_distance(targets)
    confident = before >= distance
    assert torch.allclose(targets[confident], features[confident], atol=TOLERANCE)
    assert torch.allclose(after[~confident], torch.full_like(after[~confident], distance), atol=1e-3)
    # A correction must never move a point that was already fine.
    assert (after >= before - TOLERANCE).all()
    print(f"  margin targets        OK  ({int((~confident).sum())}/256 corrected)")


def check_velocity_field(device: torch.device) -> None:
    """The field starts as the identity flow, so z_T equals z_0 before training."""
    features = torch.randn(64, 16, device=device) * 3.0 + 1.5
    mean, scale = feature_statistics(features)
    field = build_velocity_field(16, seed=0, device=device, feature_mean=mean, feature_scale=scale)

    assert torch.allclose(field(x=features, t=0.0), torch.zeros_like(features), atol=TOLERANCE)
    final, states = rollout(field, features, 12)
    assert torch.allclose(final, features, atol=TOLERANCE)
    assert states.shape == (13, 64, 16)
    print("  velocity field        OK")


def check_rollout_gradients(device: torch.device) -> None:
    """Backpropagation must reach the field through every unrolled Euler step."""
    features = torch.randn(32, 16, device=device)
    field = build_velocity_field(16, seed=0, device=device)
    # Break the zero initialisation so the rolled gradient is not trivially zero.
    with torch.no_grad():
        field.output_projection.weight.normal_(std=0.01)

    final, _ = rollout(field, features, 4)
    final.pow(2).mean().backward()
    assert field.output_projection.weight.grad.abs().sum() > 0
    assert field.trunk[0].weight.grad.abs().sum() > 0
    print("  rollout gradients     OK")


def check_folds(device: torch.device) -> None:
    """Folds must be class-stratified so every fold-probe still sees every class."""
    labels = torch.arange(5, device=device).repeat_interleave(9)
    folds = assign_folds(labels, 3, seed=0)
    for class_id in range(5):
        counts = torch.bincount(folds[labels == class_id], minlength=3)
        assert counts.min() >= 2, counts
    print("  fold assignment       OK")


def check_centroids(device: torch.device) -> None:
    """Centroids must equal the per-class mean of the training features."""
    features = torch.randn(120, 16, device=device)
    labels = torch.randint(0, 4, (120,), device=device)
    centroids = class_centroids(features, labels, 4)
    for class_id in range(4):
        expected = features[labels == class_id].mean(dim=0)
        assert torch.allclose(centroids[class_id], expected, atol=1e-4)

    bank = ProbeBank([make_probe(4, 16, 0, device)])
    provider = build_target_provider(MARGIN, features, labels, bank, make_probe(4, 16, 0, device), 4)
    assert provider(features, labels).shape == features.shape
    print("  centroids / provider  OK")


def run_all_checks(device: torch.device | None = None) -> None:
    """Validate the Stage 3 components that carry non-trivial tensor logic.

    Args:
        device: Device to run on; defaults to CUDA when available.
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 3 component checks on {device}")
    check_probe_bank(device)
    check_margin_targets(device)
    check_velocity_field(device)
    check_rollout_gradients(device)
    check_folds(device)
    check_centroids(device)
    print("All Stage 3 component checks passed.")
