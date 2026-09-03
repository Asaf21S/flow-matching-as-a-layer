import math
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn

from src.fmlayer.data.fewshot import load_train_subset
from src.fmlayer.data.specs import get_spec
from src.fmlayer.encoders.base import default_device
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_ode import transport
from src.fmlayer.train.probes import get_probe
from src.fmlayer.train.train_linear import to_tensors

EPSILON = 1e-8


@torch.no_grad()
def flow_diagnostics(
    result: dict, features: Tensor, labels: Tensor, steps: int | None = None
) -> dict:
    """Measure how far a trained flow moves features and whether the moves help.

    A flow that cannot beat the baseline is either doing nothing or doing harm. Counting
    label flips in both directions separates those two cases, which a single accuracy
    number cannot.

    Args:
        result: One entry returned by :func:`run_stage3`.
        features: Features to score, already on the probe's device.
        labels: Labels matching ``features``.
        steps: Euler steps; defaults to the largest T the run was evaluated at.

    Returns:
        Displacement statistics and the correct/incorrect flip counts.
    """
    probe: nn.Linear = result["classifier"]
    # The joint variant fine-tunes its own classifier, so "before" has to be read off the
    # untouched Stage 1 probe or the baseline would move with the method under test.
    baseline_probe: nn.Linear = result.get("baseline_classifier", probe)
    probe.eval()
    baseline_probe.eval()
    steps = steps if steps is not None else max(result["accuracy_by_steps"])

    transported = transport(result["fm_layer"], features, steps)
    base_prediction = baseline_probe(features).argmax(dim=1)
    flow_prediction = probe(transported).argmax(dim=1)

    base_correct = base_prediction == labels
    flow_correct = flow_prediction == labels
    displacement = (transported - features).norm(dim=1) / features.norm(dim=1).clamp_min(EPSILON)

    fixed = int((~base_correct & flow_correct).sum())
    broken = int((base_correct & ~flow_correct).sum())
    return {
        "config_name": result["config_name"],
        "encoder": result["encoder"],
        "dataset": result["dataset"],
        "k": str(result["k"]),
        "seed": result["seed"],
        "steps": steps,
        "baseline_acc": float(base_correct.float().mean()),
        "flow_acc": float(flow_correct.float().mean()),
        "delta": float(flow_correct.float().mean() - base_correct.float().mean()),
        "rel_displacement": float(displacement.mean()),
        "changed_frac": float((flow_prediction != base_prediction).float().mean()),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "num_items": int(len(labels)),
    }


def diagnose_all(
    results: dict, split: str = "test", feature_root: Path | None = None
) -> pd.DataFrame:
    """Run :func:`flow_diagnostics` over a whole result grid.

    Args:
        results: Output of :func:`run_all_stage3` or :func:`screen_configs`.
        split: Which split to diagnose on.
        feature_root: Feature cache directory.

    Returns:
        One row per run, sorted by delta, with displacement and flip counts.
    """
    cache: dict[tuple, tuple[Tensor, Tensor]] = {}
    rows = []

    for result in results.values():
        device = result["classifier"].weight.device
        key = (result["encoder"], result["dataset"], split, str(device))
        if key not in cache:
            features, labels, _ = load_split(
                result["encoder"], result["dataset"], split, feature_root
            )
            cache[key] = to_tensors(features, labels, device)
        features, labels = cache[key]
        rows.append(flow_diagnostics(result, features, labels))

    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def print_diagnostics(frame: pd.DataFrame) -> None:
    """Print the diagnostic table in a readable fixed-width layout.

    Args:
        frame: Output of :func:`diagnose_all`.
    """
    print(
        f"{'configuration':<40} {'delta':>8} {'move':>7} {'flip%':>7} "
        f"{'fixed':>6} {'broken':>7} {'net':>5}"
    )
    print("-" * 84)
    for _, row in frame.iterrows():
        print(
            f"{row['config_name']:<40} {row['delta']:>+8.4f} {row['rel_displacement']:>7.3f} "
            f"{100 * row['changed_frac']:>6.1f}% {row['fixed']:>6} {row['broken']:>7} "
            f"{row['net']:>+5}"
        )
    print(
        "\nmove   = mean ||z_T - z_0|| / ||z_0||   (0 means the flow is the identity)\n"
        "flip%  = share of items whose predicted label changed\n"
        "fixed  = wrong -> right,  broken = right -> wrong"
    )


def signal_diagnostics(
    encoder: str,
    dataset: str,
    k: int | str = 10,
    seed: int = 0,
    noise_levels: tuple[float, ...] = (0.0, 0.15, 0.3, 0.5, 1.0),
    target_lr: float = 0.1,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Measure how much training signal the frozen classifier can actually supply.

    Both Stage 3 strategies derive everything they learn from the gradient of the
    classification loss with respect to the features. If the frozen probe has memorised its
    K-shot training subset, that gradient is ~0 on exactly the points the flow trains on,
    and a field initialised at the identity has nothing to move towards. This quantifies
    that, on clean sources and on increasingly perturbed ones.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        noise_levels: Source perturbations to probe, as fractions of the mean feature norm.
        target_lr: The guided step size, used to report how far a target would actually move.
        feature_root: Feature cache directory.
        subset_root: Subset index directory.
        results_root: Results directory holding the probe cache.
        device: Device to compute on.

    Returns:
        One row per noise level: accuracy and loss of the probe on the sources, the mean
        per-sample gradient norm, and the relative displacement one guided step would make.
    """
    device = device if device is not None else default_device()
    num_classes = get_spec(dataset).num_classes

    train_features, train_labels, _ = load_train_subset(
        encoder, dataset, k, seed, feature_root, subset_root
    )
    val_features, val_labels, _ = load_split(encoder, dataset, "val", feature_root)
    train_x, train_y = to_tensors(train_features, train_labels, device)
    val_x, val_y = to_tensors(val_features, val_labels, device)

    probe = get_probe(
        encoder, dataset, k, seed, train_x, train_y, val_x, val_y,
        num_classes, device, results_root,
    )
    probe.eval()

    mean_norm = train_x.norm(p=2, dim=1).mean()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []

    for sigma in noise_levels:
        source = train_x
        if sigma > 0:
            noise = torch.randn(train_x.shape, generator=generator).to(device)
            source = train_x + noise * (sigma * mean_norm / math.sqrt(train_x.shape[1]))

        source = source.detach().requires_grad_(True)
        logits = probe(source)
        # Summed, so each row of the gradient is that sample's own dCE_i/dz_i.
        loss = nn.functional.cross_entropy(logits, train_y, reduction="sum")
        gradient = torch.autograd.grad(loss, source)[0]

        with torch.no_grad():
            correct = (logits.argmax(dim=1) == train_y).float().mean()
            per_sample_loss = loss / len(train_y)
            gradient_norm = gradient.norm(dim=1)
            step_fraction = target_lr * gradient_norm / source.norm(dim=1).clamp_min(EPSILON)

        rows.append(
            {
                "encoder": encoder,
                "dataset": dataset,
                "k": str(k),
                "seed": seed,
                "noise_std": sigma,
                "train_accuracy": float(correct),
                "train_ce": float(per_sample_loss),
                "grad_norm": float(gradient_norm.mean()),
                "guided_step_frac": float(step_fraction.mean()),
            }
        )

    return pd.DataFrame(rows)


def print_signal_diagnostics(frame: pd.DataFrame) -> None:
    """Print :func:`signal_diagnostics` in a readable fixed-width layout."""
    print(
        f"{'cell':<28} {'noise':>6} {'train acc':>10} {'train CE':>10} "
        f"{'|grad|':>10} {'step/|z|':>10}"
    )
    print("-" * 78)
    for _, row in frame.iterrows():
        cell = f"{row['dataset']}/{row['encoder']}"
        print(
            f"{cell:<28} {row['noise_std']:>6.2f} {row['train_accuracy']:>10.4f} "
            f"{row['train_ce']:>10.2e} {row['grad_norm']:>10.2e} "
            f"{row['guided_step_frac']:>10.2e}"
        )
    print(
        "\ntrain acc = frozen probe on the sources the flow trains from\n"
        "train CE  = its cross-entropy there; ~0 means the loss is saturated\n"
        "|grad|    = mean ||dCE_i/dz_i||, the only signal either strategy receives\n"
        "step/|z|  = relative distance one guided target step would move a feature"
    )

