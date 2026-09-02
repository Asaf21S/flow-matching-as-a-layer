from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn

from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_ode import transport
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
