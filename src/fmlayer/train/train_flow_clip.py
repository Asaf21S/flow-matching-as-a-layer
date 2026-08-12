import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.fmlayer.data.fewshot import K_VALUES, SEEDS, load_train_subset
from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.encoders.base import default_device
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_matching_clip import (
    TEMPERATURE,
    ClipFlowWrapper,
    build_path,
    build_velocity_field,
    cosine_logits,
    field_config,
    predicted_endpoint,
    sample_paths,
)
from src.fmlayer.models.flow_ode import (
    make_time_grid,
    rollout,
    rollout_predictions,
    trajectory_accuracies,
    trajectory_predictions,
)
from src.fmlayer.models.prototypes import classify, l2_normalize
from src.fmlayer.train.evaluate import top1_accuracy
from src.fmlayer.utils.results import default_results_root, record_run
from src.fmlayer.utils.seeding import set_seed

STANDARD = "standard"
ROLLED = "rolled"
METHODS = {STANDARD: "fm_clip_standard", ROLLED: "fm_clip_rolled"}

ENCODER = ClipRN50Encoder.NAME
# The brief evaluates T in {4, 12}.
STEP_COUNTS = (4, 12)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 300
EVAL_EVERY = 10
PRINT_EVERY = 50

# Extensions, off by default. The brief's objective is the bare velocity regression.
CE_WEIGHT = 0.0
TARGET_NOISE = 0.0

# Diagnostic accuracy-versus-t sweep; not a deliverable, so it is opt-in.
CURVE_POINTS = 51
RECORD_STRIDE = 5
CURVE_STEPS = 50

# The t=0 endpoint is the untouched feature, so it must land on the zero-shot number.
# The slack only absorbs float32/TF32 tie-breaking between the torch and numpy matmuls.
ZEROSHOT_TOLERANCE = 5e-3
CURVES_DIRNAME = "flow_curves"
CHECKPOINT_DIRNAME = "flow_ckpt"
TEST_SPLIT = "test"


def load_clip_features(
    dataset: str, split: str, feature_root: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load one cached CLIP split, L2-normalised so the flow lives where cosine 1-NN scores.

    Args:
        dataset: Dataset key.
        split: Official split name.
        feature_root: Feature cache directory; defaults to the resolved feature root.

    Returns:
        Unit-norm features and their integer labels.
    """
    features, labels, _ = load_split(ENCODER, dataset, split, feature_root)
    return l2_normalize(features).astype(np.float32), labels


def load_clip_subset(
    dataset: str,
    k: int | str,
    seed: int,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one Stage 1 K-shot training subset, L2-normalised.

    Reuses the persisted Stage 1 indices so the FM layer trains on exactly the images the
    baseline protocol drew.

    Args:
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Subset seed, ignored when ``k`` is ``"full"``.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        subset_root: Subset index directory; defaults to the resolved subset root.

    Returns:
        Unit-norm subset features and their integer labels.
    """
    features, labels, _ = load_train_subset(
        ENCODER, dataset, k, seed, feature_root, subset_root
    )
    return l2_normalize(features).astype(np.float32), labels


def load_text_prototypes(dataset: str, feature_root: Path | None = None) -> np.ndarray:
    """Load the frozen CLIP text prototypes the flow transports features towards.

    Args:
        dataset: Dataset key.
        feature_root: Feature cache directory; defaults to the resolved feature root.

    Returns:
        Unit-norm prototypes of shape ``(num_classes, embed_dim)``.
    """
    prototypes, _, _ = load_split(ENCODER, dataset, None, feature_root)
    num_classes = get_spec(dataset).num_classes
    if len(prototypes) != num_classes:
        raise ValueError(
            f"{dataset}: {len(prototypes)} text prototypes for {num_classes} classes"
        )
    return l2_normalize(prototypes).astype(np.float32)


def zeroshot_accuracy(
    dataset: str, feature_root: Path | None = None
) -> float:
    """Score the Stage 1 prototype baseline the FM layer is compared against.

    Args:
        dataset: Dataset key.
        feature_root: Feature cache directory; defaults to the resolved feature root.

    Returns:
        Top-1 accuracy of the untouched features under the cosine rule.
    """
    features, labels = load_clip_features(dataset, TEST_SPLIT, feature_root)
    prototypes = load_text_prototypes(dataset, feature_root)
    return top1_accuracy(classify(features, prototypes), labels)


def batch_loss(
    model: ClipFlowWrapper,
    path,
    source: Tensor,
    target: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    generator: torch.Generator,
    objective: str,
    steps: int,
    ce_weight: float,
    temperature: float,
    target_noise: float,
) -> tuple[Tensor, float]:
    """Compute one training batch loss for either objective.

    Standard training regresses the velocity at a random point on the ideal path. Rolled-out
    training runs the same T-step Euler sequence used at inference and supervises only how
    close the final state lands to the prototype, backpropagating through all T predictions.

    Args:
        model: The wrapped velocity field.
        path: Probability path supplying ``z_t`` and ``u_i``; unused when rolled out.
        source: Features of shape ``(batch, embed_dim)``.
        target: Prototype of each feature's class, same shape.
        labels: Integer labels, for the cross-entropy extension.
        prototypes: All class prototypes, for the cross-entropy extension.
        generator: CPU generator making time and noise sampling reproducible.
        objective: ``"standard"`` or ``"rolled"``.
        steps: Euler steps ``T``, used by the rolled-out objective.
        ce_weight: Weight of the endpoint cross-entropy extension; 0 follows the brief.
        temperature: Softmax temperature of that extension.
        target_noise: Gaussian smoothing of the target, an extension; 0 follows the brief.

    Returns:
        The differentiable loss and the auxiliary cross-entropy value for logging.
    """
    if objective == ROLLED:
        final, _ = rollout(model, source, steps)
        return nn.functional.mse_loss(final, target), 0.0

    sample = sample_paths(path, source, target, generator, target_noise)
    velocity = model(x=sample.x_t, t=sample.t)
    loss = nn.functional.mse_loss(velocity, sample.dx_t)

    if ce_weight <= 0.0:
        return loss, 0.0

    endpoint = predicted_endpoint(sample.x_t, sample.t, velocity)
    class_loss = nn.functional.cross_entropy(
        cosine_logits(endpoint, prototypes, temperature), labels
    )
    return loss + ce_weight * class_loss, float(class_loss)


def validation_accuracies(
    model: ClipFlowWrapper,
    val_features: Tensor,
    val_labels: np.ndarray,
    prototypes: Tensor,
    step_counts: tuple[int, ...],
) -> dict:
    """Score the rollout endpoint on the validation split at each step count.

    Args:
        model: The wrapped velocity field.
        val_features: Validation features.
        val_labels: Validation labels.
        prototypes: Class prototypes.
        step_counts: Euler step counts to evaluate.

    Returns:
        Top-1 accuracy keyed by step count.
    """
    return {
        steps: top1_accuracy(
            rollout_predictions(model, val_features, prototypes, steps), val_labels
        )
        for steps in step_counts
    }


def train_flow(
    source: Tensor,
    target: Tensor,
    source_labels: Tensor,
    val_features: Tensor,
    val_labels: np.ndarray,
    prototypes: Tensor,
    seed: int,
    device: torch.device,
    objective: str = STANDARD,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    eval_every: int = EVAL_EVERY,
    ce_weight: float = CE_WEIGHT,
    temperature: float = TEMPERATURE,
    target_noise: float = TARGET_NOISE,
    verbose: bool = True,
    progress_desc: str | None = None,
) -> tuple[ClipFlowWrapper, list[dict], int]:
    """Fit the velocity field and keep the checkpoint with the best validation accuracy.

    Selection uses the mean rollout accuracy over ``step_counts``. For rolled-out training
    that tuple holds a single T, so the criterion is that T; for standard training it averages
    the step counts the field will be evaluated at, which keeps the choice fair to both.

    Args:
        source: Unit-norm training features, the t=0 end of each path.
        target: Prototype of each training feature's class, the t=1 end.
        source_labels: Labels of the training features.
        val_features: Unit-norm validation features.
        val_labels: Labels of the validation split.
        prototypes: Unit-norm class prototypes.
        seed: Seed controlling initialisation, batch order and time sampling.
        device: Device to train on.
        objective: ``"standard"`` or ``"rolled"``.
        step_counts: Euler step counts used for validation, and for the rollout when rolled.
        max_epochs: Number of epochs.
        batch_size: Examples per optimisation step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        eval_every: Epochs between validation sweeps.
        ce_weight: Weight of the endpoint cross-entropy extension; 0 follows the brief.
        temperature: Softmax temperature of that extension.
        target_noise: Gaussian smoothing of the target, an extension; 0 follows the brief.
        verbose: Print periodic losses and validation accuracy.
        progress_desc: Label of this run's own tqdm bar; ``None`` disables it.

    Returns:
        The field at its best checkpoint, the per-epoch history and the best epoch.
    """
    if objective not in (STANDARD, ROLLED):
        raise ValueError(f"objective must be {STANDARD!r} or {ROLLED!r}, got {objective!r}")

    model = build_velocity_field(source.shape[1], seed, device)
    path = build_path()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_steps = step_counts[0] if objective == ROLLED else 0

    history = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_accuracy = -1.0
    best_epoch = 0
    num_train = len(source)

    epochs = range(1, max_epochs + 1)
    if progress_desc:
        epochs = tqdm(epochs, desc=progress_desc, leave=False)

    for epoch in epochs:
        model.train()
        order = torch.randperm(num_train, generator=generator).to(device)
        loss_total = 0.0
        class_total = 0.0

        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)

            loss, class_value = batch_loss(
                model,
                path,
                source[batch],
                target[batch],
                source_labels[batch],
                prototypes,
                generator,
                objective,
                train_steps,
                ce_weight,
                temperature,
                target_noise,
            )
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * len(batch)
            class_total += class_value * len(batch)

        entry = {
            "epoch": epoch,
            "train_loss": loss_total / num_train,
            "class_loss": class_total / num_train,
        }

        if epoch % eval_every == 0 or epoch == max_epochs:
            model.eval()
            accuracies = validation_accuracies(
                model, val_features, val_labels, prototypes, step_counts
            )
            mean_accuracy = float(np.mean(list(accuracies.values())))
            entry["val_accuracy"] = mean_accuracy
            entry["val_by_steps"] = {str(key): value for key, value in accuracies.items()}

            if verbose and (epoch % PRINT_EVERY == 0 or epoch == max_epochs):
                detail = "  ".join(f"T={key} {value:.4f}" for key, value in accuracies.items())
                print(f"  epoch {epoch:3d}  loss {entry['train_loss']:.6f}  {detail}")

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
        elif verbose and epoch % PRINT_EVERY == 0:
            print(f"  epoch {epoch:3d}  loss {entry['train_loss']:.6f}")

        history.append(entry)

    model.load_state_dict(best_state)
    return model, history, best_epoch


def run_tag(objective: str, dataset: str, k: int | str, seed: int, steps: int | str) -> str:
    """Build the filename stem identifying one trained field."""
    return f"{METHODS[objective]}_{dataset}_k{k}_seed{seed}_T{steps}"


def curves_path(tag: str, results_root: Path | None = None) -> Path:
    """Build the path of a saved run record.

    Args:
        tag: Stem from :func:`run_tag`.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        Path of the JSON record.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / CURVES_DIRNAME / f"{tag}.json"


def checkpoint_path(tag: str, results_root: Path | None = None) -> Path:
    """Build the path of a saved velocity field.

    Args:
        tag: Stem from :func:`run_tag`.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        Path of the ``.pt`` checkpoint.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / CHECKPOINT_DIRNAME / f"{tag}.pt"


def load_flow_checkpoint(
    objective: str,
    dataset: str,
    k: int | str = "full",
    seed: int = 0,
    steps: int | str = "any",
    results_root: Path | None = None,
    device: torch.device | None = None,
) -> ClipFlowWrapper:
    """Rebuild a trained velocity field from disk, so the figures need no retraining.

    Args:
        objective: ``"standard"`` or ``"rolled"``.
        dataset: Dataset key.
        k: Shots per class the field was trained on.
        seed: Run seed.
        steps: Euler steps baked into training; ``"any"`` for standard runs.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to load onto; defaults to CUDA when available.

    Returns:
        The field in eval mode, wrapped for the ODE solver.
    """
    device = device if device is not None else default_device()
    path = checkpoint_path(run_tag(objective, dataset, k, seed, steps), results_root)
    if not path.is_file():
        raise FileNotFoundError(f"No flow checkpoint at {path}. Train it first.")

    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = build_velocity_field(
        config["embed_dim"], seed, device, config["hidden_dim"], config["num_layers"]
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def diagnostic_curve(
    model: ClipFlowWrapper,
    test_features: Tensor,
    test_labels: np.ndarray,
    prototypes: Tensor,
    points: int = CURVE_POINTS,
    steps: int = CURVE_STEPS,
) -> tuple[list[float], list[float]]:
    """Sweep accuracy along the trajectory, finely, for diagnosis only.

    Not a deliverable: the brief scores the endpoint after T steps. This sweep exists to
    show whether accuracy peaks before t=1 and how far the flow overshoots.

    Args:
        model: The trained field.
        test_features: Unit-norm test features.
        test_labels: Test labels.
        prototypes: Class prototypes.
        points: Number of grid times.
        steps: Sub-steps per unit of time for the fine integration.

    Returns:
        The grid times and the accuracy at each of them.
    """
    grid = make_time_grid(points)
    accuracies = trajectory_accuracies(
        trajectory_predictions(model, test_features, prototypes, grid, steps=steps),
        test_labels,
    )
    return [round(float(value), 4) for value in grid], accuracies


def run_flow_clip(
    dataset: str,
    objective: str = STANDARD,
    k: int | str = "full",
    seed: int = 0,
    steps: int | None = None,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    ce_weight: float = CE_WEIGHT,
    target_noise: float = TARGET_NOISE,
    with_curve: bool = False,
    record: bool = True,
    verbose: bool = True,
) -> dict:
    """Train one FM layer and score the transported test features.

    Standard training does not depend on T, so one field is evaluated at every entry of
    ``step_counts``. Rolled-out training bakes T in, so ``steps`` must be given and the field
    is scored at that T alone.

    Args:
        dataset: Dataset key, ``"dtd"`` or ``"aircraft"``.
        objective: ``"standard"`` or ``"rolled"``.
        k: Shots per class, or ``"full"``.
        seed: Run seed; also selects the K-shot subset, as in Stage 1.
        steps: Euler steps for rolled-out training; ignored when standard.
        step_counts: Step counts a standard field is evaluated at.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        subset_root: Subset index directory; defaults to the resolved subset root.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Number of training epochs.
        ce_weight: Weight of the endpoint cross-entropy extension; 0 follows the brief.
        target_noise: Gaussian smoothing of the target, an extension; 0 follows the brief.
        with_curve: Also compute the diagnostic accuracy-versus-t sweep.
        record: Append the results to ``runs.csv``.
        verbose: Print progress and a summary line.

    Returns:
        The baseline accuracy, the accuracy per evaluated T, the delta against the baseline,
        the training history and the artefact paths.
    """
    device = device if device is not None else default_device()
    set_seed(seed)

    if objective == ROLLED and steps is None:
        raise ValueError("Rolled-out training needs an explicit steps=T.")
    evaluated = (steps,) if objective == ROLLED else tuple(step_counts)
    tag = run_tag(objective, dataset, k, seed, steps if objective == ROLLED else "any")

    train_features, train_labels = load_clip_subset(
        dataset, k, seed, feature_root, subset_root
    )
    val_features, val_labels = load_clip_features(dataset, "val", feature_root)
    test_features, test_labels = load_clip_features(dataset, TEST_SPLIT, feature_root)
    prototypes = load_text_prototypes(dataset, feature_root)

    source = torch.from_numpy(train_features).to(device)
    target = torch.from_numpy(prototypes[train_labels]).to(device)
    source_labels = torch.from_numpy(train_labels).long().to(device)
    val_x = torch.from_numpy(val_features).to(device)
    test_x = torch.from_numpy(test_features).to(device)
    prototypes_x = torch.from_numpy(prototypes).to(device)

    model, history, best_epoch = train_flow(
        source,
        target,
        source_labels,
        val_x,
        val_labels,
        prototypes_x,
        seed,
        device,
        objective,
        evaluated,
        max_epochs,
        ce_weight=ce_weight,
        target_noise=target_noise,
        verbose=verbose,
        progress_desc=f"{METHODS[objective]} {dataset} k={k} seed={seed}",
    )
    model.eval()

    baseline = top1_accuracy(classify(test_features, prototypes), test_labels)
    accuracies = {
        count: top1_accuracy(
            rollout_predictions(model, test_x, prototypes_x, count), test_labels
        )
        for count in evaluated
    }
    deltas = {count: value - baseline for count, value in accuracies.items()}

    times, curve = ([], [])
    if with_curve:
        times, curve = diagnostic_curve(model, test_x, test_labels, prototypes_x)
        if abs(curve[0] - baseline) > ZEROSHOT_TOLERANCE:
            raise AssertionError(
                f"{dataset}: t=0 accuracy {curve[0]:.6f} does not reproduce the prototype "
                f"baseline {baseline:.6f}. Features and prototypes are misaligned, or the "
                "field is not the identity at t=0."
            )

    payload = {
        "method": METHODS[objective],
        "objective": objective,
        "dataset": dataset,
        "encoder": ENCODER,
        "k": k,
        "seed": seed,
        "steps": steps if objective == ROLLED else None,
        "num_train": len(train_labels),
        "best_epoch": best_epoch,
        "baseline_accuracy": baseline,
        "accuracy_by_steps": {str(key): value for key, value in accuracies.items()},
        "delta_by_steps": {str(key): value for key, value in deltas.items()},
        "ce_weight": ce_weight,
        "target_noise": target_noise,
        "times": times,
        "curve": curve,
        "history": history,
    }
    curve_file = curves_path(tag, results_root)
    curve_file.parent.mkdir(parents=True, exist_ok=True)
    curve_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    checkpoint_file = checkpoint_path(tag, results_root)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": field_config(source.shape[1]),
            "objective": objective,
            "dataset": dataset,
            "k": k,
            "seed": seed,
            "steps": steps,
        },
        checkpoint_file,
    )

    if record:
        for count, value in accuracies.items():
            record_run(
                {
                    "method": METHODS[objective],
                    "dataset": dataset,
                    "encoder": ENCODER,
                    "k": k,
                    "seed": seed,
                    "steps": count,
                    "t": "1.00",
                    "split": TEST_SPLIT,
                    "accuracy": value,
                    "num_items": len(test_labels),
                },
                results_root,
            )

    if verbose:
        scores = "  ".join(
            f"T={count} {value:.4f} ({deltas[count]:+.4f})"
            for count, value in accuracies.items()
        )
        print(
            f"    {METHODS[objective]:<18} {dataset:<9} k={str(k):<4} seed={seed}  "
            f"train={len(train_labels):>5}  base {baseline:.4f}  {scores}"
        )

    return {
        "dataset": dataset,
        "objective": objective,
        "k": k,
        "seed": seed,
        "baseline_accuracy": baseline,
        "accuracy_by_steps": accuracies,
        "delta_by_steps": deltas,
        "best_epoch": best_epoch,
        "history": history,
        "times": times,
        "curve": curve,
        "curves_path": curve_file,
        "checkpoint_path": checkpoint_file,
    }


def run_all_flow_clip(
    datasets: list[str] | None = None,
    k_values: tuple[int | str, ...] = K_VALUES,
    seeds: tuple[int, ...] = SEEDS,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    objectives: tuple[str, ...] = (STANDARD, ROLLED),
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    ce_weight: float = CE_WEIGHT,
    target_noise: float = TARGET_NOISE,
    record: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the whole Stage 2 grid.

    Standard training is T-independent, so it trains once per (dataset, K, seed) and is
    scored at every step count. Rolled-out training needs one field per step count.

    Args:
        datasets: Dataset keys; defaults to both Stage 1 datasets.
        k_values: Training-set sizes, defaulting to the Stage 1 protocol.
        seeds: Seeds per training-set size.
        step_counts: Euler step counts T.
        objectives: Which objectives to run.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        subset_root: Subset index directory; defaults to the resolved subset root.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Number of training epochs.
        ce_weight: Weight of the endpoint cross-entropy extension; 0 follows the brief.
        target_noise: Gaussian smoothing of the target, an extension; 0 follows the brief.
        record: Append the results to ``runs.csv``.
        verbose: Print progress.

    Returns:
        One result per trained field, keyed by ``"objective/dataset/k/seed/T"``.
    """
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)
    device = device if device is not None else default_device()
    print(f"Device: {device}")

    jobs = []
    for dataset in datasets:
        for k in k_values:
            for seed in seeds:
                if STANDARD in objectives:
                    jobs.append((STANDARD, dataset, k, seed, None))
                if ROLLED in objectives:
                    jobs.extend(
                        (ROLLED, dataset, k, seed, count) for count in step_counts
                    )

    results = {}
    for objective, dataset, k, seed, steps in tqdm(jobs, desc="stage 2 grid"):
        key = f"{objective}/{dataset}/{k}/{seed}/{steps if steps else 'any'}"
        results[key] = run_flow_clip(
            dataset,
            objective,
            k,
            seed,
            steps,
            step_counts,
            feature_root,
            subset_root,
            results_root,
            device,
            max_epochs,
            ce_weight,
            target_noise,
            record=record,
            verbose=verbose,
        )

    print(f"\n{len(results)} flow-matching fields trained.")
    return results


def summarize_flow_clip(results: dict) -> dict:
    """Aggregate the grid into mean and standard deviation per setting.

    Args:
        results: Output of :func:`run_all_flow_clip`.

    Returns:
        Mean accuracy, spread and delta against the baseline, keyed by
        ``"dataset/objective/T/K"``.
    """
    grouped: dict[str, list[float]] = {}
    baselines: dict[str, float] = {}

    for result in results.values():
        baselines[result["dataset"]] = result["baseline_accuracy"]
        for steps, value in result["accuracy_by_steps"].items():
            key = f"{result['dataset']}/{result['objective']}/T{steps}/K{result['k']}"
            grouped.setdefault(key, []).append(value)

    summary = {}
    for key, values in sorted(grouped.items()):
        dataset = key.split("/")[0]
        array = np.asarray(values)
        summary[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "delta": float(array.mean() - baselines[dataset]),
            "runs": len(array),
        }
        print(
            f"{key:<42} {array.mean():.4f} +/- {array.std(ddof=0):.4f}  "
            f"delta {array.mean() - baselines[dataset]:+.4f}  (n={len(array)})"
        )
    return summary

