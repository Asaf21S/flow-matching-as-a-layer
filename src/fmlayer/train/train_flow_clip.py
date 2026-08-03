import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.encoders.base import default_device
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_matching_clip import (
    ClipFlowWrapper,
    build_path,
    build_velocity_field,
    field_config,
    flow_matching_loss,
)
from src.fmlayer.models.flow_ode import (
    DEFAULT_METHOD,
    DEFAULT_STEPS,
    make_time_grid,
    trajectory_accuracies,
    trajectory_predictions,
)
from src.fmlayer.models.prototypes import classify, l2_normalize
from src.fmlayer.train.evaluate import top1_accuracy
from src.fmlayer.utils.results import default_results_root, record_run
from src.fmlayer.utils.seeding import set_seed

METHOD = "fm_clip"
ENCODER = ClipRN50Encoder.NAME
SEEDS = (0, 1, 2)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 300
EVAL_EVERY = 10
# 51 points puts a grid time on every 0.02; every 5th one is recorded in runs.csv.
CURVE_POINTS = 51
RECORD_STRIDE = 5
SOLVER_STEPS = (10, 50, 200)
# The t=0 endpoint is the untouched embedding, so it must land on the zero-shot number.
# The slack only absorbs float32/TF32 tie-breaking between the torch and numpy matmuls.
ZEROSHOT_TOLERANCE = 5e-3
CURVES_DIRNAME = "flow_curves"
CHECKPOINT_DIRNAME = "flow_ckpt"
TEST_SPLIT = "test"
K_LABEL = "full"


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


def load_text_prototypes(dataset: str, feature_root: Path | None = None) -> np.ndarray:
    """Load the frozen CLIP text prototypes that the flow transports embeddings towards.

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


def constant_shift_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    prototypes: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Score the trivial ablation of translating every embedding by one shared vector.

    The mean training displacement is essentially the CLIP modality gap. Because the same
    vector is added to every embedding it barely reorders the cosine similarities, so
    beating this number is what shows the flow learned class structure rather than a shift.

    Args:
        train_features: Unit-norm training image embeddings.
        train_labels: Labels of the training images.
        prototypes: Unit-norm text prototypes.
        test_features: Unit-norm test image embeddings.
        test_labels: Labels of the test images.

    Returns:
        Top-1 accuracy of the shifted test embeddings.
    """
    shift = (prototypes[train_labels] - train_features).mean(axis=0)
    return top1_accuracy(classify(test_features + shift, prototypes), test_labels)


def train_flow(
    source: Tensor,
    target: Tensor,
    val_features: Tensor,
    val_labels: np.ndarray,
    prototypes: Tensor,
    seed: int,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    eval_every: int = EVAL_EVERY,
    val_steps: int = DEFAULT_STEPS,
) -> tuple[ClipFlowWrapper, list[dict], int]:
    """Fit the velocity field and keep the checkpoint with the best validation accuracy.

    Validation integrates the whole val split to t=1 and classifies the endpoint, which is
    the quantity the layer is actually judged on, rather than the regression loss.

    Args:
        source: Unit-norm training image embeddings, the t=0 end of each path.
        target: Text embedding of each training image's label, the t=1 end.
        val_features: Unit-norm validation image embeddings.
        val_labels: Labels of the validation split.
        prototypes: Unit-norm text prototypes used for the validation 1-NN.
        seed: Seed controlling initialisation, batch order and time sampling.
        device: Device to train on.
        max_epochs: Number of epochs.
        batch_size: Examples per optimisation step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        eval_every: Epochs between validation integrations.
        val_steps: Euler sub-steps used when integrating the validation split.

    Returns:
        The field restored to its best checkpoint, the history and the best epoch.
    """
    model = build_velocity_field(source.shape[1], seed, device)
    path = build_path()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    endpoints = make_time_grid(2, device=device)

    history = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_accuracy = -1.0
    best_epoch = 0
    num_train = len(source)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(num_train, generator=generator).to(device)
        epoch_loss = 0.0

        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, path, source[batch], target[batch], generator)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        entry = {"epoch": epoch, "train_loss": epoch_loss / num_train}

        if epoch % eval_every == 0 or epoch == max_epochs:
            model.eval()
            predictions = trajectory_predictions(
                model, val_features, prototypes, endpoints, steps=val_steps
            )
            val_accuracy = top1_accuracy(predictions[-1], val_labels)
            entry["val_accuracy"] = val_accuracy

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

        history.append(entry)

    model.load_state_dict(best_state)
    return model, history, best_epoch


def curves_path(dataset: str, seed: int, results_root: Path | None = None) -> Path:
    """Build the path of a saved accuracy-versus-t curve.

    Args:
        dataset: Dataset key.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        Path of the JSON curve file.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / CURVES_DIRNAME / f"{METHOD}_{dataset}_seed{seed}.json"


def checkpoint_path(dataset: str, seed: int, results_root: Path | None = None) -> Path:
    """Build the path of a saved velocity field.

    Args:
        dataset: Dataset key.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        Path of the ``.pt`` checkpoint.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / CHECKPOINT_DIRNAME / f"{METHOD}_{dataset}_seed{seed}.pt"


def load_flow_checkpoint(
    dataset: str,
    seed: int = 0,
    results_root: Path | None = None,
    device: torch.device | None = None,
) -> ClipFlowWrapper:
    """Rebuild a trained velocity field from disk, so the figures need no retraining.

    Args:
        dataset: Dataset key.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to load onto; defaults to CUDA when available.

    Returns:
        The field in eval mode, wrapped for the ODE solver.
    """
    device = device if device is not None else default_device()
    path = checkpoint_path(dataset, seed, results_root)
    if not path.is_file():
        raise FileNotFoundError(f"No flow checkpoint at {path}. Run run_flow_clip() first.")

    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = build_velocity_field(
        config["embed_dim"],
        seed,
        device,
        config["hidden_dim"],
        config["num_blocks"],
        config["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run_flow_clip(
    dataset: str,
    seed: int = 0,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    solver_steps: tuple[int, ...] = SOLVER_STEPS,
    method: str = DEFAULT_METHOD,
    record: bool = True,
    verbose: bool = True,
) -> dict:
    """Train the flow-matching layer on one dataset and sweep its accuracy over t.

    Args:
        dataset: Dataset key, ``"dtd"`` or ``"aircraft"``.
        seed: Run seed controlling initialisation and batch order.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Number of training epochs.
        solver_steps: Euler sub-step counts the accuracy curve is recomputed with.
        method: Solver method used for the curves.
        record: Append the per-t accuracies to ``runs.csv``.
        verbose: Print a summary.

    Returns:
        The t=0 and t=1 accuracies, the reference baselines, the dense curve per solver
        setting, the training history and the artefact paths.
    """
    device = device if device is not None else default_device()
    set_seed(seed)

    train_features, train_labels = load_clip_features(dataset, "train", feature_root)
    val_features, val_labels = load_clip_features(dataset, "val", feature_root)
    test_features, test_labels = load_clip_features(dataset, TEST_SPLIT, feature_root)
    prototypes = load_text_prototypes(dataset, feature_root)

    source = torch.from_numpy(train_features).to(device)
    target = torch.from_numpy(prototypes[train_labels]).to(device)
    val_x = torch.from_numpy(val_features).to(device)
    test_x = torch.from_numpy(test_features).to(device)
    prototypes_x = torch.from_numpy(prototypes).to(device)

    model, history, best_epoch = train_flow(
        source,
        target,
        val_x,
        val_labels,
        prototypes_x,
        seed,
        device,
        max_epochs,
        val_steps=DEFAULT_STEPS,
    )
    model.eval()

    time_grid = make_time_grid(CURVE_POINTS)
    times = [round(float(value), 4) for value in time_grid]
    curves = {
        steps: trajectory_accuracies(
            trajectory_predictions(
                model, test_x, prototypes_x, time_grid, method, steps
            ),
            test_labels,
        )
        for steps in solver_steps
    }
    reference = curves[DEFAULT_STEPS] if DEFAULT_STEPS in curves else curves[solver_steps[-1]]

    zeroshot_accuracy = top1_accuracy(classify(test_features, prototypes), test_labels)
    if abs(reference[0] - zeroshot_accuracy) > ZEROSHOT_TOLERANCE:
        raise AssertionError(
            f"{dataset}: t=0 accuracy {reference[0]:.6f} does not reproduce the zero-shot "
            f"baseline {zeroshot_accuracy:.6f}. The features and the text prototypes are "
            "misaligned, or the field is not the identity at t=0."
        )
    shift_accuracy = constant_shift_accuracy(
        train_features, train_labels, prototypes, test_features, test_labels
    )
    best_val_accuracy = max(
        entry["val_accuracy"] for entry in history if "val_accuracy" in entry
    )

    payload = {
        "method": METHOD,
        "dataset": dataset,
        "encoder": ENCODER,
        "seed": seed,
        "num_train": len(train_labels),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "solver_method": method,
        "times": times,
        "curves": {str(steps): values for steps, values in curves.items()},
        "zeroshot_accuracy": zeroshot_accuracy,
        "constant_shift_accuracy": shift_accuracy,
        "history": history,
    }
    curve_file = curves_path(dataset, seed, results_root)
    curve_file.parent.mkdir(parents=True, exist_ok=True)
    curve_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    checkpoint_file = checkpoint_path(dataset, seed, results_root)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": field_config(source.shape[1]),
            "dataset": dataset,
            "seed": seed,
        },
        checkpoint_file,
    )

    if record:
        for index in range(0, len(times), RECORD_STRIDE):
            record_run(
                {
                    "method": METHOD,
                    "dataset": dataset,
                    "encoder": ENCODER,
                    "k": K_LABEL,
                    "seed": seed,
                    "t": f"{times[index]:.2f}",
                    "split": TEST_SPLIT,
                    "accuracy": reference[index],
                    "num_items": len(test_labels),
                },
                results_root,
            )

    if verbose:
        print(
            f"    {dataset:<9} seed={seed}  train={len(train_labels):>5}  "
            f"val={best_val_accuracy:.4f}@{best_epoch:<4} "
            f"t=0 {reference[0]:.4f} -> t=1 {reference[-1]:.4f}  "
            f"(shift ablation {shift_accuracy:.4f})"
        )

    return {
        "dataset": dataset,
        "seed": seed,
        "times": times,
        "curves": curves,
        "accuracy_t0": reference[0],
        "accuracy_t1": reference[-1],
        "zeroshot_accuracy": zeroshot_accuracy,
        "constant_shift_accuracy": shift_accuracy,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "curves_path": curve_file,
        "checkpoint_path": checkpoint_file,
    }


def run_all_flow_clip(
    datasets: list[str] | None = None,
    seeds: tuple[int, ...] = SEEDS,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    solver_steps: tuple[int, ...] = SOLVER_STEPS,
    record: bool = True,
) -> dict:
    """Train the flow-matching layer on every dataset and seed.

    Args:
        datasets: Dataset keys; defaults to both Stage 1 datasets.
        seeds: Seeds; each one re-initialises the field on the same training split.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Number of training epochs.
        solver_steps: Euler sub-step counts the accuracy curve is recomputed with.
        record: Append the results to ``runs.csv``.

    Returns:
        One result per run, keyed by ``"dataset/seed"``.
    """
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)
    device = device if device is not None else default_device()
    print(f"Device: {device}")

    jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    results = {}
    for dataset, seed in tqdm(jobs, desc="flow-matching layers"):
        results[f"{dataset}/{seed}"] = run_flow_clip(
            dataset,
            seed,
            feature_root,
            results_root,
            device,
            max_epochs,
            solver_steps,
            record=record,
        )

    print(f"\n{len(results)} flow-matching runs complete.")
    return results


def summarize_flow_clip(results: dict) -> dict:
    """Aggregate the runs into mean and standard deviation per dataset.

    Args:
        results: Output of :func:`run_all_flow_clip`.

    Returns:
        The t=0 and t=1 accuracies and the two reference baselines, keyed by dataset.
    """
    grouped: dict[str, list[dict]] = {}
    for result in results.values():
        grouped.setdefault(result["dataset"], []).append(result)

    summary = {}
    for dataset, runs in sorted(grouped.items()):
        start = np.asarray([run["accuracy_t0"] for run in runs])
        end = np.asarray([run["accuracy_t1"] for run in runs])
        summary[dataset] = {
            "t0_mean": float(start.mean()),
            "t1_mean": float(end.mean()),
            "t1_std": float(end.std(ddof=0)),
            "zeroshot": runs[0]["zeroshot_accuracy"],
            "constant_shift": runs[0]["constant_shift_accuracy"],
            "runs": len(runs),
        }
        print(
            f"{get_spec(dataset).display_name:<15} "
            f"t=0 {start.mean():.4f}  ->  t=1 {end.mean():.4f} +/- {end.std(ddof=0):.4f}  "
            f"(n={len(runs)}, shift ablation {runs[0]['constant_shift_accuracy']:.4f})"
        )
    return summary

