import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.encoders.base import default_device
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_matching_clip import (
    ClipFlowWrapper,
    TEMPERATURE,
    build_path,
    build_velocity_field,
    cosine_logits,
    field_config,
    predicted_endpoint,
    sample_paths,
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
PRINT_EVERY = 50
# Weight of the endpoint cross-entropy added to the flow-matching regression. Pure CFM
# regresses onto a barycentre of prototypes, which is the worst possible place for a cosine
# 1-NN; this term makes the layer optimise the metric it is scored on.
CE_WEIGHT = 1.0
# Gaussian smoothing of the t=1 target, so it is a cloud rather than one of C atoms.
TARGET_NOISE = 0.1
# Keep the integration on the unit sphere, where the cosine classifier lives.
RENORMALIZE = True
# 51 points puts a grid time on every 0.02; every 5th one is recorded in runs.csv.
CURVE_POINTS = 51
RECORD_STRIDE = 5
# Validation sweeps the same 11 times that get recorded, so the selected t is on the grid.
VAL_POINTS = 11
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

    The mean training displacement is essentially the CLIP modality gap. Note this is *not*
    a no-op for the classifier: with unit-norm prototypes the shifted score is
    ``(z + m) . t_c = z . t_c + m . t_c``, and the bias ``m . t_c`` differs per class, so a
    shared translation reorders the similarities and typically hurts. It is a lower bound
    the flow must clear, not a neutral reference.

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
    source_labels: Tensor,
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
    val_points: int = VAL_POINTS,
    ce_weight: float = CE_WEIGHT,
    temperature: float = TEMPERATURE,
    target_noise: float = TARGET_NOISE,
    renormalize: bool = RENORMALIZE,
    verbose: bool = True,
    progress_desc: str | None = None,
) -> tuple[ClipFlowWrapper, list[dict], int, float]:
    """Fit the velocity field, selecting both the epoch and the stopping time on validation.

    The loss is the flow-matching regression plus, optionally, a cross-entropy on the
    single-step estimate of the t=1 endpoint. Validation sweeps the whole time grid rather
    than only t=1, because the transport is most useful before it contracts the cloud.

    Args:
        source: Unit-norm training image embeddings, the t=0 end of each path.
        target: Text embedding of each training image's label, the t=1 end.
        source_labels: Labels of the training images, for the endpoint cross-entropy.
        val_features: Unit-norm validation image embeddings.
        val_labels: Labels of the validation split.
        prototypes: Unit-norm text prototypes used for the validation 1-NN.
        seed: Seed controlling initialisation, batch order and time sampling.
        device: Device to train on.
        max_epochs: Number of epochs.
        batch_size: Examples per optimisation step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        eval_every: Epochs between validation sweeps.
        val_steps: Euler sub-steps used when integrating the validation split.
        val_points: Number of times the validation sweep evaluates.
        ce_weight: Weight of the endpoint cross-entropy; 0 gives pure flow matching.
        temperature: Softmax temperature of the endpoint cross-entropy.
        target_noise: Gaussian smoothing of the t=1 target.
        renormalize: Keep the validation integration on the unit sphere.
        verbose: Print per-epoch losses and validation accuracy.
        progress_desc: Label of this run's own tqdm bar; ``None`` disables it.

    Returns:
        The field at its best checkpoint, the history, the best epoch and the time that
        maximised validation accuracy.
    """
    model = build_velocity_field(source.shape[1], seed, device)
    path = build_path()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    val_grid = make_time_grid(val_points, device=device)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_accuracy = -1.0
    best_epoch = 0
    best_time = 1.0
    num_train = len(source)

    epochs = tqdm(range(1, max_epochs + 1), desc=progress_desc, leave=False) if progress_desc else range(1, max_epochs + 1)
    for epoch in epochs:
        model.train()
        order = torch.randperm(num_train, generator=generator).to(device)
        flow_total = 0.0
        class_total = 0.0

        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)

            sample = sample_paths(
                path, source[batch], target[batch], generator, target_noise
            )
            velocity = model(x=sample.x_t, t=sample.t)
            flow_loss = nn.functional.mse_loss(velocity, sample.dx_t)
            loss = flow_loss

            class_loss = torch.zeros((), device=device)
            if ce_weight > 0.0:
                endpoint = predicted_endpoint(sample.x_t, sample.t, velocity)
                class_loss = criterion(
                    cosine_logits(endpoint, prototypes, temperature), source_labels[batch]
                )
                loss = loss + ce_weight * class_loss

            loss.backward()
            optimizer.step()
            flow_total += flow_loss.item() * len(batch)
            class_total += float(class_loss) * len(batch)

        entry = {
            "epoch": epoch,
            "flow_loss": flow_total / num_train,
            "class_loss": class_total / num_train,
        }

        if epoch % eval_every == 0 or epoch == max_epochs:
            model.eval()
            accuracies = trajectory_accuracies(
                trajectory_predictions(
                    model,
                    val_features,
                    prototypes,
                    val_grid,
                    steps=val_steps,
                    renormalize=renormalize,
                ),
                val_labels,
            )
            index = int(np.argmax(accuracies))
            entry["val_accuracy"] = accuracies[index]
            entry["val_time"] = float(val_grid[index])
            entry["val_accuracy_t1"] = accuracies[-1]

            if verbose and (epoch % PRINT_EVERY == 0 or epoch == max_epochs):
                print(
                    f"  epoch {epoch:3d}  flow {entry['flow_loss']:.4f}  "
                    f"ce {entry['class_loss']:.4f}  "
                    f"val {accuracies[index]:.4f}@t={entry['val_time']:.1f}  "
                    f"(t=1 {accuracies[-1]:.4f})"
                )

            if accuracies[index] > best_accuracy:
                best_accuracy = accuracies[index]
                best_epoch = epoch
                best_time = entry["val_time"]
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
        elif verbose and epoch % PRINT_EVERY == 0:
            print(
                f"  epoch {epoch:3d}  flow {entry['flow_loss']:.4f}  "
                f"ce {entry['class_loss']:.4f}"
            )

        history.append(entry)

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_time


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
    ce_weight: float = CE_WEIGHT,
    target_noise: float = TARGET_NOISE,
    renormalize: bool = RENORMALIZE,
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
        ce_weight: Weight of the endpoint cross-entropy; 0 gives pure flow matching.
        target_noise: Gaussian smoothing of the t=1 target.
        renormalize: Keep the integration on the unit sphere.
        record: Append the per-t accuracies to ``runs.csv``.
        verbose: Print per-epoch loss and validation, plus a final summary.

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
    source_labels = torch.from_numpy(train_labels).long().to(device)
    val_x = torch.from_numpy(val_features).to(device)
    test_x = torch.from_numpy(test_features).to(device)
    prototypes_x = torch.from_numpy(prototypes).to(device)

    model, history, best_epoch, best_time = train_flow(
        source,
        target,
        source_labels,
        val_x,
        val_labels,
        prototypes_x,
        seed,
        device,
        max_epochs,
        val_steps=DEFAULT_STEPS,
        ce_weight=ce_weight,
        target_noise=target_noise,
        renormalize=renormalize,
        verbose=verbose,
        progress_desc=f"{dataset}/seed{seed}",
    )
    model.eval()

    time_grid = make_time_grid(CURVE_POINTS)
    times = [round(float(value), 4) for value in time_grid]
    curves = {
        steps: trajectory_accuracies(
            trajectory_predictions(
                model,
                test_x,
                prototypes_x,
                time_grid,
                method,
                steps,
                renormalize=renormalize,
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
    # The stopping time is chosen on validation, so the test number stays honest.
    selected = min(range(len(times)), key=lambda index: abs(times[index] - best_time))
    accuracy_at_best_time = reference[selected]

    payload = {
        "method": METHOD,
        "dataset": dataset,
        "encoder": ENCODER,
        "seed": seed,
        "num_train": len(train_labels),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_time": best_time,
        "accuracy_at_best_time": accuracy_at_best_time,
        "ce_weight": ce_weight,
        "target_noise": target_noise,
        "renormalize": renormalize,
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
            f"epoch {best_epoch:<4} "
            f"t=0 {reference[0]:.4f} | best t={best_time:.1f} {accuracy_at_best_time:.4f} "
            f"| t=1 {reference[-1]:.4f}  (shift {shift_accuracy:.4f})"
        )

    return {
        "dataset": dataset,
        "seed": seed,
        "times": times,
        "curves": curves,
        "accuracy_t0": reference[0],
        "accuracy_t1": reference[-1],
        "best_time": best_time,
        "accuracy_at_best_time": accuracy_at_best_time,
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
    ce_weight: float = CE_WEIGHT,
    target_noise: float = TARGET_NOISE,
    renormalize: bool = RENORMALIZE,
    record: bool = True,
    verbose: bool = True,
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
        ce_weight: Weight of the endpoint cross-entropy; 0 gives pure flow matching.
        target_noise: Gaussian smoothing of the t=1 target.
        renormalize: Keep the integration on the unit sphere.
        record: Append the results to ``runs.csv``.
        verbose: Print per-epoch progress.

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
            ce_weight=ce_weight,
            target_noise=target_noise,
            renormalize=renormalize,
            record=record,
            verbose=verbose,
        )

    print(f"\n{len(results)} flow-matching runs complete.")
    return results


def summarize_flow_clip(results: dict) -> dict:
    """Aggregate the runs into mean and standard deviation per dataset.

    Args:
        results: Output of :func:`run_all_flow_clip`.

    Returns:
        The t=0, best-t and t=1 accuracies plus the reference baselines, keyed by dataset.
    """
    grouped: dict[str, list[dict]] = {}
    for result in results.values():
        grouped.setdefault(result["dataset"], []).append(result)

    summary = {}
    for dataset, runs in sorted(grouped.items()):
        start = np.asarray([run["accuracy_t0"] for run in runs])
        end = np.asarray([run["accuracy_t1"] for run in runs])
        best = np.asarray([run["accuracy_at_best_time"] for run in runs])
        times = [run["best_time"] for run in runs]
        summary[dataset] = {
            "t0_mean": float(start.mean()),
            "t1_mean": float(end.mean()),
            "t1_std": float(end.std(ddof=0)),
            "best_mean": float(best.mean()),
            "best_std": float(best.std(ddof=0)),
            "best_times": times,
            "zeroshot": runs[0]["zeroshot_accuracy"],
            "constant_shift": runs[0]["constant_shift_accuracy"],
            "runs": len(runs),
        }
        gain = best.mean() - start.mean()
        print(
            f"{get_spec(dataset).display_name:<15} "
            f"t=0 {start.mean():.4f} | best {best.mean():.4f} +/- {best.std(ddof=0):.4f} "
            f"at t={times} | t=1 {end.mean():.4f}  "
            f"=> {gain:+.4f} over the baseline (n={len(runs)})"
        )
    return summary

