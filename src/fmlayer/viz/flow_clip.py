import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.fmlayer.data.class_names import normalize_class_name
from src.fmlayer.data.datasets import build_dataset, get_class_names, get_targets
from src.fmlayer.data.fewshot import K_FULL
from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_ode import MANUAL_EULER, integrate, rollout_trajectory
from src.fmlayer.models.retrieval import nearest_neighbors
from src.fmlayer.train.train_flow_clip import (
    ENCODER,
    METHODS,
    ROLLED,
    STANDARD,
    STEP_COUNTS,
    curves_path,
    load_clip_features,
    load_flow_checkpoint,
    load_text_prototypes,
    run_tag,
)
from src.fmlayer.viz.embeddings import COLORMAP, DEFAULT_NUM_CLASSES, select_classes
from src.fmlayer.viz.figures import apply_plot_style, save_figure

BASELINE_COLOR = "#d62728"
VARIANT_STYLE = {
    (STANDARD, 4): ("#a78bfa", "o", "-"),
    (STANDARD, 12): ("#7b2cbf", "s", "-"),
    (ROLLED, 4): ("#5eead4", "^", "--"),
    (ROLLED, 12): ("#0f766e", "D", "--"),
}
VARIANT_LABEL = {STANDARD: "Standard FM", ROLLED: "Rolled-out FM"}
K_ORDER = ("5", "10", K_FULL)
TRAJECTORY_EXAMPLES = 6
PROJECTION_SEED = 0
REVERSE_TIMES = (1.0, 0.75, 0.5, 0.25, 0.0)


def load_run(
    objective: str,
    dataset: str,
    k: int | str = K_FULL,
    seed: int = 0,
    steps: int | str = "any",
    results_root: Path | None = None,
) -> dict:
    """Read the JSON record written by one training run.

    Args:
        objective: ``"standard"`` or ``"rolled"``.
        dataset: Dataset key.
        k: Shots per class the field was trained on.
        seed: Run seed.
        steps: Euler steps baked into training; ``"any"`` for standard runs.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        The parsed payload.
    """
    path = curves_path(run_tag(objective, dataset, k, seed, steps), results_root)
    if not path.is_file():
        raise FileNotFoundError(f"No run record at {path}. Train it first.")
    return json.loads(path.read_text(encoding="utf-8"))


def k_position(k: str) -> float:
    """Map a training-set size onto the x axis."""
    return float(K_ORDER.index(str(k)))


def plot_accuracy_vs_k(
    table: pd.DataFrame,
    datasets: list[str] | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Deliverable 1: accuracy versus K for every FM variant, against the flat baseline.

    Args:
        table: Aggregated rows with ``dataset``, ``method``, ``steps``, ``k``, ``mean``,
            ``std`` and ``baseline`` columns.
        datasets: Dataset keys; defaults to both.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.
        save: Write the figure to disk.

    Returns:
        Path of the saved figure, or ``None`` when ``save`` is ``False``.
    """
    apply_plot_style()
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.8 * len(datasets), 5))

    for ax, dataset in zip(np.atleast_1d(axes).ravel(), datasets):
        subset = table[table["dataset"] == dataset]
        if subset.empty:
            continue

        baseline = float(subset["baseline"].iloc[0])
        ax.axhline(
            baseline,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.8,
            label=f"Stage 1 prototype baseline ({baseline * 100:.1f}%)",
        )

        for (method, steps), rows in subset.groupby(["method", "steps"]):
            objective = STANDARD if method == METHODS[STANDARD] else ROLLED
            color, marker, style = VARIANT_STYLE.get(
                (objective, int(steps)), ("#333333", "o", "-")
            )
            rows = rows.copy()
            rows["position"] = rows["k"].map(k_position)
            rows = rows.sort_values("position")
            ax.errorbar(
                rows["position"],
                rows["mean"],
                yerr=rows["std"],
                marker=marker,
                color=color,
                linestyle=style,
                linewidth=2,
                markersize=7,
                capsize=5,
                capthick=1.5,
                label=f"{VARIANT_LABEL[objective]}, T={steps}",
            )

        ax.set_xticks(range(len(K_ORDER)), [f"K = {k}" for k in K_ORDER])
        ax.set_xlabel("Training Images per Class (K)", labelpad=8)
        ax.set_ylabel("Top-1 Test Accuracy", labelpad=8)
        ax.set_title(get_spec(dataset).display_name, pad=10)
        ax.legend(loc="best", frameon=True, framealpha=0.95)

    fig.suptitle(
        "Flow-Matching Layer vs. the Stage 1 Prototype Baseline", fontsize=13, y=1.02
    )
    fig.tight_layout()
    return save_figure(fig, "flow_accuracy_vs_k", figures_root, show=show, save=save)


def plot_training_curves(
    dataset: str,
    k: int | str = K_FULL,
    seed: int = 0,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Deliverable 2: representative training-loss curves for both objectives.

    The two losses are on different scales -- a velocity MSE against a target velocity for
    standard training, an endpoint MSE against the prototype for rolled-out -- so they get
    separate panels rather than a shared axis.

    Args:
        dataset: Dataset key.
        k: Shots per class of the representative run.
        seed: Run seed of the representative run.
        step_counts: Euler step counts whose rolled-out runs are drawn.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.
        save: Write the figure to disk.

    Returns:
        Path of the saved figure, or ``None`` when ``save`` is ``False``.
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    standard = load_run(STANDARD, dataset, k, seed, "any", results_root)
    epochs = [entry["epoch"] for entry in standard["history"]]
    axes[0].plot(
        epochs,
        [entry["train_loss"] for entry in standard["history"]],
        color=VARIANT_STYLE[(STANDARD, 12)][0],
        linewidth=2,
    )
    axes[0].set_title("Standard FM: velocity regression loss", pad=10)

    for steps in step_counts:
        rolled = load_run(ROLLED, dataset, k, seed, steps, results_root)
        color = VARIANT_STYLE.get((ROLLED, steps), ("#0f766e", "o", "--"))[0]
        axes[1].plot(
            [entry["epoch"] for entry in rolled["history"]],
            [entry["train_loss"] for entry in rolled["history"]],
            color=color,
            linewidth=2,
            label=f"T={steps}",
        )
    axes[1].set_title("Rolled-out FM: endpoint loss", pad=10)
    axes[1].legend(loc="best", frameon=True, framealpha=0.95)

    for ax in axes:
        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_ylabel("Training loss", labelpad=8)
        ax.set_yscale("log")

    fig.suptitle(
        f"{get_spec(dataset).display_name}: training stability (K={k}, seed={seed})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(
        fig, f"flow_training_curves_{dataset}", figures_root, show=show, save=save
    )


def transported_features(
    objective: str,
    dataset: str,
    features: np.ndarray,
    steps: int,
    k: int | str,
    seed: int,
    results_root: Path | None,
    device: torch.device | None,
) -> np.ndarray:
    """Push features through a trained field and return the endpoint.

    Args:
        objective: ``"standard"`` or ``"rolled"``.
        dataset: Dataset key.
        features: Unit-norm features of shape ``(num_items, dim)``.
        steps: Euler steps ``T``.
        k: Shots per class the field was trained on.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.
        device: Device to run on; defaults to CUDA when available.

    Returns:
        Transported features of the same shape.
    """
    tag_steps = steps if objective == ROLLED else "any"
    model = load_flow_checkpoint(objective, dataset, k, seed, tag_steps, results_root, device)
    parameters = next(model.parameters())
    trajectory = rollout_trajectory(
        model, torch.from_numpy(features).to(parameters.device), steps
    )
    return trajectory[-1].cpu().numpy()


def plot_feature_comparison(
    dataset: str,
    steps: int = 12,
    k: int | str = K_FULL,
    seed: int = 0,
    num_classes: int = DEFAULT_NUM_CLASSES,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    device: torch.device | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Deliverable 3: original vs standard-FM vs rolled-out features, jointly projected.

    One PCA basis is shared by all three panels, so they are directly comparable. The basis
    is fitted on the *original* features and the prototypes alone, which is exactly the
    Stage 1 embedding basis, so the first panel reproduces the Stage 1 CLIP RN50 panel and
    the other two show where the layer moved that same cloud. Fitting on all three views at
    once instead would let the transported clouds dominate the variance: two thirds of the
    points would sit on the text side of the modality gap, PC1 would become the gap
    direction and the original cloud would collapse to a sliver.

    Args:
        dataset: Dataset key.
        steps: Euler steps ``T`` used to transport the features.
        k: Shots per class the fields were trained on.
        seed: Run seed.
        num_classes: Number of classes to draw; same subset as the Stage 1 embeddings.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        device: Device to run the fields on; defaults to CUDA when available.
        show: Display the figure instead of closing it.
        save: Write the figure to disk.

    Returns:
        Path of the saved figure, or ``None`` when ``save`` is ``False``.
    """
    apply_plot_style()
    class_ids = select_classes(dataset, num_classes)

    features, labels = load_clip_features(dataset, "test", feature_root)
    mask = np.isin(labels, class_ids)
    features, labels = features[mask], labels[mask]
    prototypes = load_text_prototypes(dataset, feature_root)[class_ids]

    views = {
        "Original encoder features": features,
        f"After standard FM (T={steps})": transported_features(
            STANDARD, dataset, features, steps, k, seed, results_root, device
        ),
        f"After rolled-out FM (T={steps})": transported_features(
            ROLLED, dataset, features, steps, k, seed, results_root, device
        ),
    }

    projector = PCA(n_components=2, random_state=PROJECTION_SEED)
    projector.fit(np.concatenate([features, prototypes]))
    frames = {name: projector.transform(value) for name, value in views.items()}
    prototype_xy = projector.transform(prototypes)

    limits = np.concatenate(list(frames.values()) + [prototype_xy])
    low, high = limits.min(axis=0), limits.max(axis=0)
    margin = 0.05 * np.maximum(high - low, 1e-6)

    colors = matplotlib.colormaps[COLORMAP]
    _, _, metadata = load_split(ENCODER, dataset, "test", feature_root)
    class_names = metadata["class_names"]

    fig, axes = plt.subplots(1, len(frames), figsize=(5.2 * len(frames), 4.8))
    for ax, (name, frame) in zip(np.atleast_1d(axes).ravel(), frames.items()):
        for position, class_id in enumerate(class_ids):
            color = colors(position % colors.N)
            points = frame[labels == class_id]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=16,
                alpha=0.55,
                color=color,
                label=normalize_class_name(class_names[class_id]),
            )
            ax.scatter(
                prototype_xy[position, 0],
                prototype_xy[position, 1],
                s=220,
                marker="*",
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
        ax.set_title(name, pad=8)
        ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
        ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
        ax.set_xticks([])
        ax.set_yticks([])

    handles, entries = np.atleast_1d(axes).ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        entries,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle(
        f"{get_spec(dataset).display_name}: feature space before and after the FM layer "
        f"(stars = class prototypes, Stage 1 PCA basis)",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    return save_figure(
        fig, f"flow_feature_comparison_{dataset}", figures_root, show=show, save=save
    )


def plot_flow_trajectories(
    dataset: str,
    objective: str = ROLLED,
    steps: int = 12,
    k: int | str = K_FULL,
    seed: int = 0,
    num_classes: int = 4,
    examples_per_class: int = 2,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    device: torch.device | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Deliverable 4: individual test features traced through the flow.

    Each example is drawn as a connected path with every intermediate Euler state marked,
    from the original feature (circle) to the transported endpoint (square), alongside the
    class prototype (star).

    Args:
        dataset: Dataset key.
        objective: Which trained field to trace.
        steps: Euler steps ``T``.
        k: Shots per class the field was trained on.
        seed: Run seed.
        num_classes: Number of classes to trace.
        examples_per_class: Test features traced per class.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        device: Device to run the field on; defaults to CUDA when available.
        show: Display the figure instead of closing it.
        save: Write the figure to disk.

    Returns:
        Path of the saved figure, or ``None`` when ``save`` is ``False``.
    """
    apply_plot_style()
    class_ids = select_classes(dataset, num_classes)

    features, labels = load_clip_features(dataset, "test", feature_root)
    prototypes = load_text_prototypes(dataset, feature_root)

    rng = np.random.default_rng(PROJECTION_SEED)
    chosen = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == class_id), size=examples_per_class, replace=False)
            for class_id in class_ids
        ]
    )
    selected, selected_labels = features[chosen], labels[chosen]

    tag_steps = steps if objective == ROLLED else "any"
    model = load_flow_checkpoint(objective, dataset, k, seed, tag_steps, results_root, device)
    parameters = next(model.parameters())
    trajectory = (
        rollout_trajectory(model, torch.from_numpy(selected).to(parameters.device), steps)
        .cpu()
        .numpy()
    )

    projector = PCA(n_components=2, random_state=PROJECTION_SEED)
    flat = trajectory.reshape(-1, trajectory.shape[-1])
    projector.fit(np.concatenate([flat, prototypes[class_ids]]))
    paths = np.stack([projector.transform(state) for state in trajectory])
    prototype_xy = projector.transform(prototypes[class_ids])

    colors = matplotlib.colormaps[COLORMAP]
    _, _, metadata = load_split(ENCODER, dataset, "test", feature_root)
    class_names = metadata["class_names"]
    lookup = {int(class_id): index for index, class_id in enumerate(class_ids)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for index in range(len(chosen)):
        position = lookup[int(selected_labels[index])]
        color = colors(position % colors.N)
        path = paths[:, index, :]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.4, alpha=0.8, zorder=2)
        ax.scatter(path[:, 0], path[:, 1], s=14, color=color, alpha=0.7, zorder=3)
        ax.scatter(path[0, 0], path[0, 1], s=90, marker="o", color=color,
                   edgecolor="black", linewidth=1.0, zorder=4)
        ax.scatter(path[-1, 0], path[-1, 1], s=110, marker="s", color=color,
                   edgecolor="black", linewidth=1.0, zorder=4)

    for position, class_id in enumerate(class_ids):
        ax.scatter(
            prototype_xy[position, 0],
            prototype_xy[position, 1],
            s=320,
            marker="*",
            color=colors(position % colors.N),
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
            label=normalize_class_name(class_names[class_id]),
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, framealpha=0.95)
    ax.set_title(
        f"{get_spec(dataset).display_name}: {VARIANT_LABEL[objective]} trajectories, T={steps}\n"
        f"(circle = original feature, square = transported endpoint, star = prototype)",
        pad=12,
    )
    fig.tight_layout()
    return save_figure(
        fig, f"flow_trajectories_{dataset}_{objective}", figures_root, show=show, save=save
    )


def plot_accuracy_vs_t(
    dataset: str,
    objective: str = STANDARD,
    k: int | str = K_FULL,
    seed: int = 0,
    steps: int | str = "any",
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Diagnostic: accuracy along the trajectory, finely integrated.

    Not one of the brief's deliverables. It exists to show whether accuracy peaks before the
    endpoint the brief scores, which is what motivates rolled-out training.

    Args:
        dataset: Dataset key.
        objective: Which trained field to sweep.
        k: Shots per class the field was trained on.
        seed: Run seed.
        steps: Euler steps baked into training; ``"any"`` for standard runs.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        show: Display the figure instead of closing it.
        save: Write the figure to disk.

    Returns:
        Path of the saved figure, or ``None`` when ``save`` is ``False``.
    """
    apply_plot_style()
    record = load_run(objective, dataset, k, seed, steps, results_root)
    if not record["curve"]:
        raise ValueError(
            "This run has no diagnostic curve. Re-run it with with_curve=True."
        )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(record["times"], record["curve"], color="#7b2cbf", linewidth=2, label="Accuracy")
    ax.axhline(
        record["baseline_accuracy"],
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.6,
        label=f"Prototype baseline ({record['baseline_accuracy'] * 100:.1f}%)",
    )

    peak = int(np.argmax(record["curve"]))
    ax.axvline(
        record["times"][peak],
        color="#1a7f37",
        linestyle="-.",
        linewidth=1.6,
        label=f"Peak at t={record['times'][peak]:.2f} ({record['curve'][peak] * 100:.1f}%)",
    )

    ax.set_xlabel("Integration time t", labelpad=8)
    ax.set_ylabel("Top-1 Test Accuracy", labelpad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(
        f"{get_spec(dataset).display_name}: {VARIANT_LABEL[objective]} accuracy along t "
        f"(diagnostic)",
        pad=10,
    )
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    return save_figure(
        fig, f"flow_accuracy_vs_t_{dataset}_{objective}", figures_root, show=show, save=save
    )


def plot_reverse_retrieval(
    dataset: str,
    objective: str = ROLLED,
    steps: int = 12,
    k: int | str = K_FULL,
    seed: int = 0,
    times: tuple[float, ...] = REVERSE_TIMES,
    num_classes: int = DEFAULT_NUM_CLASSES,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    root: Path | None = None,
    device: torch.device | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Optional extra: run the flow backwards from each prototype and render by retrieval.

    Each intermediate point is shown as the nearest real training image in cosine
    similarity, which needs no decoder and stays inside the CLIP RN50 space the field was
    trained in. Cached features were extracted with ``shuffle=False``, so a gallery row index
    is also a dataset index.

    Args:
        dataset: Dataset key.
        objective: Which trained field to reverse.
        steps: Euler steps baked into training when rolled out.
        k: Shots per class the field was trained on.
        seed: Run seed.
        times: Decreasing times to render.
        num_classes: Number of classes shown, one per row.
        feature_root: Feature cache directory; defaults to the resolved feature root.
        results_root: Results directory; defaults to the resolved results root.
        figures_root: Figure directory; defaults to the figures root.
        root: Dataset download directory; defaults to the resolved data root.
        device: Device to run the field on; defaults to CUDA when available.
        show: Display the figure instead of closing it.
        save: Write the figure to disk.

    Returns:
        Path of the saved figure, or ``None`` when ``save`` is ``False``.
    """
    apply_plot_style()
    tag_steps = steps if objective == ROLLED else "any"
    model = load_flow_checkpoint(objective, dataset, k, seed, tag_steps, results_root, device)
    class_ids = select_classes(dataset, num_classes)

    gallery, gallery_labels = load_clip_features(dataset, "train", feature_root)
    prototypes = load_text_prototypes(dataset, feature_root)

    data = build_dataset(dataset, "train", transform=None, root=root)
    class_names = get_class_names(data)
    if len(get_targets(data)) != len(gallery):
        raise ValueError(
            f"{dataset}: {len(gallery)} cached train features but {len(data)} images; "
            "the retrieval indices would not line up."
        )

    parameters = next(model.parameters())
    trajectory = (
        integrate(
            model,
            torch.from_numpy(prototypes[class_ids]).to(parameters.device),
            torch.tensor(times, dtype=torch.float32),
            MANUAL_EULER,
            steps,
        )
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        len(class_ids), len(times), figsize=(2.4 * len(times), 2.8 * len(class_ids))
    )
    grid = np.atleast_2d(axes)

    for row, class_id in enumerate(class_ids):
        for column, moment in enumerate(times):
            ax = grid[row, column]
            indices, similarities = nearest_neighbors(
                trajectory[column, row : row + 1], gallery
            )
            index = int(indices[0, 0])
            image, _ = data[index]
            ax.imshow(image)
            ax.axis("off")

            retrieved = normalize_class_name(class_names[gallery_labels[index]])
            correct = gallery_labels[index] == class_id
            ax.set_title(
                f"t={moment:g}   cos={similarities[0, 0]:.2f}\n{retrieved}",
                fontsize=8,
                pad=4,
                color="#1a7f37" if correct else "#b3261e",
            )

        grid[row, 0].text(
            -0.06,
            0.5,
            normalize_class_name(class_names[class_id]),
            transform=grid[row, 0].transAxes,
            rotation=90,
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="right",
        )

    fig.suptitle(
        f"{get_spec(dataset).display_name}: reverse {VARIANT_LABEL[objective]} from each "
        f"class prototype, rendered by nearest-neighbour retrieval",
        fontsize=13,
        y=1.005,
    )
    fig.tight_layout()
    return save_figure(
        fig, f"flow_reverse_{dataset}", figures_root, show=show, save=save
    )

