import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.fmlayer.data.class_names import normalize_class_name
from src.fmlayer.data.datasets import build_dataset, get_class_names, get_targets
from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_ode import DEFAULT_METHOD, DEFAULT_STEPS, MANUAL_EULER, integrate
from src.fmlayer.models.retrieval import nearest_neighbors
from src.fmlayer.train.train_flow_clip import (
    ENCODER,
    curves_path,
    load_clip_features,
    load_flow_checkpoint,
    load_text_prototypes,
)
from src.fmlayer.viz.embeddings import COLORMAP, DEFAULT_NUM_CLASSES, select_classes
from src.fmlayer.viz.figures import apply_plot_style, save_figure

FLOW_COLOR = "#7b2cbf"
ZEROSHOT_COLOR = "#d62728"
ABLATION_COLOR = "#7f7f7f"
STEP_COLORS = ("#c4b5fd", "#7b2cbf", "#3b0764")
SNAPSHOT_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
REVERSE_TIMES = (1.0, 0.75, 0.5, 0.25, 0.0)
PROJECTION_SEED = 0


def load_curve(dataset: str, seed: int = 0, results_root: Path | None = None) -> dict:
    """Read the accuracy-versus-t curve written by a training run.

    Args:
        dataset: Dataset key.
        seed: Run seed.
        results_root: Results directory; defaults to the resolved results root.

    Returns:
        The parsed JSON payload.
    """
    path = curves_path(dataset, seed, results_root)
    if not path.is_file():
        raise FileNotFoundError(f"No flow curve at {path}. Run run_flow_clip() first.")
    return json.loads(path.read_text(encoding="utf-8"))


def draw_accuracy_vs_t(ax: plt.Axes, curve: dict, dataset: str) -> None:
    """Draw one accuracy-versus-t panel, with the two reference levels.

    Args:
        ax: Axes to draw on.
        curve: Payload from :func:`load_curve`.
        dataset: Dataset key, used for the panel title.
    """
    times = curve["times"]
    for color, (steps, values) in zip(STEP_COLORS, sorted(curve["curves"].items(), key=lambda item: int(item[0]))):
        ax.plot(times, values, color=color, linewidth=2, label=f"Euler, {steps} steps")

    ax.axhline(
        curve["zeroshot_accuracy"],
        color=ZEROSHOT_COLOR,
        linestyle="--",
        linewidth=1.6,
        label=f"Zero-shot baseline, t=0 ({curve['zeroshot_accuracy'] * 100:.1f}%)",
    )
    ax.axhline(
        curve["constant_shift_accuracy"],
        color=ABLATION_COLOR,
        linestyle=":",
        linewidth=1.6,
        label=f"Constant-shift ablation ({curve['constant_shift_accuracy'] * 100:.1f}%)",
    )

    ax.set_xlabel("Integration time t", labelpad=8)
    ax.set_ylabel("Top-1 Test Accuracy", labelpad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(get_spec(dataset).display_name, pad=10)
    ax.legend(loc="best", frameon=True, framealpha=0.95)


def plot_accuracy_vs_t(
    dataset: str,
    seed: int = 0,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Plot how accuracy evolves as the flow layer transports the test embeddings."""
    apply_plot_style()
    curve = load_curve(dataset, seed, results_root)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    draw_accuracy_vs_t(ax, curve, dataset)
    fig.suptitle("Accuracy Along the Flow-Matching Trajectory", fontsize=13, y=1.0)
    fig.tight_layout()

    return save_figure(fig, f"flow_accuracy_vs_t_{dataset}", figures_root, show=show, save=save)


def plot_combined_accuracy_vs_t(
    datasets: list[str] | None = None,
    seed: int = 0,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Plot the accuracy-versus-t curves for every dataset side by side."""
    apply_plot_style()
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5))
    for ax, dataset in zip(np.atleast_1d(axes).ravel(), datasets):
        draw_accuracy_vs_t(ax, load_curve(dataset, seed, results_root), dataset)

    fig.suptitle(
        "Accuracy Along the Flow-Matching Trajectory (t = 0 is the zero-shot baseline)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, "flow_accuracy_vs_t_combined", figures_root, show=show, save=save)


def plot_trajectory_embeddings(
    dataset: str,
    seed: int = 0,
    times: tuple[float, ...] = SNAPSHOT_TIMES,
    num_classes: int = DEFAULT_NUM_CLASSES,
    method: str = DEFAULT_METHOD,
    steps: int = DEFAULT_STEPS,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    device: torch.device | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Show the test cloud contracting onto the text prototypes as t advances.

    The PCA basis is fitted once on the t=0 frame and reused for every later frame, so the
    panels share one coordinate system and the motion is real rather than a re-projection.
    """
    apply_plot_style()
    model = load_flow_checkpoint(dataset, seed, results_root, device)
    class_ids = select_classes(dataset, num_classes)

    features, labels = load_clip_features(dataset, "test", feature_root)
    mask = np.isin(labels, class_ids)
    features, labels = features[mask], labels[mask]
    prototypes = load_text_prototypes(dataset, feature_root)

    parameters = next(model.parameters())
    time_grid = torch.tensor(times, dtype=torch.float32)
    trajectory = (
        integrate(
            model,
            torch.from_numpy(features).to(parameters.device),
            time_grid,
            method,
            steps,
        )
        .cpu()
        .numpy()
    )

    projector = PCA(n_components=2, random_state=PROJECTION_SEED)
    projector.fit(np.concatenate([trajectory[0], prototypes[class_ids]]))
    prototype_xy = projector.transform(prototypes[class_ids])
    frames = [projector.transform(state) for state in trajectory]

    limits = np.concatenate(frames + [prototype_xy])
    low, high = limits.min(axis=0), limits.max(axis=0)
    margin = 0.05 * np.maximum(high - low, 1e-6)
    colors = matplotlib.colormaps[COLORMAP]
    _, _, metadata = load_split(ENCODER, dataset, "test", feature_root)
    class_names = metadata["class_names"]

    fig, axes = plt.subplots(1, len(times), figsize=(3.6 * len(times), 4.0))
    for ax, moment, frame in zip(np.atleast_1d(axes).ravel(), times, frames):
        for position, class_id in enumerate(class_ids):
            color = colors(position % colors.N)
            points = frame[labels == class_id]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=14,
                alpha=0.5,
                color=color,
                label=normalize_class_name(class_names[class_id]),
            )
            ax.scatter(
                prototype_xy[position, 0],
                prototype_xy[position, 1],
                s=200,
                marker="*",
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
        ax.set_title(f"t = {moment:g}", pad=8)
        ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
        ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
        ax.set_xticks([])
        ax.set_yticks([])

    handles, entries = np.atleast_1d(axes).ravel()[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        entries,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle(
        f"{get_spec(dataset).display_name}: test embeddings transported towards the text "
        f"prototypes (stars), shared PCA basis",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    return save_figure(
        fig, f"flow_trajectory_{dataset}", figures_root, show=show, save=save
    )


def plot_reverse_retrieval(
    dataset: str,
    seed: int = 0,
    times: tuple[float, ...] = REVERSE_TIMES,
    num_classes: int = DEFAULT_NUM_CLASSES,
    steps: int = DEFAULT_STEPS,
    feature_root: Path | None = None,
    results_root: Path | None = None,
    figures_root: Path | None = None,
    root: Path | None = None,
    device: torch.device | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Run the flow backwards from each text prototype and render it by image retrieval.

    Flow matching is time-symmetric, so the same field integrated from t=1 down to t=0
    carries a class text embedding back into image-embedding space. Each intermediate point
    is rendered as the nearest real training image in cosine similarity, which needs no
    decoder and stays inside the CLIP RN50 space the field was trained in.

    Cached features were extracted with ``shuffle=False``, so a gallery row index is also a
    dataset index.
    """
    apply_plot_style()
    model = load_flow_checkpoint(dataset, seed, results_root, device)
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
    time_grid = torch.tensor(times, dtype=torch.float32)
    trajectory = (
        integrate(
            model,
            torch.from_numpy(prototypes[class_ids]).to(parameters.device),
            time_grid,
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
        f"{get_spec(dataset).display_name}: reverse flow from each class text embedding, "
        f"rendered by nearest-neighbour retrieval",
        fontsize=13,
        y=1.005,
    )
    fig.tight_layout()
    return save_figure(fig, f"flow_reverse_{dataset}", figures_root, show=show, save=save)
