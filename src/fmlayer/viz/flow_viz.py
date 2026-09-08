from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from src.fmlayer.data.class_names import normalize_class_name
from src.fmlayer.encoders.base import default_device
from src.fmlayer.models.flow_ode import rollout
from src.fmlayer.viz.embeddings import COLORMAP, DEFAULT_NUM_CLASSES, select_classes
from src.fmlayer.viz.figures import apply_plot_style, save_figure

TRAJECTORY_SEED = 0


def class_label(class_names: list[str], class_id: int) -> str:
    """Readable name of one class, falling back to its index."""
    if class_id < len(class_names):
        return normalize_class_name(class_names[class_id])
    return f"Class {class_id}"


def transported(field: nn.Module, features: np.ndarray, steps: int, device: torch.device) -> np.ndarray:
    """Run the flow over a feature array and return the transported array."""
    field.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(features)).float().to(device)
    with torch.no_grad():
        final, _ = rollout(field, tensor, steps)
    return final.cpu().numpy()


def trajectory_states(
    field: nn.Module, features: np.ndarray, steps: int, device: torch.device
) -> np.ndarray:
    """Every intermediate state of the flow, shape ``(steps + 1, num_items, dim)``."""
    field.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(features)).float().to(device)
    with torch.no_grad():
        _, states = rollout(field, tensor, steps)
    return states.cpu().numpy()


def sample_per_class(
    labels: np.ndarray, class_ids: np.ndarray, per_class: int, seed: int = TRAJECTORY_SEED
) -> np.ndarray:
    """Pick a few example indices from each requested class."""
    rng = np.random.default_rng(seed)
    picked = []
    for class_id in class_ids:
        members = np.flatnonzero(labels == class_id)
        if len(members):
            picked.extend(rng.choice(members, size=min(per_class, len(members)), replace=False))
    return np.asarray(picked, dtype=int)


def fit_joint_pca(blocks: list[np.ndarray], seed: int = 0) -> PCA:
    """Fit one PCA over every feature set that will be compared.

    Fitting jointly is what makes the panels of a comparison figure readable against
    each other and against the class targets.
    """
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(np.concatenate([block for block in blocks if len(block)], axis=0))
    return pca


def draw_targets(ax, target_xy: np.ndarray, class_ids: np.ndarray, colors) -> None:
    """Overlay the class targets as stars."""
    for position, _ in enumerate(class_ids):
        ax.scatter(
            target_xy[position, 0],
            target_xy[position, 1],
            s=260,
            marker="*",
            color=colors(position % colors.N),
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )


def curve_series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Epochs and values of one logged metric, skipping the epochs that lack it."""
    points = [(entry["epoch"], entry[key]) for entry in history if key in entry]
    return [epoch for epoch, _ in points], [value for _, value in points]


def draw_training_curves(ax, history: list[dict], title: str) -> None:
    """Plot the training loss with the train and validation accuracy on a twin axis."""
    epochs, losses = curve_series(history, "train_loss")
    ax.plot(epochs, losses, color="#1f77b4", linewidth=2, label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    if losses and min(losses) > 0:
        ax.set_yscale("log")

    val_epochs, val_accuracy = curve_series(history, "val_accuracy")
    if val_epochs:
        twin = ax.twinx()
        twin.plot(val_epochs, val_accuracy, color="#d62728", linewidth=2, label="val accuracy")
        train_epochs, train_accuracy = curve_series(history, "train_accuracy")
        if train_epochs:
            twin.plot(
                train_epochs, train_accuracy, color="#d62728", linewidth=1.6,
                linestyle="--", alpha=0.8, label="train accuracy",
            )
        twin.set_ylabel("Accuracy", color="#d62728")
        twin.tick_params(axis="y", labelcolor="#d62728")
        twin.grid(False)
        twin.legend(loc="lower right", fontsize=8)
    ax.set_title(title, pad=8)



def draw_vector_field(
    ax,
    field: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    targets: np.ndarray | None = None,
    t: float = 0.0,
    grid_res: int = 22,
    device: torch.device | None = None,
) -> None:
    """Stream-plot the velocity field in the 2D PCA plane of the shown classes."""
    device = device if device is not None else default_device()
    colors = matplotlib.colormaps[COLORMAP]

    mask = np.isin(labels, class_ids)
    subset, subset_labels = features[mask], labels[mask]
    target_subset = targets[class_ids] if targets is not None else np.empty((0, features.shape[1]))

    pca = fit_joint_pca([subset, target_subset])
    subset_xy = pca.transform(subset)

    x_min, x_max = subset_xy[:, 0].min() - 0.5, subset_xy[:, 0].max() + 0.5
    y_min, y_max = subset_xy[:, 1].min() - 0.5, subset_xy[:, 1].max() + 0.5
    axis_x = np.linspace(x_min, x_max, grid_res)
    axis_y = np.linspace(y_min, y_max, grid_res)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)

    grid_high = pca.inverse_transform(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    field.eval()
    with torch.no_grad():
        velocity = field(torch.from_numpy(grid_high).float().to(device), t).cpu().numpy()

    velocity_2d = velocity @ pca.components_.T
    component_x = velocity_2d[:, 0].reshape(grid_x.shape)
    component_y = velocity_2d[:, 1].reshape(grid_y.shape)

    for position, class_id in enumerate(class_ids):
        points = subset_xy[subset_labels == class_id]
        ax.scatter(
            points[:, 0], points[:, 1], s=20, alpha=0.55,
            color=colors(position % colors.N), label=class_label(class_names, class_id),
        )
    if targets is not None:
        draw_targets(ax, pca.transform(target_subset), class_ids, colors)

    ax.streamplot(
        axis_x, axis_y, component_x, component_y,
        color=np.sqrt(component_x**2 + component_y**2),
        cmap="autumn", linewidth=1.1, density=1.1, arrowsize=1.1,
    )
    ax.set_title(f"Vector field $v_\\theta(z, t={t:.1f})$", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_trajectories(
    ax,
    field: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    targets: np.ndarray | None = None,
    steps: int = 12,
    per_class: int = 2,
    device: torch.device | None = None,
) -> None:
    """Draw a few ODE trajectories with their origin, endpoint and class target."""
    device = device if device is not None else default_device()
    colors = matplotlib.colormaps[COLORMAP]

    picked = sample_per_class(labels, class_ids, per_class)
    states = trajectory_states(field, features[picked], steps, device)
    picked_labels = labels[picked]
    target_subset = targets[class_ids] if targets is not None else np.empty((0, features.shape[1]))

    # The projection spans the whole trajectory and the targets, so endpoints and
    # prototypes are comparable in the same plane.
    pca = fit_joint_pca([states.reshape(-1, states.shape[-1]), target_subset])
    states_xy = np.stack([pca.transform(state) for state in states])

    for position, class_id in enumerate(class_ids):
        color = colors(position % colors.N)
        for index in np.flatnonzero(picked_labels == class_id):
            path = states_xy[:, index, :]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.6, alpha=0.85, marker=".", markersize=4)
            ax.scatter(path[0, 0], path[0, 1], color=color, s=42, marker="o", edgecolors="black", linewidths=0.5)
            ax.scatter(path[-1, 0], path[-1, 1], color=color, s=84, marker="^", edgecolors="black", linewidths=1.0)
        ax.scatter([], [], color=color, label=class_label(class_names, class_id), s=40)

    if targets is not None:
        draw_targets(ax, pca.transform(target_subset), class_ids, colors)

    ax.set_title(f"Trajectories $z_0 \\rightarrow z_T$ (T={steps})", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_flow_dynamics(
    result: dict,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    targets: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    step_counts: tuple[int, ...] = (4, 12),
    num_classes: int = DEFAULT_NUM_CLASSES,
    device: torch.device | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Training curves, vector field and trajectories for one run, on a single row.

    Args:
        result: One entry returned by :func:`run_stage3`.
        features: Features to visualise, typically the validation split.
        labels: Labels matching ``features``.
        class_names: Class names of the dataset.
        targets: Optional ``(num_classes, dim)`` target table to overlay as stars.
        class_ids: Classes to show; defaults to the shared readable subset.
        step_counts: Trajectory step counts, one panel each.
        num_classes: Number of classes to show when ``class_ids`` is not given.
        device: Device to evaluate the field on.
        figures_root: Output directory.
        show: Display the figure.
        save: Write a PNG.

    Returns:
        Path of the saved figure, or ``None``.
    """
    apply_plot_style()
    dataset = result["dataset"]
    class_ids = class_ids if class_ids is not None else select_classes(dataset, num_classes)
    field = result["fm_layer"]

    columns = 2 + len(step_counts)
    fig, axes = plt.subplots(1, columns, figsize=(5.0 * columns, 4.4))

    draw_training_curves(axes[0], result["history"], "Training loss / val accuracy")
    draw_vector_field(
        axes[1], field, features, labels, class_ids, class_names, targets, 0.0, device=device
    )
    for offset, steps in enumerate(step_counts):
        draw_trajectories(
            axes[2 + offset], field, features, labels, class_ids, class_names,
            targets, steps, device=device,
        )

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.suptitle(
        f"[{result['config_name']}]  {dataset.upper()} ({result['encoder']}) "
        f"K={result['k']} seed={result['seed']}",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()

    name = (
        f"viz_{result['config_name']}_{dataset}_{result['encoder']}"
        f"_k{result['k']}_seed{result['seed']}"
    )
    return save_figure(fig, name, figures_root, show=show, save=save)


def plot_feature_comparison(
    fields: dict[str, nn.Module],
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    dataset: str,
    encoder: str = "",
    targets: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    steps: int = 12,
    num_classes: int = DEFAULT_NUM_CLASSES,
    device: torch.device | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Compare the original features against several transported versions.

    Every panel shares one projection, fitted jointly over all the feature sets and
    the class targets, so the same point is comparable across panels.

    Args:
        fields: Named velocity fields, e.g. ``{"standard FM": ..., "rolled FM": ...}``.
        features: Features to visualise, typically the test split.
        labels: Labels matching ``features``.
        class_names: Class names of the dataset.
        dataset: Dataset key, used for the title and the class selection.
        encoder: Encoder key, kept in the figure name so cells cannot overwrite each other.
        targets: Optional ``(num_classes, dim)`` target table to overlay as stars.
        class_ids: Classes to show; defaults to the shared readable subset.
        steps: Euler steps used to transport the features.
        num_classes: Number of classes to show when ``class_ids`` is not given.
        device: Device to evaluate the fields on.
        figures_root: Output directory.
        show: Display the figure.
        save: Write a PNG.

    Returns:
        Path of the saved figure, or ``None``.
    """
    apply_plot_style()
    device = device if device is not None else default_device()
    class_ids = class_ids if class_ids is not None else select_classes(dataset, num_classes)
    colors = matplotlib.colormaps[COLORMAP]

    mask = np.isin(labels, class_ids)
    subset, subset_labels = features[mask], labels[mask]
    views = {"Original features": subset}
    for name, field in fields.items():
        views[name] = transported(field, subset, steps, device)

    target_subset = targets[class_ids] if targets is not None else np.empty((0, features.shape[1]))
    pca = fit_joint_pca(list(views.values()) + [target_subset])
    target_xy = pca.transform(target_subset) if targets is not None else None

    fig, axes = plt.subplots(1, len(views), figsize=(4.8 * len(views), 4.4))
    axes = np.atleast_1d(axes)

    for ax, (name, view) in zip(axes, views.items()):
        coordinates = pca.transform(view)
        for position, class_id in enumerate(class_ids):
            points = coordinates[subset_labels == class_id]
            ax.scatter(
                points[:, 0], points[:, 1], s=20, alpha=0.55,
                color=colors(position % colors.N), label=class_label(class_names, class_id),
            )
        if target_xy is not None:
            draw_targets(ax, target_xy, class_ids, colors)
        ax.set_title(name, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.suptitle(
        f"{dataset.upper()}{f' ({encoder})' if encoder else ''}: feature space before and "
        f"after the flow (T={steps}, joint PCA)",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    suffix = f"_{encoder}" if encoder else ""
    return save_figure(
        fig, f"flow_comparison_{dataset}{suffix}_T{steps}", figures_root, show=show, save=save
    )
