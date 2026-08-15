import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA

from src.fmlayer.data.class_names import normalize_class_name
from src.fmlayer.encoders.base import default_device
from src.fmlayer.viz.embeddings import COLORMAP, DEFAULT_NUM_CLASSES, select_classes
from src.fmlayer.viz.figures import apply_plot_style, save_figure


def plot_flow_vector_field_2d(
    fm_layer: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
    class_ids: np.ndarray | None = None,
    t: float = 0.0,
    grid_res: int = 25,
    device: torch.device | None = None,
    figures_root: str | None = None,
    show: bool = True,
    save: bool = False,
) -> None:
    """Visualize 2D PCA projection of the Flow Matching vector field v_theta(z, t)."""
    apply_plot_style()
    device = device if device is not None else default_device()
    dataset = metadata.get("dataset", "dtd")
    class_names = metadata.get("class_names", [])

    class_ids = class_ids if class_ids is not None else select_classes(dataset, DEFAULT_NUM_CLASSES)
    mask = np.isin(labels, class_ids)
    features_sub = features[mask]
    labels_sub = labels[mask]

    pca = PCA(n_components=2, random_state=0)
    features_2d = pca.fit_transform(features_sub)

    x_min, x_max = features_2d[:, 0].min() - 0.5, features_2d[:, 0].max() + 0.5
    y_min, y_max = features_2d[:, 1].min() - 0.5, features_2d[:, 1].max() + 0.5

    u1 = np.linspace(x_min, x_max, grid_res)
    u2 = np.linspace(y_min, y_max, grid_res)
    U1, U2 = np.meshgrid(u1, u2)
    grid_2d = np.column_stack([U1.ravel(), U2.ravel()])

    grid_high = pca.inverse_transform(grid_2d)
    grid_tensor = torch.from_numpy(grid_high).float().to(device)

    fm_layer.eval()
    with torch.no_grad():
        v_high = fm_layer(grid_tensor, t).cpu().numpy()

    # Project high-dimensional velocity onto PCA principal components
    v_2d = v_high @ pca.components_.T
    V1 = v_2d[:, 0].reshape(U1.shape)
    V2 = v_2d[:, 1].reshape(U2.shape)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = matplotlib.colormaps[COLORMAP]

    for position, class_id in enumerate(class_ids):
        color = colors(position % colors.N)
        points = features_2d[labels_sub == class_id]
        cname = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=25,
            alpha=0.6,
            color=color,
            label=normalize_class_name(cname),
        )

    # Overlay streamplot vector field lines
    speed = np.sqrt(V1**2 + V2**2)
    ax.streamplot(
        u1,
        u2,
        V1,
        V2,
        color=speed,
        cmap="autumn",
        linewidth=1.2,
        density=1.2,
        arrowsize=1.2,
    )

    objective = metadata.get("objective", "standard")
    target_type = metadata.get("target_type", "centroids")
    title = f"[{objective} - {target_type}] Vector Field $v_\\theta(z_t, t={t:.1f})$ | {dataset.upper()}"

    ax.set_title(
        title,
        pad=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, framealpha=0.95)
    fig.tight_layout()

    save_figure(fig, f"flow_vector_field_t{int(t*10)}", figures_root, show=show, save=save)


def plot_flow_trajectories_2d(
    fm_layer: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
    class_ids: np.ndarray | None = None,
    num_steps: int = 12,
    samples_per_class: int = 2,
    device: torch.device | None = None,
    figures_root: str | None = None,
    show: bool = True,
    save: bool = False,
) -> None:
    """Visualize discrete sample ODE flow trajectories z0 -> z1 in 2D PCA space."""
    apply_plot_style()
    device = device if device is not None else default_device()
    dataset = metadata.get("dataset", "dtd")
    class_names = metadata.get("class_names", [])

    class_ids = class_ids if class_ids is not None else select_classes(dataset, DEFAULT_NUM_CLASSES)

    # Select representative samples per class
    sample_indices = []
    rng = np.random.default_rng(0)
    for cid in class_ids:
        c_idxs = np.where(labels == cid)[0]
        if len(c_idxs) > 0:
            sample_indices.extend(rng.choice(c_idxs, size=min(samples_per_class, len(c_idxs)), replace=False))

    sample_features = features[sample_indices]
    sample_labels = labels[sample_indices]

    # Integrate Euler steps and record trajectory states
    dt = 1.0 / num_steps
    z_tensor = torch.from_numpy(sample_features).float().to(device)
    trajectories = [z_tensor.cpu().numpy()]

    fm_layer.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = i * dt
            v = fm_layer(z_tensor, t)
            z_tensor = z_tensor + v * dt
            trajectories.append(z_tensor.cpu().numpy())

    all_states = np.concatenate(trajectories, axis=0)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_states)

    trajectories_2d = np.array([pca.transform(traj) for traj in trajectories])  # (steps+1, num_samples, 2)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = matplotlib.colormaps[COLORMAP]

    for position, class_id in enumerate(class_ids):
        color = colors(position % colors.N)
        sample_mask = sample_labels == class_id
        cname = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

        for idx in np.where(sample_mask)[0]:
            path = trajectories_2d[:, idx, :]  # (steps+1, 2)
            # Plot trajectory curve
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=0.8, marker=".", markersize=4)
            # Start point (z0)
            ax.scatter(path[0, 0], path[0, 1], color=color, s=40, marker="o", edgecolors="black", linewidths=0.5)
            # End point (z1)
            ax.scatter(path[-1, 0], path[-1, 1], color=color, s=80, marker="^", edgecolors="black", linewidths=1.0)

        # Legend entry
        ax.scatter([], [], color=color, label=normalize_class_name(cname), s=40)

    objective = metadata.get("objective", "standard")
    target_type = metadata.get("target_type", "centroids")
    title = f"[{objective} - {target_type}] ODE Sample Trajectories ($z_0 \\rightarrow z_1$) | {dataset.upper()}"

    ax.set_title(
        title,
        pad=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, framealpha=0.95)
    fig.tight_layout()

    save_figure(fig, "flow_trajectories_2d", figures_root, show=show, save=save)


def plot_before_after_embeddings(
    fm_layer: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
    class_ids: np.ndarray | None = None,
    device: torch.device | None = None,
    figures_root: str | None = None,
    show: bool = True,
    save: bool = False,
) -> None:
    """Side-by-side 2D PCA comparison of raw features z0 vs flow-transformed features z1."""
    apply_plot_style()
    device = device if device is not None else default_device()
    dataset = metadata.get("dataset", "dtd")
    class_names = metadata.get("class_names", [])

    class_ids = class_ids if class_ids is not None else select_classes(dataset, DEFAULT_NUM_CLASSES)
    mask = np.isin(labels, class_ids)
    features_sub = features[mask]
    labels_sub = labels[mask]

    from src.fmlayer.models.flow_ode import rollout
    # Transform features with Flow Matching layer
    z0_tensor = torch.from_numpy(features_sub).float().to(device)
    fm_layer.eval()
    with torch.no_grad():
        z1_tensor, _ = rollout(fm_layer, z0_tensor, 12)
    z1_sub = z1_tensor.cpu().numpy()

    # Fit PCA jointly on z0 and z1
    pca = PCA(n_components=2, random_state=0)
    all_combined = np.concatenate([features_sub, z1_sub], axis=0)
    pca.fit(all_combined)

    z0_2d = pca.transform(features_sub)
    z1_2d = pca.transform(z1_sub)

    colors = matplotlib.colormaps[COLORMAP]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    for position, class_id in enumerate(class_ids):
        color = colors(position % colors.N)
        cname = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

        p0 = z0_2d[labels_sub == class_id]
        ax1.scatter(p0[:, 0], p0[:, 1], s=25, alpha=0.6, color=color, label=normalize_class_name(cname))

        p1 = z1_2d[labels_sub == class_id]
        ax2.scatter(p1[:, 0], p1[:, 1], s=25, alpha=0.6, color=color)

    ax1.set_title("Before: Raw Frozen Features ($z_0$)", pad=8)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_title("After: Flow-Transformed Features ($z_1$)", pad=8)
    ax2.set_xticks([])
    ax2.set_yticks([])

    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=True, framealpha=0.95)
    
    objective = metadata.get("objective", "standard")
    target_type = metadata.get("target_type", "centroids")
    title = f"[{objective} - {target_type}] Representation Refinement | {dataset.upper()}"
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    save_figure(fig, "flow_before_after_pca", figures_root, show=show, save=save)
