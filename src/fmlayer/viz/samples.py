from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fmlayer.data.class_names import normalize_class_name
from src.fmlayer.data.datasets import build_dataset, get_class_names, get_targets
from src.fmlayer.data.specs import DATASET_SPECS, get_spec
from src.fmlayer.viz.embeddings import CLASS_SELECTION_SEED, select_classes
from src.fmlayer.viz.figures import apply_plot_style, save_figure

SAMPLES_SPLIT = "test"
SAMPLES_COLUMNS = 4


def sample_indices(
    targets: np.ndarray, class_ids: np.ndarray, seed: int = CLASS_SELECTION_SEED
) -> list[int]:
    """Pick one example index per class, reproducibly."""
    rng = np.random.default_rng(seed)
    indices = []
    for class_id in class_ids:
        candidates = np.flatnonzero(targets == class_id)
        if not len(candidates):
            raise ValueError(f"No examples for class {class_id}.")
        indices.append(int(rng.choice(candidates)))
    return indices


def plot_dataset_samples(
    dataset: str,
    class_ids: np.ndarray | None = None,
    split: str = SAMPLES_SPLIT,
    columns: int = SAMPLES_COLUMNS,
    seed: int = CLASS_SELECTION_SEED,
    root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Show one example image per selected class, titled with its ground-truth label."""
    apply_plot_style()
    class_ids = class_ids if class_ids is not None else select_classes(dataset)

    data = build_dataset(dataset, split, transform=None, root=root)
    targets = get_targets(data)
    class_names = get_class_names(data)
    indices = sample_indices(targets, class_ids, seed)

    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.3 * rows))
    flat = np.atleast_1d(axes).ravel()

    for ax, index in zip(flat, indices):
        image, label = data[index]
        ax.imshow(image)
        ax.set_title(normalize_class_name(class_names[label]), fontsize=10, pad=6)
        ax.axis("off")

    for ax in flat[len(indices) :]:
        ax.axis("off")

    fig.suptitle(
        f"{get_spec(dataset).display_name}: example {split} images with ground-truth labels",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, f"samples_{dataset}", figures_root, show=show, save=save)


def plot_samples_for_datasets(
    datasets: list[str] | None = None,
    split: str = SAMPLES_SPLIT,
    columns: int = SAMPLES_COLUMNS,
    seed: int = CLASS_SELECTION_SEED,
    root: Path | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> list[Path | None]:
    """Show example images with their labels for several datasets."""
    datasets = datasets if datasets is not None else sorted(DATASET_SPECS)
    return [
        plot_dataset_samples(
            dataset,
            split=split,
            columns=columns,
            seed=seed,
            root=root,
            figures_root=figures_root,
            show=show,
            save=save,
        )
        for dataset in datasets
    ]
