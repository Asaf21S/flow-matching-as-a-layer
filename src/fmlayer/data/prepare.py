from pathlib import Path

from src.fmlayer.data.class_names import build_prompts
from src.fmlayer.data.datasets import build_dataset, get_class_names, verify_split
from src.fmlayer.data.specs import DATASET_SPECS, SPLITS, default_data_root, get_spec


def prepare_dataset(name: str, root: Path | None = None, download: bool = True) -> dict:
    """Download one dataset and verify every official split against the Stage 1 protocol.

    Args:
        name: Dataset key, ``"dtd"`` or ``"aircraft"``.
        root: Download directory; defaults to :func:`default_data_root`.
        download: Whether to fetch missing archives.

    Returns:
        A report with the per-split summaries, the class names and the CLIP prompts.
        Raises AssertionError if a split does not match the protocol.
    """
    spec = get_spec(name)
    root = Path(root) if root is not None else default_data_root()

    print(f"=== {spec.display_name} ({spec.key}) ===")
    print(f"    protocol: {spec.extra_kwargs}, approx. {spec.download_size_mb} MB")

    split_reports = []
    class_names: list[str] = []
    for split in SPLITS:
        dataset = build_dataset(name, split, None, root, download)
        info = verify_split(dataset, spec, split)
        class_names = get_class_names(dataset)
        split_reports.append(info)
        print(
            f"    [ok] {split:<5} {info['images']:>6} images, {info['classes']} classes, "
            f"{info['min_per_class']}-{info['max_per_class']} per class"
        )

    prompts = build_prompts(name, class_names)
    print(f"    prompt: {prompts[0]!r} ... {prompts[-1]!r}")

    return {
        "display_name": spec.display_name,
        "num_classes": spec.num_classes,
        "extra_kwargs": spec.extra_kwargs,
        "splits": split_reports,
        "class_names": class_names,
        "prompts": prompts,
    }


def prepare_datasets(
    names: list[str] | None = None, root: Path | None = None, download: bool = True
) -> dict:
    """Download and verify several datasets in one call.

    Args:
        names: Dataset keys; defaults to every Stage 1 dataset.
        root: Download directory; defaults to :func:`default_data_root`.
        download: Whether to fetch missing archives.

    Returns:
        The data root and one report per dataset, keyed by dataset key.
    """
    names = names if names is not None else sorted(DATASET_SPECS)
    root = Path(root) if root is not None else default_data_root()
    print(f"Data root: {root}\n")

    return {
        "root": str(root),
        "datasets": {name: prepare_dataset(name, root, download) for name in names},
    }

