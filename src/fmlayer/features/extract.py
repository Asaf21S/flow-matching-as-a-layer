from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.fmlayer.data.class_names import build_prompts
from src.fmlayer.data.datasets import build_dataset, get_class_names
from src.fmlayer.data.specs import SPLITS, Split, get_spec
from src.fmlayer.encoders.base import Encoder, default_device
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.encoders.registry import build_encoder, feature_cells
from src.fmlayer.features.cache import feature_path, is_cached, save_features

BATCH_SIZE = 64
NUM_WORKERS = 2


def build_metadata(
    encoder: Encoder, dataset: str, split: str | None, num_items: int, class_names: list[str]
) -> dict:
    """Describe how a cache was produced, for integrity checks and provenance.

    Args:
        encoder: Encoder that produced the features.
        dataset: Dataset key.
        split: Official split, or ``None`` for text prototypes.
        num_items: Number of rows in the cache.
        class_names: Raw class names in label-index order.

    Returns:
        The metadata dict stored next to the arrays.
    """
    return {
        "encoder": encoder.name,
        "dataset": dataset,
        "split": split,
        "embed_dim": encoder.embed_dim,
        "num_items": num_items,
        "transform": str(encoder.transform),
        "class_names": class_names,
    }


def extract_split(
    encoder: Encoder,
    dataset: str,
    split: Split,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    data_root: Path | None = None,
    feature_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Run one forward pass over a split and cache the resulting features.

    Args:
        encoder: Frozen encoder supplying both the preprocessing and the model.
        dataset: Dataset key.
        split: Official split to encode.
        batch_size: Images per forward pass.
        num_workers: DataLoader worker processes.
        data_root: Dataset directory; defaults to the resolved data root.
        feature_root: Cache directory; defaults to the resolved feature root.
        overwrite: Recompute even when a matching cache exists.

    Returns:
        Path of the written (or reused) cache file.
    """
    spec = get_spec(dataset)
    images_dataset = build_dataset(dataset, split, encoder.transform, data_root)
    class_names = get_class_names(images_dataset)
    metadata = build_metadata(encoder, dataset, split, len(images_dataset), class_names)
    path = feature_path(encoder.name, dataset, split, feature_root)

    if not overwrite and is_cached(path, metadata):
        print(f"    [cached] {encoder.name}/{dataset}/{split} -> {path.name}")
        return path

    # shuffle=False keeps features aligned with the labels the loader yields.
    loader = DataLoader(
        images_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=encoder.device.type == "cuda",
    )

    feature_batches = []
    label_batches = []
    for images, targets in tqdm(loader, desc=f"{encoder.name}/{dataset}/{split}", leave=False):
        feature_batches.append(encoder.embed_images(images).cpu().numpy())
        label_batches.append(targets.numpy())

    features = np.concatenate(feature_batches)
    labels = np.concatenate(label_batches)

    expected = spec.split_sizes.get(split)
    assert expected is None or len(features) == expected, (
        f"{dataset}/{split}: encoded {len(features)} images, expected {expected}"
    )
    assert features.shape[1] == encoder.embed_dim

    save_features(path, features, labels, metadata)
    print(f"    [ok]     {encoder.name}/{dataset}/{split} -> {features.shape} {path.name}")
    return path


def extract_text_prototypes(
    encoder: ClipRN50Encoder,
    dataset: str,
    data_root: Path | None = None,
    feature_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Encode one CLIP prompt per class and cache the text prototypes.

    Args:
        encoder: The frozen CLIP RN50 encoder.
        dataset: Dataset key, which fixes the prompt template.
        data_root: Dataset directory, used only to read the class names.
        feature_root: Cache directory; defaults to the resolved feature root.
        overwrite: Recompute even when a matching cache exists.

    Returns:
        Path of the written (or reused) cache file.
    """
    # transform=None avoids decoding any image; only `.classes` is needed here.
    class_names = get_class_names(build_dataset(dataset, "train", None, data_root))
    metadata = build_metadata(encoder, dataset, None, len(class_names), class_names)
    path = feature_path(encoder.name, dataset, None, feature_root)

    if not overwrite and is_cached(path, metadata):
        print(f"    [cached] {encoder.name}/{dataset}/text -> {path.name}")
        return path

    prompts = build_prompts(dataset, class_names)
    features = encoder.embed_texts(prompts).cpu().numpy()
    labels = np.arange(len(prompts))

    save_features(path, features, labels, {**metadata, "prompts": prompts})
    print(f"    [ok]     {encoder.name}/{dataset}/text -> {features.shape} {path.name}")
    return path


def extract_all(
    cells: list[tuple[str, str]] | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    data_root: Path | None = None,
    feature_root: Path | None = None,
    overwrite: bool = False,
    device: torch.device | None = None,
) -> dict:
    """Cache every feature file Stage 1 needs, loading each encoder only once.

    Args:
        cells: ``(encoder, dataset)`` pairs; defaults to the full Stage 1 set.
        batch_size: Images per forward pass.
        num_workers: DataLoader worker processes.
        data_root: Dataset directory; defaults to the resolved data root.
        feature_root: Cache directory; defaults to the resolved feature root.
        overwrite: Recompute even when matching caches exist.
        device: Device to run on; defaults to CUDA when available.

    Returns:
        The written cache paths, keyed by ``"encoder/dataset/split"``.
    """
    cells = cells if cells is not None else feature_cells()
    device = device if device is not None else default_device()
    print(f"Device: {device}")

    datasets_by_encoder: dict[str, list[str]] = {}
    for encoder_name, dataset in cells:
        datasets_by_encoder.setdefault(encoder_name, []).append(dataset)

    paths = {}
    for encoder_name, datasets in datasets_by_encoder.items():
        print(f"\n=== {encoder_name} ===")
        encoder = build_encoder(encoder_name, device)
        for dataset in datasets:
            for split in SPLITS:
                paths[f"{encoder_name}/{dataset}/{split}"] = extract_split(
                    encoder,
                    dataset,
                    split,
                    batch_size,
                    num_workers,
                    data_root,
                    feature_root,
                    overwrite,
                )
            if isinstance(encoder, ClipRN50Encoder):
                paths[f"{encoder_name}/{dataset}/text"] = extract_text_prototypes(
                    encoder, dataset, data_root, feature_root, overwrite
                )
        del encoder
        torch.cuda.empty_cache()

    print(f"\n{len(paths)} feature files ready.")
    return paths
