import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetSpec:
    """Static, checkable facts about a dataset, taken from the Stage 1 spec.

    Attributes:
        key: Short identifier used on the command line and in result rows.
        display_name: Human-readable name for tables and figures.
        num_classes: Number of classes in the official annotation.
        prompt_template: CLIP prompt containing a ``{class}`` placeholder.
        split_sizes: Official image count per split, asserted after download.
        extra_kwargs: Protocol kwargs forwarded to the torchvision constructor.
        download_size_mb: Approximate archive size, reported before downloading.
    """

    key: str
    display_name: str
    num_classes: int
    prompt_template: str
    split_sizes: dict[str, int] = field(default_factory=dict)
    extra_kwargs: dict = field(default_factory=dict)
    download_size_mb: int = 0

    @property
    def num_images(self) -> int:
        """Total number of images across all official splits."""
        return sum(self.split_sizes.values())


DATASET_SPECS: dict[str, DatasetSpec] = {
    "dtd": DatasetSpec(
        key="dtd",
        display_name="DTD",
        num_classes=47,
        prompt_template="a photo of a {class} texture",
        # Partition 1: 40 train / 40 val / 40 test images per class.
        split_sizes={"train": 1880, "val": 1880, "test": 1880},
        extra_kwargs={"partition": 1},
        download_size_mb=625,
    ),
    "aircraft": DatasetSpec(
        key="aircraft",
        display_name="FGVC-Aircraft",
        num_classes=100,
        prompt_template="a photo of a {class} aircraft",
        split_sizes={"train": 3334, "val": 3333, "test": 3333},
        extra_kwargs={"annotation_level": "variant"},
        download_size_mb=2750,
    ),
}


def default_data_root() -> Path:
    """Resolve the dataset directory: ``FMLAYER_DATA_ROOT``, then Colab, then the repo.

    Returns:
        Directory the datasets are downloaded into. Datasets are never committed,
        so each Colab session re-downloads them.
    """
    env = os.environ.get("FMLAYER_DATA_ROOT")
    if env:
        return Path(env)
    if Path("/content").is_dir():
        return Path("/content/data")
    return REPO_ROOT / "data"


def get_spec(name: str) -> DatasetSpec:
    """Look up a dataset specification by key.

    Args:
        name: Dataset key, ``"dtd"`` or ``"aircraft"``.

    Returns:
        The matching DatasetSpec.
    """
    try:
        return DATASET_SPECS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASET_SPECS)}"
        ) from None
