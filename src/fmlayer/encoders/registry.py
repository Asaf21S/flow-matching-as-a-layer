import torch

from src.fmlayer.encoders.base import Encoder, default_device
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.encoders.dinov2 import Dinov2Encoder
from src.fmlayer.encoders.resnet18 import ResNet18Encoder

ENCODER_CLASSES = {
    ResNet18Encoder.NAME: ResNet18Encoder,
    Dinov2Encoder.NAME: Dinov2Encoder,
    ClipRN50Encoder.NAME: ClipRN50Encoder,
}

EMBED_DIMS = {name: cls.EMBED_DIM for name, cls in ENCODER_CLASSES.items()}

# Which encoder is applied to which dataset, per the Stage 1 protocol.
ENCODER_DATASETS = {
    "resnet18": ("dtd", "aircraft"),
    "dinov2_vits14": ("dtd",),
    "clip_rn50": ("dtd", "aircraft"),
}

# Encoder/dataset pairs the linear probe is trained on; CLIP is zero-shot only.
LINEAR_PROBE_CELLS = (("resnet18", "dtd"), ("resnet18", "aircraft"), ("dinov2_vits14", "dtd"))


def build_encoder(name: str, device: torch.device | None = None) -> Encoder:
    """Instantiate a frozen encoder by key.

    Args:
        name: Encoder key, ``"resnet18"``, ``"dinov2_vits14"`` or ``"clip_rn50"``.
        device: Device to load onto; defaults to CUDA when available.

    Returns:
        The frozen encoder, in eval mode with gradients disabled.
    """
    if name not in ENCODER_CLASSES:
        raise KeyError(f"Unknown encoder {name!r}. Available: {sorted(ENCODER_CLASSES)}")
    return ENCODER_CLASSES[name](device if device is not None else default_device())


def feature_cells() -> list[tuple[str, str]]:
    """List every encoder/dataset pair whose features must be cached.

    Returns:
        Pairs of ``(encoder_name, dataset_name)``.
    """
    return [
        (encoder, dataset)
        for encoder, datasets in ENCODER_DATASETS.items()
        for dataset in datasets
    ]
