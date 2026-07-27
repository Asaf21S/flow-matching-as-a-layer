import numpy as np
import torch
from PIL import Image

from src.fmlayer.encoders.base import default_device
from src.fmlayer.encoders.clip_rn50 import ClipRN50Encoder
from src.fmlayer.encoders.registry import ENCODER_CLASSES, build_encoder

SMOKE_TEST_IMAGE_SIZE = (400, 300)
SMOKE_TEST_BATCH = 2
SMOKE_TEST_PROMPTS = ["a photo of a banded texture", "a photo of a bubbly texture"]


def random_image(seed: int = 0) -> Image.Image:
    """Create a random RGB image to push through a preprocessing pipeline.

    Args:
        seed: Seed making the image reproducible.

    Returns:
        A random PIL image.
    """
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (*SMOKE_TEST_IMAGE_SIZE[::-1], 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def smoke_test_encoder(name: str, device: torch.device | None = None) -> dict:
    """Load one encoder and check its output shape, dtype and frozen state.

    Args:
        name: Encoder key.
        device: Device to load onto; defaults to CUDA when available.

    Returns:
        A summary with the feature shape, the preprocessed input shape and the
        number of trainable parameters, which must be zero.
    """
    encoder = build_encoder(name, device)

    batch = torch.stack([encoder.transform(random_image(i)) for i in range(SMOKE_TEST_BATCH)])
    features = encoder.embed_images(batch)

    trainable = sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)
    assert trainable == 0, f"{name}: {trainable} parameters are still trainable"
    assert features.shape == (SMOKE_TEST_BATCH, encoder.embed_dim)
    assert features.dtype == torch.float32

    result = {
        "encoder": name,
        "embed_dim": encoder.embed_dim,
        "input_shape": tuple(batch.shape[1:]),
        "feature_shape": tuple(features.shape),
        "trainable_params": trainable,
        "device": str(encoder.device),
    }

    if isinstance(encoder, ClipRN50Encoder):
        text_features = encoder.embed_texts(SMOKE_TEST_PROMPTS)
        assert text_features.shape == (len(SMOKE_TEST_PROMPTS), encoder.embed_dim)
        result["text_feature_shape"] = tuple(text_features.shape)

    return result


def smoke_test_encoders(
    names: list[str] | None = None, device: torch.device | None = None
) -> dict:
    """Smoke-test several encoders and report their feature dimensions.

    Args:
        names: Encoder keys; defaults to every registered encoder.
        device: Device to load onto; defaults to CUDA when available.

    Returns:
        One summary per encoder, keyed by encoder name.
    """
    names = names if names is not None else list(ENCODER_CLASSES)
    device = device if device is not None else default_device()
    print(f"Device: {device}\n")

    results = {}
    for name in names:
        result = smoke_test_encoder(name, device)
        results[name] = result
        print(
            f"[ok] {name:<14} input {result['input_shape']} -> features {result['feature_shape']}"
        )
        if "text_feature_shape" in result:
            print(f"     {'':<14} text prompts -> {result['text_feature_shape']}")
    return results
