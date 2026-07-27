import torch
from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from src.fmlayer.encoders.base import Encoder

DINOV2_HUB_REPO = "facebookresearch/dinov2"
DINOV2_HUB_MODEL = "dinov2_vits14"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SIZE = 256
CROP_SIZE = 224


def dinov2_transform() -> Compose:
    """Build the official DINOv2 classification eval transform.

    The 224 crop is divisible by the ViT-S/14 patch size, giving a 16x16 patch grid.

    Returns:
        Preprocessing that maps a PIL image to a normalised tensor.
    """
    return Compose(
        [
            Resize(RESIZE_SIZE, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(CROP_SIZE),
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class Dinov2Encoder(Encoder):
    """DINOv2 ViT-S/14, using the final class-token representation."""

    NAME = "dinov2_vits14"
    EMBED_DIM = 384

    def __init__(self, device: torch.device):
        """Load the frozen DINOv2 ViT-S/14 checkpoint from torch.hub.

        Args:
            device: Device the model is moved to.
        """
        model = torch.hub.load(DINOV2_HUB_REPO, DINOV2_HUB_MODEL, trust_repo=True)
        super().__init__(model, dinov2_transform(), device)

    def forward_features(self, images: Tensor) -> Tensor:
        """Run the backbone, whose forward already returns the class token.

        Args:
            images: Batch of preprocessed images.

        Returns:
            Features of shape ``(batch, 384)``.
        """
        return self.model(images)
