import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from src.fmlayer.encoders.base import Encoder


class ResNet18Encoder(Encoder):
    """ImageNet-1K ResNet-18, truncated to the 512-d penultimate representation."""

    NAME = "resnet18"
    EMBED_DIM = 512

    def __init__(self, device: torch.device):
        """Load the frozen ImageNet-1K checkpoint and its eval preprocessing.

        Args:
            device: Device the model is moved to.
        """
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        # Drop the 1000-way classifier; global average pooling then yields the 512-d vector.
        model.fc = nn.Identity()
        super().__init__(model, weights.transforms(), device)

    def forward_features(self, images: Tensor) -> Tensor:
        """Run the truncated network.

        Args:
            images: Batch of preprocessed images.

        Returns:
            Features of shape ``(batch, 512)``.
        """
        return self.model(images)
