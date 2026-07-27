from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor, nn


class Encoder(ABC):
    """A frozen pretrained image encoder with its own eval preprocessing.

    Subclasses set ``NAME`` and ``EMBED_DIM`` and implement :meth:`forward_features`.
    The wrapped module is always in eval mode with gradients disabled, so the encoder
    can never be trained by accident.
    """

    NAME: str
    EMBED_DIM: int

    def __init__(self, model: nn.Module, transform: Callable, device: torch.device):
        """Freeze a model and bind it to its preprocessing.

        Args:
            model: The pretrained module to freeze.
            transform: Eval preprocessing shipped with the checkpoint; maps a PIL image to a tensor.
            device: Device the model is moved to.
        """
        self.model = model.to(device).eval().requires_grad_(False)
        self.transform = transform
        self.device = device

    @property
    def name(self) -> str:
        """Registry key of the encoder."""
        return self.NAME

    @property
    def embed_dim(self) -> int:
        """Dimension of the produced feature vectors."""
        return self.EMBED_DIM

    @abstractmethod
    def forward_features(self, images: Tensor) -> Tensor:
        """Map a preprocessed image batch to its representation.

        Args:
            images: Batch of preprocessed images, already on the right device.

        Returns:
            Features of shape ``(batch, embed_dim)``.
        """

    @torch.inference_mode()
    def embed_images(self, images: Tensor) -> Tensor:
        """Embed a batch of preprocessed images.

        Args:
            images: Batch of shape ``(batch, 3, H, W)``.

        Returns:
            Float32 features of shape ``(batch, embed_dim)``.
        """
        features = self.forward_features(images.to(self.device)).float()
        if features.shape[1] != self.EMBED_DIM:
            raise ValueError(
                f"{self.NAME}: expected {self.EMBED_DIM}-d features, got {features.shape[1]}"
            )
        return features


def default_device() -> torch.device:
    """Pick the compute device.

    Returns:
        CUDA when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
