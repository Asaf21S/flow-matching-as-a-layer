import open_clip
import torch
from torch import Tensor

from src.fmlayer.encoders.base import Encoder

CLIP_ARCH = "RN50"
CLIP_PRETRAINED = "openai"


class ClipRN50Encoder(Encoder):
    """OpenAI CLIP RN50 with both towers frozen, for the zero-shot branch.

    Image and text embeddings share a 1024-d space, so class prototypes are built by
    encoding one prompt per class and comparing with cosine similarity.
    """

    NAME = "clip_rn50"
    EMBED_DIM = 1024

    def __init__(self, device: torch.device):
        """Load the frozen CLIP RN50 checkpoint, its preprocessing and its tokenizer.

        Args:
            device: Device the model is moved to.
        """
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_ARCH, pretrained=CLIP_PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
        super().__init__(model, preprocess, device)

    def forward_features(self, images: Tensor) -> Tensor:
        """Run the image tower.

        Args:
            images: Batch of preprocessed images.

        Returns:
            Image embeddings of shape ``(batch, 1024)``, not normalised.
        """
        return self.model.encode_image(images)

    @torch.inference_mode()
    def embed_texts(self, prompts: list[str]) -> Tensor:
        """Embed class prompts with the frozen text tower.

        Args:
            prompts: One prompt per class, in label-index order.

        Returns:
            Float32 text embeddings of shape ``(len(prompts), 1024)``, not normalised.
        """
        tokens = self.tokenizer(prompts).to(self.device)
        return self.model.encode_text(tokens).float()
