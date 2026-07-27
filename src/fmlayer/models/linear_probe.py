import torch
from torch import nn

from src.fmlayer.utils.seeding import set_seed


def build_linear_probe(
    embed_dim: int, num_classes: int, seed: int, device: torch.device
) -> nn.Linear:
    """Create the multiclass linear classifier ``s = Wz + b``.

    Only ``W`` and ``b`` are ever trained; the encoder stays frozen upstream.

    Args:
        embed_dim: Dimension of the frozen features.
        num_classes: Number of output classes.
        seed: Seed controlling the weight initialisation.
        device: Device the classifier is created on.

    Returns:
        The initialised linear layer.
    """
    set_seed(seed)
    return nn.Linear(embed_dim, num_classes).to(device)
