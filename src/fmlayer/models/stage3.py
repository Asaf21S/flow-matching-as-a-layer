import torch
from torch import Tensor, nn

from src.fmlayer.models.flow_matching import FlowMatchingLayer, build_flow_matching_layer
from src.fmlayer.models.linear_probe import build_linear_probe
from src.fmlayer.utils.seeding import set_seed


class Stage3(nn.Module):
    """Composite model inserting Flow Matching layer BEFORE the linear probe classifier.

    z (input feature) ---> FlowMatchingLayer (t=0 -> 1 integration) ---> z_tilde ---> nn.Linear ---> logits
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_steps: int = 10,
        fm_layer: FlowMatchingLayer | None = None,
    ):
        super().__init__()
        if fm_layer is not None:
            self.fm_layer = fm_layer
        else:
            self.fm_layer = FlowMatchingLayer(
                embed_dim=embed_dim, hidden_dim=hidden_dim, num_steps=num_steps
            )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, z0: Tensor, num_steps: int | None = None) -> Tensor:
        """Args:

        z0: Raw feature tensor of shape (batch_size, embed_dim).
        num_steps: Number of Euler integration steps for flow matching layer.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        z_transformed = self.fm_layer(z0, num_steps=num_steps)
        return self.classifier(z_transformed)


def build_stage3(
    embed_dim: int,
    num_classes: int,
    seed: int,
    device: torch.device,
    hidden_dim: int = 128,
    num_steps: int = 10,
) -> Stage3:
    """Instantiate and seed a Stage3 composite model."""
    set_seed(seed)
    fm_layer = build_flow_matching_layer(
        embed_dim=embed_dim,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
    )
    linear_probe = build_linear_probe(
        embed_dim=embed_dim, num_classes=num_classes, seed=seed, device=device
    )

    model = Stage3(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        fm_layer=fm_layer,
    ).to(device)

    # Copy initial weights from initialized linear probe for consistency
    model.classifier.weight.data.copy_(linear_probe.weight.data)
    model.classifier.bias.data.copy_(linear_probe.bias.data)

    return model
