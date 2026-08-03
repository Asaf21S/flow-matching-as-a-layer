import math

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils import ModelWrapper
from torch import Tensor, nn

from src.fmlayer.utils.seeding import set_seed

TIME_EMBED_DIM = 128
HIDDEN_DIM = 2048
NUM_BLOCKS = 3
DROPOUT = 0.0


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Encode the integration time as a fixed sinusoidal feature vector.

    Args:
        t: Times of shape ``(batch,)`` in ``[0, 1]``.
        dim: Width of the embedding; must be even.

    Returns:
        Embeddings of shape ``(batch, dim)``.
    """
    half = dim // 2
    exponents = torch.arange(half, device=t.device, dtype=torch.float32) / half
    frequencies = torch.exp(-math.log(10000.0) * exponents)
    angles = t.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """Pre-norm MLP block whose normalisation is modulated by the time embedding."""

    def __init__(self, width: int, time_dim: int, dropout: float):
        """Build one residual block.

        Args:
            width: Hidden width of the trunk.
            time_dim: Width of the incoming time embedding.
            dropout: Dropout probability applied inside the block.
        """
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.time_projection = nn.Linear(time_dim, 2 * width)
        self.linear_in = nn.Linear(width, width)
        self.linear_out = nn.Linear(width, width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, time_embedding: Tensor) -> Tensor:
        """Apply one modulated residual update.

        Args:
            hidden: Trunk activations of shape ``(batch, width)``.
            time_embedding: Time features of shape ``(batch, time_dim)``.

        Returns:
            Updated activations of the same shape as ``hidden``.
        """
        scale, shift = self.time_projection(time_embedding).chunk(2, dim=-1)
        residual = self.norm(hidden) * (1.0 + scale) + shift
        residual = self.activation(self.linear_in(residual))
        return hidden + self.linear_out(self.dropout(residual))


class VelocityField(nn.Module):
    """The learned layer ``v(x, t)`` that moves a CLIP embedding towards its class text embedding.

    Takes no class label, which is what allows the same field to be applied to an
    unlabelled test embedding at inference time.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_blocks: int = NUM_BLOCKS,
        time_dim: int = TIME_EMBED_DIM,
        dropout: float = DROPOUT,
    ):
        """Build the field.

        Args:
            embed_dim: Width of the frozen CLIP embeddings.
            hidden_dim: Width of the residual trunk.
            num_blocks: Number of residual blocks.
            time_dim: Width of the sinusoidal time embedding.
            dropout: Dropout probability inside each block.
        """
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_blocks)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)

        # Zero-init the output so the untrained field is the identity map and t=0 and t=1
        # both start out at exactly the zero-shot baseline.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the velocity at a point and a time.

        Args:
            x: Embeddings of shape ``(batch, embed_dim)``.
            t: Times of shape ``(batch,)``, or a scalar broadcast across the batch.

        Returns:
            Velocities of shape ``(batch, embed_dim)``.
        """
        t = t.reshape(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])

        time_embedding = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        return self.output_projection(self.norm_out(hidden))


class ClipFlowWrapper(ModelWrapper):
    """Adapt :class:`VelocityField` to the solver's ``(x, t, **extras)`` calling convention."""

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """Forward the call to the wrapped velocity field, ignoring solver extras.

        Args:
            x: Embeddings of shape ``(batch, embed_dim)``.
            t: Times of shape ``(batch,)``, or a scalar.

        Returns:
            Velocities of shape ``(batch, embed_dim)``.
        """
        return self.model(x, t)


def build_velocity_field(
    embed_dim: int,
    seed: int,
    device: torch.device,
    hidden_dim: int = HIDDEN_DIM,
    num_blocks: int = NUM_BLOCKS,
    dropout: float = DROPOUT,
) -> ClipFlowWrapper:
    """Create a seeded velocity field already wrapped for the ODE solver.

    Args:
        embed_dim: Width of the frozen CLIP embeddings.
        seed: Seed controlling the weight initialisation.
        device: Device the field is created on.
        hidden_dim: Width of the residual trunk.
        num_blocks: Number of residual blocks.
        dropout: Dropout probability inside each block.

    Returns:
        The wrapped field, ready for both training and integration.
    """
    set_seed(seed)
    field = VelocityField(embed_dim, hidden_dim, num_blocks, TIME_EMBED_DIM, dropout)
    return ClipFlowWrapper(field).to(device)


def field_config(
    embed_dim: int,
    hidden_dim: int = HIDDEN_DIM,
    num_blocks: int = NUM_BLOCKS,
    dropout: float = DROPOUT,
) -> dict:
    """Collect the architecture arguments needed to rebuild a field from a checkpoint."""
    return {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "dropout": dropout,
    }


def build_path() -> AffineProbPath:
    """Build the straight-line conditional optimal transport path used for training.

    In the package's convention ``x_0`` is the source and ``x_1`` the target, so the image
    embedding sits at t=0 and the class text embedding at t=1.

    Returns:
        The affine probability path with a conditional-OT schedule.
    """
    return AffineProbPath(scheduler=CondOTScheduler())


def flow_matching_loss(
    model: ClipFlowWrapper,
    path: AffineProbPath,
    x_0: Tensor,
    x_1: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Regress the velocity field onto the conditional path velocity for one batch.

    Args:
        model: The wrapped velocity field.
        path: The probability path supplying ``x_t`` and ``dx_t``.
        x_0: Source embeddings of shape ``(batch, embed_dim)``.
        x_1: Target embeddings of shape ``(batch, embed_dim)``.
        generator: CPU generator making the time sampling reproducible.

    Returns:
        The mean squared error between the predicted and the conditional velocity.
    """
    t = torch.rand(x_0.shape[0], generator=generator).to(x_0.device)
    sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    return nn.functional.mse_loss(model(x=sample.x_t, t=sample.t), sample.dx_t)
