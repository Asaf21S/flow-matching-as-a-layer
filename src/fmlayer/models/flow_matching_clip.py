import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils import ModelWrapper
from torch import Tensor, nn

from src.fmlayer.models.prototypes import EPSILON
from src.fmlayer.utils.seeding import set_seed

# The brief asks for a small MLP: 2 hidden layers of width ~512, SiLU, scalar t concatenated
# to the input feature, output dimension equal to the feature dimension.
HIDDEN_DIM = 512
NUM_LAYERS = 2
# CLIP's own logit scale is ~100, i.e. a temperature of 0.01 on cosine similarities. Only
# used by the endpoint cross-entropy extension, not by the brief's objective.
TEMPERATURE = 0.01


class VelocityField(nn.Module):
    """The learned layer ``v(z, t)`` that transports a feature towards its class prototype.

    Takes no class label, which is what allows the same field to be applied to an unlabelled
    test feature at inference time.
    """

    def __init__(
        self, embed_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS
    ):
        """Build the field.

        Args:
            embed_dim: Width of the frozen features.
            hidden_dim: Width of each hidden layer.
            num_layers: Number of hidden layers.
        """
        super().__init__()
        widths = [embed_dim + 1] + [hidden_dim] * num_layers

        layers: list[nn.Module] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(fan_in, fan_out), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)

        # Zero-init the output so the untrained field is the identity map, which makes t=0
        # exactly the Stage 1 prototype baseline and lets each run assert that.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the velocity at a point and a time.

        Args:
            x: Features of shape ``(batch, embed_dim)``.
            t: Times of shape ``(batch,)``, or a scalar broadcast across the batch.

        Returns:
            Velocities of shape ``(batch, embed_dim)``.
        """
        t = t.reshape(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])
        return self.output_projection(self.trunk(torch.cat([x, t.reshape(-1, 1)], dim=-1)))


class ClipFlowWrapper(ModelWrapper):
    """Adapt :class:`VelocityField` to the solver's ``(x, t, **extras)`` calling convention."""

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """Forward the call to the wrapped velocity field, ignoring solver extras.

        Args:
            x: Features of shape ``(batch, embed_dim)``.
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
    num_layers: int = NUM_LAYERS,
) -> ClipFlowWrapper:
    """Create a seeded velocity field already wrapped for the ODE solver.

    Args:
        embed_dim: Width of the frozen features.
        seed: Seed controlling the weight initialisation.
        device: Device the field is created on.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.

    Returns:
        The wrapped field, ready for both training and integration.
    """
    set_seed(seed)
    return ClipFlowWrapper(VelocityField(embed_dim, hidden_dim, num_layers)).to(device)


def field_config(
    embed_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS
) -> dict:
    """Collect the architecture arguments needed to rebuild a field from a checkpoint."""
    return {"embed_dim": embed_dim, "hidden_dim": hidden_dim, "num_layers": num_layers}


def build_path() -> AffineProbPath:
    """Build the straight-line conditional optimal transport path used for standard training.

    This is the brief's interpolation ``z_t = (1 - t) z_i + t p_y`` with target velocity
    ``u_i = p_y - z_i``. In the package's convention ``x_0`` is the source and ``x_1`` the
    target, so the image feature sits at t=0 and the class prototype at t=1.

    Returns:
        The affine probability path with a conditional-OT schedule.
    """
    return AffineProbPath(scheduler=CondOTScheduler())


def sample_paths(
    path: AffineProbPath,
    x_0: Tensor,
    x_1: Tensor,
    generator: torch.Generator,
    target_noise: float = 0.0,
):
    """Draw one time per example and interpolate along the conditional path.

    ``target_noise`` is an extension, not part of the brief: it turns the t=1 end into a
    small cloud around the prototype instead of one of only ``num_classes`` atoms. Leave it
    at 0 for the brief's objective.

    Args:
        path: The probability path.
        x_0: Source features of shape ``(batch, embed_dim)``.
        x_1: Target prototypes of shape ``(batch, embed_dim)``.
        generator: CPU generator making the sampling reproducible.
        target_noise: Standard deviation of the Gaussian smoothing on ``x_1``.

    Returns:
        The path sample, exposing ``x_t``, ``dx_t`` and ``t``.
    """
    if target_noise > 0.0:
        noise = torch.randn(x_1.shape, generator=generator).to(x_1.device)
        x_1 = x_1 + target_noise * noise
    t = torch.rand(x_0.shape[0], generator=generator).to(x_0.device)
    return path.sample(t=t, x_0=x_0, x_1=x_1)


def predicted_endpoint(x_t: Tensor, t: Tensor, velocity: Tensor) -> Tensor:
    """Extrapolate the t=1 endpoint from a single velocity evaluation.

    On a straight path the velocity is constant, so ``x_1 = x_t + (1 - t) * v``. Used only by
    the endpoint cross-entropy extension.

    Args:
        x_t: States of shape ``(batch, embed_dim)``.
        t: Times of shape ``(batch,)``.
        velocity: Predicted velocities of shape ``(batch, embed_dim)``.

    Returns:
        Estimated endpoints of shape ``(batch, embed_dim)``.
    """
    return x_t + (1.0 - t).reshape(-1, 1) * velocity


def cosine_logits(
    features: Tensor, prototypes: Tensor, temperature: float = TEMPERATURE
) -> Tensor:
    """Score features against class prototypes with a temperature-scaled cosine.

    Args:
        features: Features of shape ``(batch, embed_dim)``.
        prototypes: Class prototypes of shape ``(num_classes, embed_dim)``.
        temperature: Softmax temperature applied to the cosine similarities.

    Returns:
        Logits of shape ``(batch, num_classes)``.
    """
    normalized = nn.functional.normalize(features, dim=-1, eps=EPSILON)
    targets = nn.functional.normalize(prototypes, dim=-1, eps=EPSILON)
    return (normalized @ targets.T) / temperature

