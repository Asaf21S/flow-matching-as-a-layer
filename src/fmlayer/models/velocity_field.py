import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from torch import Tensor, nn

from src.fmlayer.utils.seeding import set_seed

HIDDEN_DIM = 512
NUM_LAYERS = 2
EPSILON = 1e-6


class VelocityField(nn.Module):
    """Small MLP predicting ``v(z, t)``, with the scalar time concatenated to the input.

    Features are standardised on the way in and the predicted velocity is rescaled by
    the same per-dimension spread on the way out, so the network optimises in a
    well-conditioned space while the ODE still runs in the original feature space.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        feature_mean: Tensor | None = None,
        feature_scale: Tensor | None = None,
    ):
        super().__init__()
        mean = torch.zeros(embed_dim) if feature_mean is None else feature_mean.detach().clone()
        scale = torch.ones(embed_dim) if feature_scale is None else feature_scale.detach().clone()
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale.clamp_min(EPSILON))

        widths = [embed_dim + 1] + [hidden_dim] * num_layers
        layers: list[nn.Module] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(fan_in, fan_out), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)

        # Start as the identity flow: v = 0, so z_T = z_0 before any training.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        """Predict the velocity at ``x`` and time ``t``."""
        if isinstance(t, (float, int)):
            t = torch.tensor([float(t)], dtype=x.dtype, device=x.device)
        t = t.reshape(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])

        normalized = (x - self.feature_mean) / self.feature_scale
        hidden = self.trunk(torch.cat([normalized, t.reshape(-1, 1)], dim=-1))
        return self.output_projection(hidden) * self.feature_scale


def build_velocity_field(
    embed_dim: int,
    seed: int,
    device: torch.device,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    feature_mean: Tensor | None = None,
    feature_scale: Tensor | None = None,
) -> VelocityField:
    """Create a seeded velocity field on the given device.

    Args:
        embed_dim: Feature dimension the flow operates on.
        seed: Seed controlling the weight initialisation.
        device: Device the field is created on.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.
        feature_mean: Per-dimension mean used to standardise inputs.
        feature_scale: Per-dimension spread used to standardise inputs.

    Returns:
        The initialised velocity field.
    """
    set_seed(seed)
    field = VelocityField(embed_dim, hidden_dim, num_layers, feature_mean, feature_scale)
    return field.to(device)


def feature_statistics(features: Tensor) -> tuple[Tensor, Tensor]:
    """Per-dimension mean and standard deviation of a feature matrix.

    Args:
        features: Training features of shape ``(num_items, dim)``.

    Returns:
        The mean and the standard deviation, each of shape ``(dim,)``.
    """
    return features.mean(dim=0), features.std(dim=0).clamp_min(EPSILON)


def build_path() -> AffineProbPath:
    """Build the conditional-OT probability path used by the standard objective."""
    return AffineProbPath(scheduler=CondOTScheduler())


def sample_paths(path: AffineProbPath, x_0: Tensor, x_1: Tensor, generator: torch.Generator):
    """Sample ``t ~ U(0,1)`` and return the path state and target velocity at that time.

    Args:
        path: The probability path.
        x_0: Source features.
        x_1: Target features.
        generator: CPU generator driving the time draw.

    Returns:
        The sampled path state, carrying ``x_t``, ``dx_t`` and ``t``.
    """
    t = torch.rand(x_0.shape[0], generator=generator).to(x_0.device)
    return path.sample(t=t, x_0=x_0, x_1=x_1)
