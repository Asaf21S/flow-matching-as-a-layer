import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils import ModelWrapper
from torch import Tensor, nn

from src.fmlayer.utils.seeding import set_seed

HIDDEN_DIM = 512
NUM_LAYERS = 2

class VelocityField(nn.Module):
    def __init__(
        self, embed_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS
    ):
        super().__init__()
        widths = [embed_dim + 1] + [hidden_dim] * num_layers

        layers: list[nn.Module] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(fan_in, fan_out), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        if isinstance(t, (float, int)):
            t = torch.tensor([float(t)], dtype=x.dtype, device=x.device)
        t = t.reshape(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])
        return self.output_projection(self.trunk(torch.cat([x, t.reshape(-1, 1)], dim=-1)))

class ClipFlowWrapper(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x, t)

def build_velocity_field(
    embed_dim: int,
    seed: int,
    device: torch.device,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
) -> ClipFlowWrapper:
    set_seed(seed)
    return ClipFlowWrapper(VelocityField(embed_dim, hidden_dim, num_layers)).to(device)

def build_path() -> AffineProbPath:
    return AffineProbPath(scheduler=CondOTScheduler())

def sample_paths(
    path: AffineProbPath,
    x_0: Tensor,
    x_1: Tensor,
    generator: torch.Generator,
):
    t = torch.rand(x_0.shape[0], generator=generator).to(x_0.device)
    return path.sample(t=t, x_0=x_0, x_1=x_1)
