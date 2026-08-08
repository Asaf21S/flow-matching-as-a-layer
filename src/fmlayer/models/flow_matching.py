import math
import torch
from torch import Tensor, nn

from src.fmlayer.utils.seeding import set_seed


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for continuous time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Args:

        t: Tensor of shape (batch_size,) or (batch_size, 1) with values in [0, 1].

        Returns:
            Embedding tensor of shape (batch_size, dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb_scale)
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t)], dim=-1)
        return emb


class VectorFieldMLP(nn.Module):
    """MLP vector field v_theta(z_t, t) for Conditional Flow Matching."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        blocks = []
        for _ in range(num_layers):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # Zero-initialize the final linear layer so the initial velocity field is exactly zero.
        # This ensures the Flow Matching ODE starts as an identity mapping (z_1 = z_0),
        # allowing the linear probe to learn from stable, undistorted features initially.
        nn.init.zeros_(self.output_proj[3].weight)
        nn.init.zeros_(self.output_proj[3].bias)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """Args:

        z: Feature tensor of shape (batch_size, embed_dim).
        t: Continuous time tensor of shape (batch_size,) or scalar.

        Returns:
            Velocity field tensor v_theta(z, t) of shape (batch_size, embed_dim).
        """
        if isinstance(t, (float, int)):
            t = torch.full((z.shape[0],), float(t), device=z.device, dtype=z.dtype)
        elif t.dim() == 0:
            t = t.expand(z.shape[0])

        t_emb = self.time_embed(t)
        h = self.input_proj(z) + t_emb

        for block in self.blocks:
            h = h + block(h)

        return self.output_proj(h)


class FlowMatchingLayer(nn.Module):
    """Continuous/discrete Flow Matching Layer performing t: 0 -> 1 ODE integration."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 128,
        num_layers: int = 3,
        num_steps: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_steps = num_steps
        self.vector_field = VectorFieldMLP(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
        )

    def forward(self, z0: Tensor, num_steps: int | None = None) -> Tensor:
        """Integrate ODE from t=0 to t=1 using Euler method.

        Args:
            z0: Initial state of shape (batch_size, embed_dim).
            num_steps: Number of integration steps (defaults to self.num_steps).

        Returns:
            Integrated output z1 of shape (batch_size, embed_dim).
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps
        z = z0

        for i in range(steps):
            t = i * dt
            v = self.vector_field(z, t)
            z = z + v * dt

        return z


def compute_cfm_loss(
    vector_field: VectorFieldMLP,
    x1: Tensor,
    x0: Tensor | None = None,
) -> Tensor:
    """Compute the Conditional Flow Matching (CFM) regression loss.

    Args:
        vector_field: VectorFieldMLP model.
        x1: Target feature tensor (batch_size, embed_dim).
        x0: Source feature/noise tensor (batch_size, embed_dim). If None, standard Gaussian noise is sampled.

    Returns:
        Scalar MSE loss between predicted velocity and target velocity (x1 - x0).
    """
    batch_size, embed_dim = x1.shape
    device = x1.device

    if x0 is None:
        x0 = torch.randn_like(x1)

    t = torch.rand(batch_size, device=device, dtype=x1.dtype)
    t_expand = t.unsqueeze(-1)

    # Linear interpolation trajectory z_t = t * x1 + (1 - t) * x0
    zt = t_expand * x1 + (1.0 - t_expand) * x0
    ut = x1 - x0  # Target velocity

    vt = vector_field(zt, t)
    return nn.functional.mse_loss(vt, ut)


def build_flow_matching_layer(
    embed_dim: int,
    seed: int,
    device: torch.device,
    hidden_dim: int = 128,
    num_steps: int = 10,
) -> FlowMatchingLayer:
    """Instantiate and seed a FlowMatchingLayer."""
    set_seed(seed)
    return FlowMatchingLayer(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
    ).to(device)
