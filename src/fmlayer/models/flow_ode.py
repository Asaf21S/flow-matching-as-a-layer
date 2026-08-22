import torch
from torch import Tensor, nn


def rollout(model: nn.Module, x_init: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    """Integrate the flow with ``steps`` explicit Euler steps from t=0 to t=1.

    Args:
        model: The velocity field.
        x_init: Starting features of shape ``(batch, dim)``.
        steps: Number of Euler steps.

    Returns:
        The final state and every intermediate state, shape ``(steps + 1, batch, dim)``.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    states = [x_init]
    x = x_init
    for index in range(steps):
        t = torch.full((x.shape[0],), index / steps, device=x.device, dtype=x.dtype)
        x = x + (1.0 / steps) * model(x=x, t=t)
        states.append(x)
    return x, torch.stack(states)


@torch.no_grad()
def transport(model: nn.Module, features: Tensor, steps: int) -> Tensor:
    """Run the flow without gradients, returning only the transported features.

    Args:
        model: The velocity field.
        features: Features to transport.
        steps: Number of Euler steps; ``0`` leaves the features untouched.

    Returns:
        The transported features.
    """
    if steps < 1:
        return features
    model.eval()
    final, _ = rollout(model, features, steps)
    return final
