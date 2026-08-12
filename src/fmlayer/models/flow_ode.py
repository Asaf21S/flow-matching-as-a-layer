import numpy as np
import torch
from flow_matching.solver import ODESolver
from torch import Tensor, nn

from src.fmlayer.train.evaluate import top1_accuracy

DEFAULT_METHOD = "euler"
DEFAULT_STEPS = 50

def rollout(model: nn.Module, x_init: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    states = [x_init]
    x = x_init
    for index in range(steps):
        t = torch.full((x.shape[0],), index / steps, device=x.device, dtype=x.dtype)
        x = x + (1.0 / steps) * model(x=x, t=t)
        states.append(x)
    return x, torch.stack(states)
