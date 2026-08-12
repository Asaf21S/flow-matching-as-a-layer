import numpy as np
import torch
from flow_matching.solver import ODESolver
from torch import Tensor, nn

from src.fmlayer.models.prototypes import EPSILON
from src.fmlayer.train.evaluate import top1_accuracy

DEFAULT_METHOD = "euler"
DEFAULT_STEPS = 50
MANUAL_EULER = "manual_euler"
FIXED_STEP_METHODS = ("euler", "midpoint", "heun2", "heun3", "rk4")
CHUNK_SIZE = 512


def rollout(model: nn.Module, x_init: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    """Apply the brief's T-step Euler rule, keeping gradients.

    Implements ``z[k+1] = z[k] + (1/T) * v(z[k], k/T)`` for ``k = 0 .. T-1``, which is both
    the inference rule and, when differentiated, the rolled-out training objective.

    Args:
        model: Velocity field callable as ``model(x=..., t=...)``.
        x_init: Initial features of shape ``(num_items, dim)``.
        steps: Number of Euler steps ``T``.

    Returns:
        The final state of shape ``(num_items, dim)`` and every state including both
        endpoints, of shape ``(steps + 1, num_items, dim)``.
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
def rollout_trajectory(model: nn.Module, features: Tensor, steps: int) -> Tensor:
    """Run :func:`rollout` for inspection, without building a graph.

    Args:
        model: Velocity field wrapped for the solver.
        features: Features of shape ``(num_items, dim)``.
        steps: Number of Euler steps ``T``.

    Returns:
        Every state, of shape ``(steps + 1, num_items, dim)``.
    """
    return rollout(model, features, steps)[1]


@torch.no_grad()
def rollout_predictions(
    model: nn.Module,
    features: Tensor,
    prototypes: Tensor,
    steps: int,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Classify the endpoint of the T-step rollout against fixed prototypes.

    This is the brief's scored metric: transport the feature with ``T`` Euler steps, then
    apply the Stage 1 cosine rule to the final state.

    Args:
        model: Velocity field wrapped for the solver.
        features: Features of shape ``(num_items, dim)``.
        prototypes: Class prototypes of shape ``(num_classes, dim)``.
        steps: Number of Euler steps ``T``.
        chunk_size: Items transported at once.

    Returns:
        Nearest-prototype labels of shape ``(num_items,)``.
    """
    normalized_prototypes = nn.functional.normalize(prototypes, dim=-1, eps=EPSILON)
    predictions = []

    for start in range(0, len(features), chunk_size):
        final, _ = rollout(model, features[start : start + chunk_size], steps)
        similarities = (
            nn.functional.normalize(final, dim=-1, eps=EPSILON) @ normalized_prototypes.T
        )
        predictions.append(similarities.argmax(dim=-1).cpu().numpy())

    return np.concatenate(predictions)


def make_time_grid(
    num_points: int, reverse: bool = False, device: torch.device | None = None
) -> Tensor:
    """Build a uniform grid of times the trajectory is reported at.

    Args:
        num_points: Number of grid points, endpoints included.
        reverse: Run from t=1 down to t=0 instead of t=0 up to t=1.
        device: Device the grid is placed on; defaults to CPU.

    Returns:
        Times of shape ``(num_points,)``.
    """
    grid = (
        torch.linspace(1.0, 0.0, num_points)
        if reverse
        else torch.linspace(0.0, 1.0, num_points)
    )
    return grid if device is None else grid.to(device)


@torch.no_grad()
def euler_trajectory(
    model: nn.Module, x_init: Tensor, time_grid: Tensor, steps: int = DEFAULT_STEPS
) -> Tensor:
    """Integrate along an arbitrary time grid with an explicit Euler loop.

    Direction-agnostic, so it also serves the reverse flow. Used for the diagnostic
    accuracy-versus-t sweep; the brief's scored metric goes through :func:`rollout` instead.

    Args:
        model: Velocity field callable as ``model(x=..., t=...)``.
        x_init: Initial states of shape ``(num_items, dim)``.
        time_grid: Times to report at; may increase or decrease.
        steps: Sub-steps per unit of time, so the step size is ``1 / steps``.

    Returns:
        Trajectory of shape ``(len(time_grid), num_items, dim)``.
    """
    # The field builds its time input on the device of ``t``, so the grid must follow x.
    time_grid = time_grid.to(x_init.device)
    states = [x_init]
    x = x_init
    for start, end in zip(time_grid[:-1], time_grid[1:]):
        span = float(end - start)
        substeps = max(1, round(abs(span) * steps))
        size = span / substeps
        t = start
        for _ in range(substeps):
            x = x + size * model(x=x, t=t)
            t = t + size
        states.append(x)
    return torch.stack(states)


@torch.no_grad()
def integrate(
    model: nn.Module,
    x_init: Tensor,
    time_grid: Tensor,
    method: str = DEFAULT_METHOD,
    steps: int = DEFAULT_STEPS,
) -> Tensor:
    """Integrate the velocity field and return every intermediate state.

    Args:
        model: Velocity field wrapped for the solver.
        x_init: Initial states of shape ``(num_items, dim)``.
        time_grid: Times to report at; a decreasing grid runs the flow in reverse.
        method: ``"euler"``, ``"midpoint"``, ``"dopri5"``, or ``"manual_euler"``.
        steps: Sub-steps per unit of time for the fixed-step methods.

    Returns:
        Trajectory of shape ``(len(time_grid), num_items, dim)``.
    """
    if method == MANUAL_EULER:
        return euler_trajectory(model, x_init, time_grid, steps)

    solver = ODESolver(velocity_model=model)
    return solver.sample(
        x_init=x_init,
        step_size=(1.0 / steps) if method in FIXED_STEP_METHODS else None,
        method=method,
        time_grid=time_grid.to(x_init.device),
        return_intermediates=True,
    )


@torch.no_grad()
def trajectory_predictions(
    model: nn.Module,
    features: Tensor,
    prototypes: Tensor,
    time_grid: Tensor,
    method: str = DEFAULT_METHOD,
    steps: int = DEFAULT_STEPS,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Classify against fixed prototypes at every point along the trajectory.

    Integration is chunked over items so the full trajectory is never materialised for a
    whole split at once.

    Args:
        model: Velocity field wrapped for the solver.
        features: Features of shape ``(num_items, dim)``.
        prototypes: Class prototypes of shape ``(num_classes, dim)``.
        time_grid: Times to classify at.
        method: Solver method.
        steps: Sub-steps per unit of time for the fixed-step methods.
        chunk_size: Items integrated at once.

    Returns:
        Nearest-prototype labels of shape ``(len(time_grid), num_items)``.
    """
    normalized_prototypes = nn.functional.normalize(prototypes, dim=-1, eps=EPSILON)
    predictions = []

    for start in range(0, len(features), chunk_size):
        trajectory = integrate(
            model, features[start : start + chunk_size], time_grid, method, steps
        )
        similarities = (
            nn.functional.normalize(trajectory, dim=-1, eps=EPSILON)
            @ normalized_prototypes.T
        )
        predictions.append(similarities.argmax(dim=-1).cpu().numpy())

    return np.concatenate(predictions, axis=1)


def trajectory_accuracies(predictions: np.ndarray, labels: np.ndarray) -> list[float]:
    """Score top-1 accuracy at every trajectory time.

    Args:
        predictions: Output of :func:`trajectory_predictions`.
        labels: Ground-truth labels of shape ``(num_items,)``.

    Returns:
        One accuracy per time, in grid order.
    """
    return [top1_accuracy(step, labels) for step in predictions]

