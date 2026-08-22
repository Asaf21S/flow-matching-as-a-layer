import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.fmlayer.data.fewshot import K_VALUES, SEEDS, load_train_subset
from src.fmlayer.data.specs import get_spec
from src.fmlayer.encoders.base import default_device
from src.fmlayer.encoders.registry import LINEAR_PROBE_CELLS
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_ode import rollout, transport
from src.fmlayer.models.probe_bank import ProbeBank
from src.fmlayer.models.targets import (
    CENTROIDS,
    DEFAULT_MARGIN_RATIO,
    MARGIN,
    NO_TARGET,
    PROBE_WEIGHTS,
    build_target_provider,
)
from src.fmlayer.models.velocity_field import (
    build_path,
    build_velocity_field,
    feature_statistics,
    sample_paths,
)
from src.fmlayer.train.probes import get_probe, get_probe_bank
from src.fmlayer.train.train_linear import to_tensors
from src.fmlayer.utils.results import default_results_root, record_run
from src.fmlayer.utils.seeding import set_seed

METHOD = "stage3"
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 500
EVAL_EVERY = 10
CURVES_DIRNAME = "curves_fm_stage3"
MODELS_DIRNAME = "models_stage3"
STEP_COUNTS = (4, 12)

STANDARD = "standard"
ROLLED_MSE = "rolled_mse"
ROLLED_CE = "rolled_ce"
HYBRID = "hybrid"
OBJECTIVES = (STANDARD, ROLLED_MSE, ROLLED_CE, HYBRID)

# Objectives whose loss never reads a geometric target; sweeping target types for
# them would train identical models.
TARGET_FREE_OBJECTIVES = (ROLLED_CE,)


@dataclass(frozen=True)
class FlowConfig:
    """One flow-matching variant: an objective, a target, and their knobs.

    Attributes:
        objective: One of :data:`OBJECTIVES`.
        target_type: Geometric target; ignored by target-free objectives.
        train_steps: Euler steps used for rolled-out training and at inference.
        noise_std: Gaussian source perturbation, as a fraction of the mean feature norm.
        mixup_alpha: Upper bound of the same-class mixup coefficient.
        cross_fit_folds: Number of held-out probes the flow trains against; 1 disables it.
        hybrid_lambda: Weight of the geometric term in the hybrid objective.
        margin_ratio: Margin distance as a fraction of the mean feature norm.
    """

    objective: str = STANDARD
    target_type: str = CENTROIDS
    train_steps: int = 12
    noise_std: float = 0.0
    mixup_alpha: float = 0.0
    cross_fit_folds: int = 1
    hybrid_lambda: float = 1.0
    margin_ratio: float = DEFAULT_MARGIN_RATIO

    @property
    def uses_targets(self) -> bool:
        """Whether the loss reads a geometric target."""
        return self.objective not in TARGET_FREE_OBJECTIVES

    @property
    def uses_rollout(self) -> bool:
        """Whether training backpropagates through the ODE solver."""
        return self.objective != STANDARD

    @property
    def name(self) -> str:
        """Short identifier used in tags, filenames and the ``method`` column."""
        parts = [self.objective]
        if self.uses_targets:
            parts.append(self.target_type)
        if self.uses_rollout:
            parts.append(f"T{self.train_steps}")
        if self.noise_std > 0:
            parts.append(f"n{self.noise_std:g}".replace(".", ""))
        if self.mixup_alpha > 0:
            parts.append(f"mx{self.mixup_alpha:g}".replace(".", ""))
        if self.cross_fit_folds > 1:
            parts.append(f"x{self.cross_fit_folds}")
        return "_".join(parts)

    @property
    def method(self) -> str:
        """Value written to the ``method`` column of ``runs.csv``."""
        return f"{METHOD}_{self.name}"

    def resolved_target(self) -> str:
        """Target type, normalised to ``"none"`` for target-free objectives."""
        return self.target_type if self.uses_targets else NO_TARGET

    def eval_steps(self, step_counts: tuple[int, ...] = STEP_COUNTS) -> tuple[int, ...]:
        """Step counts to evaluate at.

        Rolled-out training fixes T, so it is only evaluated at the T it trained with.
        Standard training is T-agnostic and is evaluated at every count.
        """
        return (self.train_steps,) if self.uses_rollout else tuple(step_counts)


def default_configs(step_counts: tuple[int, ...] = STEP_COUNTS) -> tuple[FlowConfig, ...]:
    """The eight configurations the full grid runs.

    Selected by screening every variant on all three cells at K=full, seed 0. Each entry
    is either a winner on at least one cell or the reference a winner has to be read
    against, and together they still span the objective x target comparison. See
    ``docs/PLAN_stage3.md`` §8.
    """
    shortest, longest = min(step_counts), max(step_counts)
    return (
        # Naive flow matching to class means: the reference everything else is read against.
        FlowConfig(STANDARD, CENTROIDS),
        # Transport toward the probe's own weight directions: strongest family on DINOv2.
        FlowConfig(STANDARD, PROBE_WEIGHTS, noise_std=0.15),
        FlowConfig(ROLLED_MSE, PROBE_WEIGHTS, shortest),
        FlowConfig(HYBRID, PROBE_WEIGHTS, longest),
        # Minimal margin corrections: the only family positive on every cell.
        FlowConfig(STANDARD, MARGIN, noise_std=0.15),
        FlowConfig(ROLLED_MSE, MARGIN, longest),
        FlowConfig(HYBRID, MARGIN, longest),
        # Pure classification loss, no geometric target.
        FlowConfig(ROLLED_CE, NO_TARGET, longest),
    )


def exploratory_configs(step_counts: tuple[int, ...] = STEP_COUNTS) -> tuple[FlowConfig, ...]:
    """Variants that were screened and dropped, kept so the pruning stays reproducible.

    ``rolled_mse`` toward class centroids was negative on all three cells (-0.006 to
    -0.074). The rest are each dominated by an entry in :func:`default_configs`, usually
    by their noised version or by their other step count.
    """
    shortest, longest = min(step_counts), max(step_counts)
    return (
        FlowConfig(STANDARD, PROBE_WEIGHTS),
        FlowConfig(STANDARD, MARGIN),
        FlowConfig(ROLLED_MSE, CENTROIDS, shortest),
        FlowConfig(ROLLED_MSE, CENTROIDS, longest),
        FlowConfig(ROLLED_MSE, PROBE_WEIGHTS, longest),
        FlowConfig(ROLLED_MSE, MARGIN, shortest),
        FlowConfig(ROLLED_CE, NO_TARGET, shortest),
        FlowConfig(HYBRID, MARGIN, shortest),
        FlowConfig(ROLLED_MSE, PROBE_WEIGHTS, longest, noise_std=0.15, mixup_alpha=0.3),
    )


def negative_control_configs(step_counts: tuple[int, ...] = STEP_COUNTS) -> tuple[FlowConfig, ...]:
    """Configurations kept as documented evidence rather than as candidates.

    Cross-fitting was meant to de-saturate the rolled classification loss. It does, but
    the flow then learns the fold probes' boundaries while being scored against the full
    probe, so the corrections are aimed at the wrong decision surface: ``rolled_ce`` fell
    from -0.056 to -0.080. Run explicitly to reproduce the finding.
    """
    longest = max(step_counts)
    return (
        FlowConfig(ROLLED_CE, NO_TARGET, longest, noise_std=0.15, cross_fit_folds=3),
        FlowConfig(HYBRID, MARGIN, longest, noise_std=0.15, cross_fit_folds=3),
    )


def all_configs(step_counts: tuple[int, ...] = STEP_COUNTS) -> tuple[FlowConfig, ...]:
    """Every configuration ever screened, including the dropped and control ones."""
    return (
        default_configs(step_counts)
        + exploratory_configs(step_counts)
        + negative_control_configs(step_counts)
    )


def same_class_permutation(labels: Tensor, generator: torch.Generator) -> Tensor:
    """Pair every sample with another sample of its own class.

    Args:
        labels: Training labels.
        generator: CPU generator driving the shuffle.

    Returns:
        Partner index per sample, shape ``(num_items,)``.
    """
    partners = torch.arange(len(labels), device=labels.device)
    for class_id in labels.unique():
        members = torch.nonzero(labels == class_id, as_tuple=True)[0]
        order = torch.randperm(len(members), generator=generator).to(labels.device)
        partners[members] = members[order]
    return partners


def perturb_sources(
    features: Tensor,
    partners: Tensor,
    config: FlowConfig,
    mean_norm: float,
    generator: torch.Generator,
) -> Tensor:
    """Apply same-class mixup and isotropic noise to the flow's starting points.

    Perturbing the source is what stops the rolled classification loss from only ever
    being evaluated on points the frozen probe has already memorised.

    Args:
        features: Clean source features of the batch.
        partners: Same-class partner features of the batch.
        config: Configuration supplying the perturbation strengths.
        mean_norm: Mean feature norm, which makes the noise scale-free.
        generator: CPU generator driving the draws.

    Returns:
        The perturbed sources.
    """
    source = features
    if config.mixup_alpha > 0:
        weights = torch.rand(len(source), 1, generator=generator).to(source.device)
        source = source + (config.mixup_alpha * weights) * (partners - source)
    if config.noise_std > 0:
        noise = torch.randn(source.shape, generator=generator).to(source.device)
        source = source + noise * (config.noise_std * mean_norm / math.sqrt(source.shape[1]))
    return source


def batch_loss(
    field: nn.Module,
    path,
    source: Tensor,
    labels: Tensor,
    provider,
    bank: ProbeBank,
    folds: Tensor | None,
    config: FlowConfig,
    generator: torch.Generator,
) -> Tensor:
    """Compute the training loss of one batch under the configured objective."""
    if config.objective == STANDARD:
        sample = sample_paths(path, source, provider(source, labels, folds), generator)
        return nn.functional.mse_loss(field(x=sample.x_t, t=sample.t), sample.dx_t)

    final, _ = rollout(field, source, config.train_steps)

    if config.objective == ROLLED_MSE:
        return nn.functional.mse_loss(final, provider(source, labels, folds))
    if config.objective == ROLLED_CE:
        return nn.functional.cross_entropy(bank.logits(final, folds), labels)
    if config.objective == HYBRID:
        classification = nn.functional.cross_entropy(bank.logits(final, folds), labels)
        geometric = nn.functional.mse_loss(final, provider(source, labels, folds))
        return classification + config.hybrid_lambda * geometric

    raise ValueError(f"Unknown objective {config.objective!r}. Available: {sorted(OBJECTIVES)}")


@torch.no_grad()
def probe_accuracy(probe: nn.Linear, features: Tensor, labels: Tensor) -> float:
    """Top-1 accuracy of the frozen probe on untransported features."""
    probe.eval()
    return (probe(features).argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate_transported(
    field: nn.Module, probe: nn.Linear, features: Tensor, labels: Tensor, steps: int
) -> float:
    """Top-1 accuracy of the frozen probe on features transported for ``steps`` steps."""
    probe.eval()
    return (probe(transport(field, features, steps)).argmax(dim=1) == labels).float().mean().item()


def train_flow(
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    num_classes: int,
    seed: int,
    device: torch.device,
    config: FlowConfig,
    probe: nn.Linear,
    bank: ProbeBank,
    folds: Tensor | None,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    min_learning_rate: float = MIN_LEARNING_RATE,
) -> tuple[dict[int, dict], list[dict], dict[int, int]]:
    """Train one velocity field, keeping the best checkpoint for each evaluated T.

    Args:
        train_features: Features of the K-shot training subset.
        train_labels: Labels of the K-shot training subset.
        val_features: Features of the validation split.
        val_labels: Labels of the validation split.
        num_classes: Number of classes.
        seed: Seed controlling initialisation, shuffling and augmentation.
        device: Device to train on.
        config: The flow configuration.
        probe: Frozen probe used for validation scoring.
        bank: Probe bank used inside the training loss.
        folds: Per-sample fold index, or ``None`` without cross-fitting.
        step_counts: Step counts a standard-objective field is evaluated at.
        max_epochs: Number of epochs.
        batch_size: Examples per optimisation step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        min_learning_rate: Floor of the cosine schedule.

    Returns:
        The best state dict per evaluated T, the per-epoch history, and the best epoch per T.
    """
    mean, scale = feature_statistics(train_features)
    field = build_velocity_field(
        embed_dim=train_features.shape[1],
        seed=seed,
        device=device,
        feature_mean=mean,
        feature_scale=scale,
    )
    path = build_path()

    provider = None
    if config.uses_targets:
        provider = build_target_provider(
            config.target_type, train_features, train_labels, bank, probe,
            num_classes, config.margin_ratio,
        )

    optimizer = torch.optim.AdamW(field.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=min_learning_rate
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    num_train = len(train_features)
    mean_norm = train_features.norm(p=2, dim=1).mean().item()
    augmenting = config.noise_std > 0 or config.mixup_alpha > 0
    evaluated = config.eval_steps(step_counts)

    history: list[dict] = []
    initial_state = {key: value.detach().clone() for key, value in field.state_dict().items()}
    best_states = {steps: initial_state for steps in evaluated}
    best_accuracy = {steps: -1.0 for steps in evaluated}
    best_epoch = {steps: 0 for steps in evaluated}

    for epoch in range(1, max_epochs + 1):
        field.train()
        order = torch.randperm(num_train, generator=generator).to(device)
        partners = same_class_permutation(train_labels, generator) if config.mixup_alpha > 0 else None
        loss_total = 0.0

        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)

            source = train_features[batch]
            if augmenting:
                partner = train_features[partners[batch]] if partners is not None else source
                source = perturb_sources(source, partner, config, mean_norm, generator)

            loss = batch_loss(
                field,
                path,
                source,
                train_labels[batch],
                provider,
                bank,
                folds[batch] if folds is not None else None,
                config,
                generator,
            )
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(batch)

        entry = {
            "epoch": epoch,
            "train_loss": loss_total / num_train,
            "lr": scheduler.get_last_lr()[0],
        }

        if epoch % EVAL_EVERY == 0 or epoch == max_epochs:
            by_steps = {
                steps: evaluate_transported(field, probe, val_features, val_labels, steps)
                for steps in evaluated
            }
            entry["val_by_steps"] = {str(key): value for key, value in by_steps.items()}
            entry["val_accuracy"] = max(by_steps.values())

            # Each T keeps its own best checkpoint, so the selection criterion and the
            # reported metric are the same quantity.
            for steps, accuracy in by_steps.items():
                if accuracy > best_accuracy[steps]:
                    best_accuracy[steps] = accuracy
                    best_epoch[steps] = epoch
                    best_states[steps] = {
                        key: value.detach().clone() for key, value in field.state_dict().items()
                    }

        history.append(entry)
        scheduler.step()

    return best_states, history, best_epoch


def stage3_tag(encoder: str, dataset: str, k: int | str, seed: int, config: FlowConfig) -> str:
    """Build the cache key of one Stage 3 run; ``config.name`` carries T and the knobs."""
    return f"{METHOD}_{config.name}_{encoder}_{dataset}_k{k}_seed{seed}"


def run_stage3(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    config: FlowConfig | None = None,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    record: bool = True,
    verbose: bool = True,
) -> dict:
    """Train and evaluate one flow-matching layer in front of the frozen Stage 1 probe.

    Args:
        encoder: Encoder key supplying the cached features.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        config: The flow configuration; defaults to standard FM towards class centroids.
        step_counts: Step counts a standard-objective field is evaluated at.
        feature_root: Feature cache directory.
        subset_root: Subset index directory.
        results_root: Results directory.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Number of epochs.
        record: Append the results to ``runs.csv``.
        verbose: Print a one-line summary.

    Returns:
        The baseline accuracy, the accuracy and delta per T, the history and the field.
    """
    config = config if config is not None else FlowConfig()
    device = device if device is not None else default_device()
    num_classes = get_spec(dataset).num_classes
    set_seed(seed)

    train_features, train_labels, indices = load_train_subset(
        encoder, dataset, k, seed, feature_root, subset_root
    )
    val_features, val_labels, _ = load_split(encoder, dataset, "val", feature_root)
    test_features, test_labels, _ = load_split(encoder, dataset, "test", feature_root)

    train_x, train_y = to_tensors(train_features, train_labels, device)
    val_x, val_y = to_tensors(val_features, val_labels, device)
    test_x, test_y = to_tensors(test_features, test_labels, device)

    probe = get_probe(
        encoder, dataset, k, seed, train_x, train_y, val_x, val_y, num_classes, device, results_root
    )
    bank, folds = get_probe_bank(
        encoder, dataset, k, seed, train_x, train_y, val_x, val_y,
        num_classes, device, config.cross_fit_folds, results_root,
    )
    baseline_accuracy = probe_accuracy(probe, test_x, test_y)

    root = Path(results_root) if results_root is not None else default_results_root()
    tag = stage3_tag(encoder, dataset, k, seed, config)
    curves_path = root / CURVES_DIRNAME / f"{tag}.json"
    model_path = root / MODELS_DIRNAME / f"{tag}.pt"
    evaluated = config.eval_steps(step_counts)

    loaded = curves_path.is_file() and model_path.is_file()
    best_states: dict[int, dict] = {}
    history: list[dict] = []
    best_epoch: dict[int, int] = {}
    accuracies: dict[int, float] = {}

    if loaded:
        payload = json.loads(curves_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        best_states = {int(key): value for key, value in checkpoint["best_states"].items()}
        history = payload["history"]
        best_epoch = {int(key): value for key, value in payload["best_epoch"].items()}
        accuracies = {int(key): value for key, value in payload["accuracy_by_steps"].items()}
        # A cache written for different step counts cannot answer this request.
        if not set(evaluated).issubset(best_states):
            loaded = False
            best_states, history, best_epoch, accuracies = {}, [], {}, {}

    if not loaded:
        best_states, history, best_epoch = train_flow(
            train_x, train_y, val_x, val_y, num_classes, seed, device, config,
            probe, bank, folds, step_counts=step_counts, max_epochs=max_epochs,
        )
        accuracies = {}

    mean, scale = feature_statistics(train_x)
    field = build_velocity_field(
        embed_dim=train_x.shape[1], seed=seed, device=device, feature_mean=mean, feature_scale=scale
    )

    if not accuracies:
        for steps in evaluated:
            field.load_state_dict(best_states[steps])
            accuracies[steps] = evaluate_transported(field, probe, test_x, test_y, steps)

    # The returned field is the checkpoint of the largest evaluated T, for the figures.
    field.load_state_dict(best_states[max(evaluated)])
    field.eval()
    deltas = {steps: value - baseline_accuracy for steps, value in accuracies.items()}

    if not loaded:
        curves_path.parent.mkdir(parents=True, exist_ok=True)
        curves_path.write_text(
            json.dumps(
                {
                    "encoder": encoder,
                    "dataset": dataset,
                    "k": k,
                    "seed": seed,
                    "config": asdict(config),
                    "config_name": config.name,
                    "num_train": len(indices),
                    "best_epoch": {str(key): value for key, value in best_epoch.items()},
                    "baseline_accuracy": baseline_accuracy,
                    "accuracy_by_steps": {str(key): value for key, value in accuracies.items()},
                    "delta_by_steps": {str(key): value for key, value in deltas.items()},
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"best_states": {str(key): value for key, value in best_states.items()}}, model_path
        )

    # Recorded on cache hits too, so runs.csv can always be rebuilt from the checkpoints.
    if record:
        for steps, value in accuracies.items():
            record_run(
                {
                    "method": config.method,
                    "dataset": dataset,
                    "encoder": encoder,
                    "k": k,
                    "seed": seed,
                    "steps": steps,
                    "split": "test",
                    "accuracy": value,
                    "num_items": len(test_labels),
                },
                results_root,
            )

    if verbose:
        scores = "  ".join(
            f"T={steps} {value:.4f} ({deltas[steps]:+.4f})"
            for steps, value in sorted(accuracies.items())
        )
        print(
            f"    [{config.name:<36}] {encoder:<14} {dataset:<9} k={str(k):<4} seed={seed}  "
            f"base {baseline_accuracy:.4f}  {scores}  {'(loaded)' if loaded else ''}"
        )

    return {
        "encoder": encoder,
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "config": config,
        "config_name": config.name,
        "objective": config.objective,
        "target_type": config.resolved_target(),
        "train_steps": config.train_steps,
        "fm_layer": field,
        "classifier": probe,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_by_steps": accuracies,
        "delta_by_steps": deltas,
        "best_epoch": best_epoch,
        "num_train": len(indices),
        "history": history,
    }


def run_all_stage3(
    cells: tuple[tuple[str, str], ...] | None = None,
    k_values: tuple[int | str, ...] = K_VALUES,
    seeds: tuple[int, ...] = SEEDS,
    configs: tuple[FlowConfig, ...] | None = None,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    record: bool = True,
    verbose: bool = False,
) -> dict:
    """Sweep flow configurations over the encoder/dataset cells, K values and seeds.

    Args:
        cells: ``(encoder, dataset)`` pairs; defaults to the Stage 1 probe cells.
        k_values: Training-set sizes to sweep.
        seeds: Seeds per training-set size.
        configs: Flow configurations; defaults to :func:`default_configs`.
        step_counts: Step counts a standard-objective field is evaluated at.
        feature_root: Feature cache directory.
        subset_root: Subset index directory.
        results_root: Results directory.
        device: Device to train on; defaults to CUDA when available.
        max_epochs: Number of epochs.
        record: Append the results to ``runs.csv``.
        verbose: Print one line per run instead of only the progress bar.

    Returns:
        One result per run, keyed by ``"config/encoder/dataset/k/seed"``.
    """
    cells = cells if cells is not None else LINEAR_PROBE_CELLS
    configs = configs if configs is not None else default_configs(step_counts)
    device = device if device is not None else default_device()
    print(
        f"Device: {device}  |  {len(configs)} configs x {len(cells)} cells "
        f"x {len(k_values)} K x {len(seeds)} seeds"
    )

    jobs = [
        (config, encoder, dataset, k, seed)
        for config in configs
        for encoder, dataset in cells
        for k in k_values
        for seed in seeds
    ]

    results = {}
    for config, encoder, dataset, k, seed in tqdm(jobs, desc="stage 3 grid"):
        results[f"{config.name}/{encoder}/{dataset}/{k}/{seed}"] = run_stage3(
            encoder,
            dataset,
            k,
            seed,
            config=config,
            step_counts=step_counts,
            feature_root=feature_root,
            subset_root=subset_root,
            results_root=results_root,
            device=device,
            max_epochs=max_epochs,
            record=record,
            verbose=verbose,
        )

    print(f"\n{len(results)} Stage 3 FM runs complete.")
    return results


def screen_configs(
    encoder: str = "dinov2_vits14",
    dataset: str = "dtd",
    k: int | str = "full",
    seed: int = 0,
    configs: tuple[FlowConfig, ...] | None = None,
    **kwargs,
) -> dict:
    """Run every configuration on a single cell, to decide what deserves the full grid.

    The default cell is the one where the flow has the best chance: the strongest encoder
    and the most training data. Screening on a weak encoder at low K is misleading,
    because the probe then has zero training error, which collapses the margin target
    into the identity and leaves the classification objectives with no gradient.

    Args:
        encoder: Encoder key.
        dataset: Dataset key.
        k: Shots per class, or ``"full"``.
        seed: Run seed.
        configs: Flow configurations; defaults to :func:`default_configs`.
        kwargs: Forwarded to :func:`run_all_stage3`.

    Returns:
        One result per configuration.
    """
    return run_all_stage3(
        cells=((encoder, dataset),),
        k_values=(k,),
        seeds=(seed,),
        configs=configs,
        verbose=True,
        **kwargs,
    )
