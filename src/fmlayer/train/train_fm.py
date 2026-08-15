import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from src.fmlayer.data.fewshot import K_VALUES, SEEDS, load_train_subset
from src.fmlayer.data.specs import get_spec
from src.fmlayer.encoders.base import default_device
from src.fmlayer.encoders.registry import LINEAR_PROBE_CELLS
from src.fmlayer.features.cache import load_split
from src.fmlayer.models.flow_matching_mlp import build_path, sample_paths
from src.fmlayer.models.flow_ode import rollout
from src.fmlayer.train.train_linear import to_tensors, train_probe
from src.fmlayer.utils.results import default_results_root, record_run
from src.fmlayer.utils.seeding import set_seed

METHOD = "stage3"
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 1000
EVAL_EVERY = 10
CURVES_DIRNAME = "curves_fm_stage3"
STEP_COUNTS = (4, 12)

STANDARD = "standard"
ROLLED_MSE = "rolled_mse"
ROLLED_CE = "rolled_ce"
OBJECTIVES = (STANDARD, ROLLED_MSE, ROLLED_CE)

CENTROIDS = "centroids"
PROBE_WEIGHTS = "probe_weights"
ORTHOGONAL = "orthogonal"
TARGET_TYPES = (CENTROIDS, PROBE_WEIGHTS, ORTHOGONAL)


def compute_centroids(features: Tensor, labels: Tensor, num_classes: int, device: torch.device) -> Tensor:
    embed_dim = features.shape[1]
    centroids = torch.zeros(num_classes, embed_dim, device=device)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = features[mask].mean(dim=0)
        else:
            centroids[c] = features.mean(dim=0)
    return centroids


def compute_targets(
    features: Tensor, labels: Tensor, linear_probe: nn.Module, num_classes: int, target_type: str, device: torch.device
) -> Tensor:
    if target_type == CENTROIDS:
        return compute_centroids(features, labels, num_classes, device)
    elif target_type == PROBE_WEIGHTS:
        weights = linear_probe.weight.detach()
        avg_norm = features.norm(p=2, dim=1).mean()
        weights_normalized = torch.nn.functional.normalize(weights, p=2, dim=1)
        return weights_normalized * avg_norm
    elif target_type == ORTHOGONAL:
        embed_dim = features.shape[1]
        avg_norm = features.norm(p=2, dim=1).mean()
        Q, _ = torch.linalg.qr(torch.randn(embed_dim, embed_dim, device=device))
        targets = Q[:num_classes]
        return targets * avg_norm
    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def batch_loss(
    model: nn.Module,
    path,
    source: Tensor,
    target: Tensor,
    labels: Tensor,
    linear_probe: nn.Module,
    criterion: nn.Module,
    generator: torch.Generator,
    objective: str,
    steps: int,
) -> Tensor:
    if objective == STANDARD:
        sample = sample_paths(path, source, target, generator)
        velocity = model(x=sample.x_t, t=sample.t)
        return nn.functional.mse_loss(velocity, sample.dx_t)
    elif objective == ROLLED_MSE:
        z_T, _ = rollout(model, source, steps)
        return nn.functional.mse_loss(z_T, target)
    elif objective == ROLLED_CE:
        z_T, _ = rollout(model, source, steps)
        logits = linear_probe(z_T)
        return criterion(logits, labels)
    else:
        raise ValueError(f"Unknown objective: {objective}")


@torch.no_grad()
def evaluate_fm_probe(
    fm_layer: nn.Module, classifier: nn.Module, features: Tensor, labels: Tensor, criterion: nn.Module, steps: int
) -> tuple[float, float]:
    fm_layer.eval()
    classifier.eval()
    if steps > 0:
        z, _ = rollout(fm_layer, features, steps)
    else:
        z = features
    logits = classifier(z)
    loss = criterion(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def train_fm_probe(
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    num_classes: int,
    seed: int,
    device: torch.device,
    objective: str = STANDARD,
    target_type: str = CENTROIDS,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    train_steps: int = 12,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    min_learning_rate: float = MIN_LEARNING_RATE,
) -> tuple[nn.Module, nn.Module, list[dict], int]:
    embed_dim = train_features.shape[1]
    
    linear_probe, _, _ = train_probe(
        train_features, train_labels, val_features, val_labels, num_classes, seed, device, max_epochs=200
    )
    for param in linear_probe.parameters():
        param.requires_grad = False
    linear_probe.eval()

    class_targets = compute_targets(train_features, train_labels, linear_probe, num_classes, target_type, device)
    
    from src.fmlayer.models.flow_matching_mlp import build_velocity_field, build_path
    fm_layer = build_velocity_field(embed_dim=embed_dim, seed=seed, device=device)
    path = build_path()
    
    optimizer = torch.optim.AdamW(fm_layer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    num_train = len(train_features)
    history = []
    criterion = nn.CrossEntropyLoss()
    
    best_state = {key: value.detach().clone() for key, value in fm_layer.state_dict().items()}
    best_accuracy = -1.0
    best_epoch = 0
    
    evaluated = (train_steps,) if objective != STANDARD else tuple(step_counts)

    for epoch in range(1, max_epochs + 1):
        fm_layer.train()
        order = torch.randperm(num_train, generator=generator).to(device)
        loss_total = 0.0
        
        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            
            source = train_features[batch]
            target = class_targets[train_labels[batch]]
            labels = train_labels[batch]
            
            loss = batch_loss(
                fm_layer, path, source, target, labels, linear_probe, criterion, generator, objective, train_steps
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
            accuracies = {}
            for steps in evaluated:
                val_loss, val_accuracy = evaluate_fm_probe(fm_layer, linear_probe, val_features, val_labels, criterion, steps)
                accuracies[steps] = val_accuracy
                
            mean_accuracy = float(np.mean(list(accuracies.values())))
            entry["val_accuracy"] = mean_accuracy
            entry["val_by_steps"] = {str(k): v for k, v in accuracies.items()}
            
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_epoch = epoch
                best_state = {key: value.detach().clone() for key, value in fm_layer.state_dict().items()}
                
        history.append(entry)
        scheduler.step()
        
    fm_layer.load_state_dict(best_state)
    return fm_layer, linear_probe, history, best_epoch


def run_stage3(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
    objective: str = STANDARD,
    target_type: str = CENTROIDS,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    train_steps: int = 12,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    record: bool = True,
    verbose: bool = True,
) -> dict:
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

    tag = f"{METHOD}_{objective}_{target_type}_{encoder}_{dataset}_k{k}_seed{seed}"
    root = Path(results_root) if results_root is not None else default_results_root()
    path = root / CURVES_DIRNAME / f"{tag}.json"
    model_path = root / "models_stage3" / f"{tag}.pt"

    if path.is_file() and model_path.is_file():
        # Load from disk
        payload = json.loads(path.read_text(encoding="utf-8"))
        
        linear_probe, _, _ = train_probe(
            train_x, train_y, val_x, val_y, num_classes, seed, device, max_epochs=200
        )
        embed_dim = train_features.shape[1]
        from src.fmlayer.models.flow_matching_mlp import build_velocity_field
        fm_layer = build_velocity_field(embed_dim=embed_dim, seed=seed, device=device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        fm_layer.load_state_dict(checkpoint["fm_layer"])
        linear_probe.load_state_dict(checkpoint["linear_probe"])
        fm_layer.eval()
        linear_probe.eval()
        
        history = payload["history"]
        baseline_accuracy = payload["baseline_accuracy"]
        accuracies = {int(key): value for key, value in payload["accuracy_by_steps"].items()}
        deltas = {int(key): value for key, value in payload["delta_by_steps"].items()}
        best_epoch = payload["best_epoch"]
        loaded = True
    else:
        # Train
        loaded = False
        fm_layer, linear_probe, history, best_epoch = train_fm_probe(
            train_x, train_y, val_x, val_y, num_classes, seed, device, 
            objective=objective, target_type=target_type, step_counts=step_counts, train_steps=train_steps, max_epochs=max_epochs
        )

        criterion = nn.CrossEntropyLoss()
        
        baseline_loss, baseline_accuracy = evaluate_fm_probe(fm_layer, linear_probe, test_x, test_y, criterion, steps=0)
        
        evaluated = (train_steps,) if objective != STANDARD else tuple(step_counts)
        
        accuracies = {}
        for steps in evaluated:
            _, test_accuracy = evaluate_fm_probe(fm_layer, linear_probe, test_x, test_y, criterion, steps=steps)
            accuracies[steps] = test_accuracy
            
        deltas = {count: value - baseline_accuracy for count, value in accuracies.items()}

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": encoder,
            "dataset": dataset,
            "k": k,
            "seed": seed,
            "objective": objective,
            "target_type": target_type,
            "num_train": len(indices),
            "best_epoch": best_epoch,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_by_steps": {str(key): value for key, value in accuracies.items()},
            "delta_by_steps": {str(key): value for key, value in deltas.items()},
            "history": history,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "fm_layer": fm_layer.state_dict(),
            "linear_probe": linear_probe.state_dict()
        }, model_path)

        if record:
            for count, value in accuracies.items():
                record_run(
                    {
                        "method": f"{METHOD}_{objective}_{target_type}",
                        "dataset": dataset,
                        "encoder": encoder,
                        "k": k,
                        "seed": seed,
                        "steps": count,
                        "split": "test",
                        "accuracy": value,
                        "num_items": len(test_labels),
                    },
                    results_root,
                )

    if verbose:
        scores = "  ".join(
            f"T={count} {value:.4f} ({deltas[count]:+.4f})"
            for count, value in accuracies.items()
        )
        tag_str = "(loaded)" if loaded else ""
        print(
            f"    [{objective} {target_type}] {encoder:<14} {dataset:<9} k={str(k):<4} seed={seed}  "
            f"train={len(indices):>5}  base {baseline_accuracy:.4f}  {scores}  {tag_str}"
        )

    return {
        "encoder": encoder,
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "objective": objective,
        "target_type": target_type,
        "fm_layer": fm_layer, 
        "classifier": linear_probe,
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
    objectives: tuple[str, ...] = OBJECTIVES,
    target_types: tuple[str, ...] = TARGET_TYPES,
    step_counts: tuple[int, ...] = STEP_COUNTS,
    feature_root: Path | None = None,
    subset_root: Path | None = None,
    results_root: Path | None = None,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    record: bool = True,
) -> dict:
    cells = cells if cells is not None else LINEAR_PROBE_CELLS
    device = device if device is not None else default_device()
    print(f"Device: {device}")

    jobs = [
        (encoder, dataset, k, seed, objective, target_type)
        for encoder, dataset in cells
        for k in k_values
        for seed in seeds
        for objective in objectives
        for target_type in target_types
    ]

    results = {}
    for encoder, dataset, k, seed, objective, target_type in tqdm(jobs, desc="stage 3 grid"):
        result = run_stage3(
            encoder,
            dataset,
            k,
            seed,
            objective=objective,
            target_type=target_type,
            step_counts=step_counts,
            feature_root=feature_root,
            subset_root=subset_root,
            results_root=results_root,
            device=device,
            max_epochs=max_epochs,
            record=record,
        )
        key = f"{objective}/{target_type}/{encoder}/{dataset}/{k}/{seed}"
        results[key] = result

    print(f"\n{len(results)} Stage 3 FM runs complete.")
    return results
