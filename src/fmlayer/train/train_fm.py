import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

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
CURVES_DIRNAME = "curves_fm"

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


def batch_loss(
    model: nn.Module,
    path,
    source: Tensor,
    target: Tensor,
    generator: torch.Generator,
) -> Tensor:
    sample = sample_paths(path, source, target, generator)
    velocity = model(x=sample.x_t, t=sample.t)
    loss = nn.functional.mse_loss(velocity, sample.dx_t)
    return loss


@torch.no_grad()
def evaluate_fm_probe(
    fm_layer: nn.Module, classifier: nn.Module, features: Tensor, labels: Tensor, criterion: nn.Module
) -> tuple[float, float]:
    fm_layer.eval()
    classifier.eval()
    from src.fmlayer.models.flow_ode import rollout
    z, _ = rollout(fm_layer, features, 12)
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

    from src.fmlayer.models.flow_matching_mlp import build_velocity_field
    fm_layer = build_velocity_field(embed_dim=embed_dim, seed=seed, device=device)

    centroids = compute_centroids(train_features, train_labels, num_classes, device)
    target = centroids[train_labels]
    
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
    
    for epoch in range(1, max_epochs + 1):
        fm_layer.train()
        order = torch.randperm(num_train, generator=generator).to(device)
        loss_total = 0.0
        
        for start in range(0, num_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            
            loss = batch_loss(
                fm_layer, path, train_features[batch], target[batch], generator
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
            val_loss, val_accuracy = evaluate_fm_probe(fm_layer, linear_probe, val_features, val_labels, criterion)
            entry["val_loss"] = val_loss
            entry["val_accuracy"] = val_accuracy
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_state = {key: value.detach().clone() for key, value in fm_layer.state_dict().items()}
                
        history.append(entry)
        scheduler.step()
        
    fm_layer.load_state_dict(best_state)
    return fm_layer, linear_probe, history, best_epoch

def curves_path_fm(
    encoder: str, dataset: str, k: int | str, seed: int, results_root: Path | None = None
) -> Path:
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / CURVES_DIRNAME / f"{encoder}_{dataset}_k{k}_seed{seed}.json"

def run_stage3(
    encoder: str,
    dataset: str,
    k: int | str,
    seed: int,
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

    fm_layer, linear_probe, history, best_epoch = train_fm_probe(
        train_x, train_y, val_x, val_y, num_classes, seed, device, max_epochs=max_epochs
    )

    criterion = nn.CrossEntropyLoss()
    _, test_accuracy = evaluate_fm_probe(fm_layer, linear_probe, test_x, test_y, criterion)
    best_val_accuracy = history[best_epoch - 1]["val_accuracy"]

    with torch.no_grad():
        from src.fmlayer.models.flow_ode import rollout
        z, _ = rollout(fm_layer, test_x, 12)
        predictions = linear_probe(z).argmax(dim=1).cpu().numpy()

    path = curves_path_fm(encoder, dataset, k, seed, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "encoder": encoder,
                "dataset": dataset,
                "k": k,
                "seed": seed,
                "num_train": len(indices),
                "best_epoch": best_epoch,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if record:
        record_run(
            {
                "method": METHOD,
                "dataset": dataset,
                "encoder": encoder,
                "k": k,
                "seed": seed,
                "split": "test",
                "accuracy": test_accuracy,
                "num_items": len(test_labels),
            },
            results_root,
        )

    if verbose:
        print(
            f"    [FM Layer] {encoder:<14} {dataset:<9} k={str(k):<4} seed={seed}  "
            f"train={len(indices):>5}  val={best_val_accuracy:.4f}@{best_epoch:<3}  "
            f"test={test_accuracy:.4f}"
        )

    return {
        "encoder": encoder,
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "fm_layer": fm_layer, "classifier": linear_probe,
        "test_accuracy": test_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "num_train": len(indices),
        "history": history,
        "predictions": predictions,
        "labels": test_labels,
    }

def run_all_stage3(
    cells: tuple[tuple[str, str], ...] | None = None,
    k_values: tuple[int | str, ...] = K_VALUES,
    seeds: tuple[int, ...] = SEEDS,
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
        (encoder, dataset, k, seed)
        for encoder, dataset in cells
        for k in k_values
        for seed in seeds
    ]

    results = {}
    for encoder, dataset, k, seed in tqdm(jobs, desc="fm linear probes"):
        result = run_stage3(
            encoder,
            dataset,
            k,
            seed,
            feature_root,
            subset_root,
            results_root,
            device,
            max_epochs,
            record,
        )
        results[f"{encoder}/{dataset}/{k}/{seed}"] = result

    print(f"\n{len(results)} FM linear-probe runs complete.")
    return results
