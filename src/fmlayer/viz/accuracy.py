from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fmlayer.data.fewshot import K_FULL
from src.fmlayer.data.specs import get_spec
from src.fmlayer.models.zeroshot import METHOD as ZEROSHOT_METHOD
from src.fmlayer.train.train_linear import METHOD as PROBE_METHOD
from src.fmlayer.viz.figures import (
    ENCODER_COLORS,
    ENCODER_LABELS,
    ENCODER_MARKERS,
    apply_plot_style,
    save_figure,
)

K_ORDER = ("5", "10", K_FULL)


def k_position(k: str) -> float:
    """Map a training-set size onto the x axis."""
    return float(K_ORDER.index(k))


def plot_combined_accuracy_vs_k(
    table: pd.DataFrame,
    datasets: list[str] | None = None,
    show: bool = True,
) -> None:
    """Plot accuracy vs K for all datasets together side-by-side in a single combined figure."""
    apply_plot_style()
    datasets = datasets if datasets is not None else ["dtd", "aircraft"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = table[table["dataset"] == dataset]
        probes = subset[subset["method"] == PROBE_METHOD]
        for encoder, rows in probes.groupby("encoder"):
            rows = rows.copy()
            rows["position"] = rows["k"].map(k_position)
            rows = rows.sort_values("position")
            color = ENCODER_COLORS.get(encoder, "#333333")
            marker = ENCODER_MARKERS.get(encoder, "o")
            label = ENCODER_LABELS.get(encoder, encoder)
            ax.errorbar(
                rows["position"],
                rows["mean"],
                yerr=rows["std"],
                marker=marker,
                color=color,
                linewidth=2,
                markersize=7,
                capsize=5,
                capthick=1.5,
                label=f"Linear probe ({label})",
            )

        zeroshot = subset[subset["method"] == ZEROSHOT_METHOD]
        for _, row in zeroshot.iterrows():
            clip_color = ENCODER_COLORS.get(row["encoder"], "#d62728")
            clip_label = ENCODER_LABELS.get(row["encoder"], row["encoder"])
            ax.axhline(
                row["mean"],
                color=clip_color,
                linestyle="--",
                linewidth=1.8,
                label=f"Zero-shot {clip_label} ({row['mean'] * 100:.1f}%)",
            )

        ax.set_xticks(range(len(K_ORDER)), [f"K = {k}" for k in K_ORDER])
        ax.set_xlabel("Training Images per Class (K)", labelpad=8)
        ax.set_ylabel("Top-1 Test Accuracy", labelpad=8)
        ax.set_title(f"{get_spec(dataset).display_name}", pad=10)
        ax.legend(loc="best", frameon=True, framealpha=0.95)

    fig.suptitle("Top-1 Accuracy vs. Training-Set Size (K)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, "accuracy_vs_k_combined", show=show, save=False)


def plot_accuracy_vs_k(
    dataset: str,
    table: pd.DataFrame,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Plot accuracy vs K for a single dataset."""
    apply_plot_style()
    subset = table[table["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(7, 4.8))

    probes = subset[subset["method"] == PROBE_METHOD]
    for encoder, rows in probes.groupby("encoder"):
        rows = rows.copy()
        rows["position"] = rows["k"].map(k_position)
        rows = rows.sort_values("position")
        color = ENCODER_COLORS.get(encoder, "#333333")
        marker = ENCODER_MARKERS.get(encoder, "o")
        label = ENCODER_LABELS.get(encoder, encoder)
        ax.errorbar(
            rows["position"],
            rows["mean"],
            yerr=rows["std"],
            marker=marker,
            color=color,
            linewidth=2,
            markersize=7,
            capsize=5,
            capthick=1.5,
            label=f"Linear probe ({label})",
        )

    zeroshot = subset[subset["method"] == ZEROSHOT_METHOD]
    for _, row in zeroshot.iterrows():
        clip_color = ENCODER_COLORS.get(row["encoder"], "#d62728")
        clip_label = ENCODER_LABELS.get(row["encoder"], row["encoder"])
        ax.axhline(
            row["mean"],
            color=clip_color,
            linestyle="--",
            linewidth=1.8,
            label=f"Zero-shot {clip_label} ({row['mean'] * 100:.1f}%)",
        )

    ax.set_xticks(range(len(K_ORDER)), [f"K = {k}" for k in K_ORDER])
    ax.set_xlabel("Training Images per Class (K)", labelpad=8)
    ax.set_ylabel("Top-1 Test Accuracy", labelpad=8)
    ax.set_title(f"{get_spec(dataset).display_name}: Accuracy vs. Training-Set Size", pad=12)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()

    return save_figure(fig, f"accuracy_vs_k_{dataset}", figures_root, show=show, save=save)


def plot_accuracy_vs_k_all(
    table: pd.DataFrame,
    datasets: list[str] | None = None,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> list[Path | None]:
    """Draw the accuracy-versus-K figure for several datasets."""
    datasets = datasets if datasets is not None else sorted(np.unique(table["dataset"]))
    return [plot_accuracy_vs_k(dataset, table, figures_root, show=show, save=save) for dataset in datasets]
