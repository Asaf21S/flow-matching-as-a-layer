from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fmlayer.data.fewshot import K_FULL
from src.fmlayer.data.specs import get_spec
from src.fmlayer.viz.figures import apply_plot_style, save_figure

K_ORDER = ("5", "10", K_FULL)
POSITIVE_COLOR = "#2ca02c"
NEGATIVE_COLOR = "#d62728"
BASELINE_COLOR = "#333333"


def k_position(k: str) -> float:
    """Map a training-set size onto the x axis."""
    return float(K_ORDER.index(str(k)))


def plot_config_ablation(
    table: pd.DataFrame,
    encoder: str,
    dataset: str,
    k: int | str = K_FULL,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Bar chart of every flow configuration against the frozen-probe baseline.

    Args:
        table: Aggregated table from :func:`stage3_table`.
        encoder: Encoder key.
        dataset: Dataset key.
        k: Training-set size to chart.
        figures_root: Output directory.
        show: Display the figure.
        save: Write a PNG.

    Returns:
        Path of the saved figure, or ``None``.
    """
    apply_plot_style()
    rows = table[
        (table["encoder"] == encoder) & (table["dataset"] == dataset) & (table["k"].astype(str) == str(k))
    ]
    if rows.empty:
        return None

    # One bar per configuration, at whichever T it scored best.
    best = rows.sort_values("acc_mean", ascending=False).groupby("config_name", as_index=False).first()
    best = best.sort_values("acc_mean", ascending=False)
    baseline = float(rows["baseline_mean"].mean())

    labels = [f"{name}\n(T={int(steps)})" for name, steps in zip(best["config_name"], best["steps"])]
    means = best["acc_mean"].to_numpy()
    errors = best["acc_std"].to_numpy()
    colors = [POSITIVE_COLOR if value >= baseline else NEGATIVE_COLOR for value in means]

    fig, ax = plt.subplots(figsize=(max(9.0, 0.85 * len(labels)), 5.2))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, means, yerr=errors, capsize=4, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(
        baseline,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Frozen linear probe ({baseline:.4f})",
    )

    for bar, value, error in zip(bars, means, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 0.004,
            f"{value - baseline:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 Test Accuracy")
    ax.set_title(f"{get_spec(dataset).display_name} ({encoder}), K = {k}", pad=12)

    lowest = min(baseline, float(np.min(means)))
    highest = max(baseline, float(np.max(means + errors)))
    padding = 0.1 * max(highest - lowest, 0.01)
    ax.set_ylim(max(0.0, lowest - padding), min(1.0, highest + padding))
    ax.legend(loc="best")
    fig.tight_layout()

    return save_figure(fig, f"stage3_ablation_{dataset}_{encoder}_k{k}", figures_root, show=show, save=save)


def plot_accuracy_vs_k(
    table: pd.DataFrame,
    encoder: str,
    dataset: str,
    config_names: list[str] | None = None,
    top: int = 4,
    figures_root: Path | None = None,
    show: bool = True,
    save: bool = False,
) -> Path | None:
    """Accuracy versus K with error bars, flow configurations against the baseline.

    Args:
        table: Aggregated table from :func:`stage3_table`.
        encoder: Encoder key.
        dataset: Dataset key.
        config_names: Configurations to draw; defaults to the ``top`` best at K = full.
        top: Number of configurations to pick when ``config_names`` is not given.
        figures_root: Output directory.
        show: Display the figure.
        save: Write a PNG.

    Returns:
        Path of the saved figure, or ``None``.
    """
    apply_plot_style()
    rows = table[(table["encoder"] == encoder) & (table["dataset"] == dataset)]
    if rows.empty:
        return None

    if config_names is None:
        at_full = rows[rows["k"].astype(str) == K_FULL].sort_values("acc_mean", ascending=False)
        config_names = list(dict.fromkeys(at_full["config_name"]))[:top]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = plt.get_cmap("tab10")

    for index, name in enumerate(config_names):
        subset = rows[rows["config_name"] == name]
        # One point per K, at the best T for that configuration.
        best = subset.sort_values("acc_mean", ascending=False).groupby("k", as_index=False).first()
        best = best.assign(position=best["k"].map(k_position)).sort_values("position")
        ax.errorbar(
            best["position"],
            best["acc_mean"],
            yerr=best["acc_std"],
            marker="o",
            linewidth=1.8,
            markersize=6,
            capsize=4,
            color=colors(index % 10),
            label=name,
        )

    baseline = rows.groupby("k", as_index=False)["baseline_mean"].mean()
    baseline = baseline.assign(position=baseline["k"].map(k_position)).sort_values("position")
    ax.errorbar(
        baseline["position"],
        baseline["baseline_mean"],
        marker="s",
        linestyle="--",
        linewidth=2.2,
        markersize=7,
        color=BASELINE_COLOR,
        label="Frozen linear probe",
    )

    ax.set_xticks(range(len(K_ORDER)), [f"K = {k}" for k in K_ORDER])
    ax.set_xlabel("Training Images per Class (K)", labelpad=8)
    ax.set_ylabel("Top-1 Test Accuracy", labelpad=8)
    ax.set_title(f"{get_spec(dataset).display_name} ({encoder})", pad=12)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    return save_figure(fig, f"stage3_accuracy_vs_k_{dataset}_{encoder}", figures_root, show=show, save=save)
