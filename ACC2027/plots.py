"""Figures for the ISL formation experiment.

``plot_comparison`` reproduces the notebook's cell-7 two-panel comparison
(algebraic connectivity and effective resistance vs time). ``plot_union_lambda2``
reproduces the cumulative windowed-union lambda_2 analysis (cells 12-13), which
is the quantity the game policy actually optimizes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # safe for headless / batch runs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Consistent per-method styling.
_STYLE = {
    "game":     ("Game Theory", "#2563eb", "o", "-"),
    "furthest": ("Furthest",    "#dc2626", "s", "--"),
    "greedy":   ("Greedy",      "#d97706", "^", "-."),
    "mdmd":     ("MDMD",        "#7c3aed", "D", ":"),
}


def plot_comparison(metrics_by_method, out_path):
    """Two-panel lambda_2 / effective-resistance comparison across methods.

    Parameters
    ----------
    metrics_by_method : dict[str, dict]
        method -> output of :func:`satgame.metrics.series_metrics`.
    out_path : str or Path
        Where to save the PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Network Connectivity over Time — All Methods",
                 fontsize=13, fontweight="bold", y=1.02)

    ax1, ax2 = axes
    for method, m in metrics_by_method.items():
        label, color, marker, ls = _STYLE.get(
            method, (method, None, "o", "-")
        )
        t = np.arange(len(m["lambda2"]))
        ax1.plot(t, m["lambda2"], marker=marker, label=label,
                 color=color, linewidth=2, linestyle=ls)
        ax2.plot(t, m["resistance"], marker=marker, label=label,
                 color=color, linewidth=2, linestyle=ls)

    ax1.set_title("Algebraic Connectivity (λ₂)", fontsize=11)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("λ₂")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Effective Graph Resistance", fontsize=11)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Resistance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_union_lambda2(graphs, out_path):
    """Per-snapshot lambda_2 vs cumulative-union lambda_2 (G_0 ∪ ... ∪ G_t).

    ``graphs`` is one method's list of per-epoch graphs (typically the game's).
    """
    per_snapshot = [nx.algebraic_connectivity(G) for G in graphs]

    cumulative = []
    union = graphs[0].copy()
    for i, G in enumerate(graphs):
        if i > 0:
            union.add_edges_from(G.edges())
        cumulative.append(nx.algebraic_connectivity(union))

    t = np.arange(len(graphs))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("λ₂ Analysis Over Time", fontsize=14, fontweight="bold")

    axes[0].plot(t, per_snapshot, color="steelblue", linewidth=1.5,
                 marker="o", markersize=3, label="λ₂ per snapshot")
    axes[0].axhline(np.mean(per_snapshot), color="red", linestyle="--",
                    label=f"Mean: {np.mean(per_snapshot):.4f}")
    axes[0].set_title("Raw λ₂ at Each Snapshot")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("λ₂")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(t, cumulative, color="darkorange", linewidth=2,
                 marker="o", markersize=3, label="λ₂ of cumulative union")
    axes[1].set_title("λ₂ of Cumulative Graph Union (G₀ ∪ G₁ ∪ … ∪ Gₜ)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("λ₂ of union graph")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
