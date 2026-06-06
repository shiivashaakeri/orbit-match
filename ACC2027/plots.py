"""Figures for the ISL formation experiment.

``plot_comparison`` draws a three-panel cross-method comparison vs time:
algebraic connectivity (lambda_2), effective graph resistance, and ``log tau``
(the log spanning-tree count -- the quantity the game policy actually optimizes).
``plot_union_lambda2`` shows the cumulative-union analysis of both lambda_2 and
``log tau``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # safe for headless / batch runs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from satgame.metrics import logtau

# Consistent per-method styling.
_STYLE = {
    "game":     ("Game Theory", "#2563eb", "o", "-"),
    "furthest": ("Furthest",    "#dc2626", "s", "--"),
    "greedy":   ("Greedy",      "#d97706", "^", "-."),
    "mdmd":     ("MDMD",        "#7c3aed", "D", ":"),
}


def plot_comparison(metrics_by_method, out_path):
    """Three-panel lambda_2 / effective-resistance / log tau comparison.

    Parameters
    ----------
    metrics_by_method : dict[str, dict]
        method -> output of :func:`satgame.metrics.series_metrics` (metrics
        evaluated on the windowed-union graph).
    out_path : str or Path
        Where to save the PNG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Network Connectivity over Time — All Methods (windowed union)",
                 fontsize=13, fontweight="bold", y=1.02)

    ax1, ax2, ax3 = axes
    for method, m in metrics_by_method.items():
        label, color, marker, ls = _STYLE.get(
            method, (method, None, "o", "-")
        )
        t = np.arange(len(m["lambda2"]))
        ax1.plot(t, m["lambda2"], marker=marker, label=label,
                 color=color, linewidth=2, linestyle=ls)
        ax2.plot(t, m["resistance"], marker=marker, label=label,
                 color=color, linewidth=2, linestyle=ls)
        ax3.plot(t, m["logtau"], marker=marker, label=label,
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

    ax3.set_title("Log Spanning-Tree Count (log τ) — optimized objective",
                  fontsize=11)
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("log τ")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_union_lambda2(graphs, out_path):
    """Cumulative-union connectivity analysis for one method (typically the game).

    Two panels: λ₂ of the cumulative union (G_0 ∪ … ∪ G_t) and log τ of the
    cumulative union -- the quantity the game policy actually optimizes. The raw
    per-snapshot (single-epoch) curves are intentionally omitted: a single epoch
    is sparse/disconnected and not what the game optimizes.

    ``graphs`` is one method's list of per-epoch graphs.
    """
    cumulative = []
    cumulative_logtau = []
    union = graphs[0].copy()
    for i, G in enumerate(graphs):
        if i > 0:
            union.add_edges_from(G.edges())
        cumulative.append(nx.algebraic_connectivity(union))
        cumulative_logtau.append(logtau(union))

    t = np.arange(len(graphs))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Cumulative-Union Connectivity Over Time",
                 fontsize=14, fontweight="bold")

    axes[0].plot(t, cumulative, color="darkorange", linewidth=2,
                 marker="o", markersize=3, label="λ₂ of cumulative union")
    axes[0].set_title("λ₂ of Cumulative Graph Union (G₀ ∪ G₁ ∪ … ∪ Gₜ)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("λ₂ of union graph")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(t, cumulative_logtau, color="seagreen", linewidth=2,
                 marker="o", markersize=3, label="log τ of cumulative union")
    axes[1].set_title("log τ of Cumulative Graph Union — optimized objective")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("log τ of union graph")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
