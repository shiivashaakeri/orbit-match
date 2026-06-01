# orbit-match/scripts/run_consensus.py
# Run: python -m scripts.run_consensus

"""Discrete-time average consensus on the realized matching sequence.

This script demonstrates that Theorem 6's certificate implies geometric
consensus convergence on the realized network. Per the corollary, when
the rolling-window union $\\mathcal{G}^\\cup(t; T)$ satisfies
$\\lambda_2 \\geq \\rho \\alpha_0 > 0$ uniformly in $t$, the discrete-time
consensus iteration

    x(t+1) = (I - mu * L_G(t)) x(t)

converges to consensus geometrically. The script runs this iteration
on the canonical predictive trace and plots the disagreement
$\\|x(t) - \\bar{x}\\mathbf{1}\\|_2$ on a log scale.

We only plot predictive. The corollary is about the certificate
holding; comparing decay rates between policies confounds
joint-connectivity (which the certificate guarantees) with mixing
(which depends on graph dynamics, not on $\\lambda_2$ alone).

Output
------
- Disagreement array saved to results/consensus/predictive_disagreement.npy
- Figure: figures/paper/fig2_consensus.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbitmatch.plotting.theme import (
    COLORS,
    FIG_HEIGHT_DEFAULT,
    FIG_WIDTH_SINGLE_COL,
    POLICY_COLORS,
    apply_theme,
)
from orbitmatch.utils.io import RESULTS_ROOT, figures_dir, load_trace
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)

CANONICAL_DIR = RESULTS_ROOT / "canonical" / "fig1"
OUT_DIR = RESULTS_ROOT / "consensus"
N_TRIALS = 20
MU = 1.0 / 3.0  # consensus step size; safe for max degree <= 2

PREDICTIVE_TRACE = "predictive_seed42.npz"
SHOW_THEORETICAL_BOUND = False  # set True to overlay the corollary's conservative rate


def load_matchings(path: Path) -> tuple[np.ndarray, int]:
    """Load actions array from a canonical trace.

    Returns (actions, n_epochs). Actions is shape (n_epochs, n); each
    entry is the partner index or -1 (NO_LINK).
    """
    arrays, _ = load_trace(path)
    actions = arrays["actions"]
    return actions, actions.shape[0]


def matching_laplacian(actions_t: np.ndarray, n: int) -> np.ndarray:
    """Build the n x n Laplacian of the realized matching at epoch t.

    A link (i, j) forms iff actions[i] == j AND actions[j] == i. The
    Laplacian L_G is diag(d) - A where A is the symmetric adjacency.
    """
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = int(actions_t[i])
        if j < 0 or j >= n:
            continue
        if int(actions_t[j]) != i:
            continue
        if i < j:  # avoid double-counting
            L[i, i] += 1.0
            L[j, j] += 1.0
            L[i, j] -= 1.0
            L[j, i] -= 1.0
    return L


def run_consensus(actions: np.ndarray, n: int, n_epochs: int,
                  n_trials: int = N_TRIALS, mu: float = MU,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Run consensus dynamics over the trace.

    Returns disagreement[trial, t]: ||x(t) - mean(x(t))||_2.
    """
    if rng is None:
        rng = np.random.default_rng(2026)

    disagreement = np.zeros((n_trials, n_epochs + 1), dtype=np.float64)

    for trial in range(n_trials):
        x = rng.standard_normal(n)
        x = x - x.mean()
        disagreement[trial, 0] = float(np.linalg.norm(x))

        for t in range(n_epochs):
            L = matching_laplacian(actions[t], n)
            x = x - mu * (L @ x)
            disagreement[trial, t + 1] = float(np.linalg.norm(x))

    return disagreement


def plot_consensus(
    delta: np.ndarray,
    T: int, n_epochs: int, dt_s: float, orbital_period_s: float,
    alpha_0: float | None,
    out_path: Path,
) -> None:
    """Plot disagreement delta(t) for predictive on a log-y axis.

    Single curve with a sigma ribbon. Warmup region shaded. Optional
    theoretical-bound reference line if SHOW_THEORETICAL_BOUND is True
    and alpha_0 is known.
    """
    apply_theme(context="paper")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT))

    x_periods = np.arange(n_epochs + 1) * dt_s / orbital_period_s
    warmup_end = T * dt_s / orbital_period_s
    ax.axvspan(0, warmup_end, color=COLORS.parchment, alpha=0.35, zorder=0)

    color = POLICY_COLORS.get("predictive", COLORS.burgundy)

    mean = delta.mean(axis=0)
    mean = np.maximum(mean, 1e-12)  # log-plot floor
    std = delta.std(axis=0)
    ax.semilogy(
        x_periods, mean,
        color=color, linestyle="-", linewidth=1.4, zorder=3,
        label="Predictive",
    )
    upper = np.maximum(mean + std, 1e-12)
    lower = np.maximum(mean - std, 1e-12)
    ax.fill_between(
        x_periods, lower, upper,
        color=color, alpha=0.20, zorder=2, linewidth=0,
    )

    # Optional: theoretical decay bound from the corollary.
    if SHOW_THEORETICAL_BOUND and alpha_0 is not None:
        # Conservative per-window decay: (1 - mu * rho * alpha_0 / T)^(t/T).
        # Plotted with rho ~ 1 (post-warmup, where the certificate saturates).
        rho_alpha = alpha_0
        per_window_rate = max(1.0 - MU * rho_alpha / T, 1e-9)
        # Convert to per-epoch rate.
        per_epoch_rate = per_window_rate ** (1.0 / T)
        d0 = float(mean[0])
        bound = d0 * per_epoch_rate ** np.arange(n_epochs + 1)
        ax.semilogy(
            x_periods, np.maximum(bound, 1e-12),
            color=COLORS.warm_gray, linestyle=":", linewidth=0.9, zorder=2,
            label="corollary bound",
        )

    ax.text(
        warmup_end / 2, ax.get_ylim()[1] * 0.5,
        "warmup", color=COLORS.warm_gray, fontsize=7, ha="center", va="center",
    )

    ax.set_xlabel("Time (orbital periods)")
    ax.set_ylabel(r"Disagreement $\|x(t) - \bar{x}\mathbf{1}\|_2$")
    ax.set_xlim(0, x_periods[-1])
    ax.legend(loc="upper right", fontsize=7)

    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    configure(level="WARNING")
    OUT_DIR.mkdir(exist_ok=True)

    path = CANONICAL_DIR / PREDICTIVE_TRACE
    if not path.exists():
        print(f"[ERROR] canonical predictive trace not found: {path}")
        print("Run scripts.stage_canonical_traces first.")
        return 1

    actions, n_epochs = load_matchings(path)
    n = actions.shape[1]
    _, manifest = load_trace(path)
    user = manifest.get("user", {})
    policy_params = user.get("policy_params", {})
    T = int(policy_params.get("T", user.get("T", 574)))
    dt_s = float(user.get("dt_s", 10.0))
    orbital_period_s = T * dt_s

    # Pull alpha_0 from the diagnostics report if available.
    report = user.get("report", {})
    alpha_0 = float(report.get("alpha_0", 0.0)) or None

    print(f"Predictive trace: n={n}, n_epochs={n_epochs}, T={T}")
    print(f"alpha_0 = {alpha_0 if alpha_0 is not None else 'unknown'}")
    print(f"Running {N_TRIALS} consensus trials...")
    t0 = time.perf_counter()
    delta = run_consensus(actions, n, n_epochs)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s. "
          f"Initial delta {delta[:, 0].mean():.3f}, "
          f"final delta {delta[:, -1].mean():.3e}")

    np.save(OUT_DIR / "predictive_disagreement.npy", delta)

    # Summary statistics.
    print()
    print("Disagreement decay (predictive):")
    print(f"  at t = 0:           {delta[:, 0].mean():.4e}")
    print(f"  at t = T (warmup):  {delta[:, T].mean():.4e}")
    print(f"  at t = end:         {delta[:, -1].mean():.4e}")
    print(f"  post-warmup ratio:  {delta[:, -1].mean() / max(delta[:, T].mean(), 1e-15):.4e}")
    decay_orders = np.log10(delta[:, 0].mean() / max(delta[:, -1].mean(), 1e-15))
    print(f"  total decay:        {decay_orders:.2f} orders of magnitude")

    out_path = figures_dir("paper") / "fig2_consensus.pdf"
    plot_consensus(delta, T, n_epochs, dt_s, orbital_period_s, alpha_0, out_path)
    print(f"\nSaved figure to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
