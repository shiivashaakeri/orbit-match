# orbit-match/scripts/run_consensus.py
# Run: python -m scripts.run_consensus

"""Discrete-time average consensus on realized matchings.

This script validates the corollary to Theorem 6: the certificate
lambda_2(L^cup_G(t; T)) >= rho * alpha_0 > 0 uniformly in t implies
that discrete-time average consensus on the realized matching graph
G(t) converges geometrically.

Procedure
---------
For each canonical trace in results/canonical/fig1/ (predictive,
greedy, random), iterate:

    x(t+1) = (I - mu * L_G(t)) x(t)

with mu = 1/3 (stable since max-degree of a matching is 1, so
lambda_max(L_G) <= 2). Initial state x(0) drawn from N(0, I) and
centered. Track disagreement delta(t) = ||x(t) - mean(x)||_2.

For statistical stability, repeat with N_TRIALS = 20 different
initial conditions per policy and report mean delta(t) and a 1-sigma
band over trials.

Output
------
- Per-policy delta(t) array saved to results/consensus/<policy>.npz
- Figure: figures/paper/fig2_consensus.pdf (semilog-y, all policies)
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

# Which policies to include and their canonical trace filenames.
POLICIES = {
    "predictive": "predictive_seed42.npz",
    "greedy":     "greedy_seed42.npz",
    "random":     "random_seed42.npz",
}

PRETTY_NAMES = {
    "predictive": "Predictive",
    "greedy":     "Greedy",
    "random":     "Random",
}


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
    delta_by_policy: dict[str, np.ndarray],
    T: int, n_epochs: int, dt_s: float, orbital_period_s: float,
    out_path: Path,
) -> None:
    """Plot disagreement delta(t) on a log-y axis."""
    apply_theme(context="paper")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT))

    x_periods = np.arange(n_epochs + 1) * dt_s / orbital_period_s
    warmup_end = T * dt_s / orbital_period_s
    ax.axvspan(0, warmup_end, color=COLORS.parchment, alpha=0.35, zorder=0)

    colors = {
        "predictive": POLICY_COLORS["predictive"],
        "greedy":     POLICY_COLORS["greedy"],
        "random":     POLICY_COLORS["random"],
    }
    styles = {"predictive": "-", "greedy": "--", "random": ":"}

    for policy, delta in delta_by_policy.items():
        mean = delta.mean(axis=0)
        # Numerical floor to keep log plot well-behaved.
        mean = np.maximum(mean, 1e-10)
        std = delta.std(axis=0)
        ax.semilogy(
            x_periods, mean,
            color=colors.get(policy, COLORS.near_black),
            linestyle=styles.get(policy, "-"),
            linewidth=1.2, zorder=3,
            label=PRETTY_NAMES.get(policy, policy),
        )
        # Optional ribbon (sigma in log space is awkward; using +sigma upper bound only).
        upper = np.maximum(mean + std, 1e-10)
        ax.fill_between(
            x_periods, mean, upper,
            color=colors.get(policy, COLORS.near_black),
            alpha=0.15, zorder=2, linewidth=0,
        )

    # Warmup label.
    y_top = ax.get_ylim()[1]
    ax.text(
        warmup_end / 2, y_top * 0.5,
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

    if not CANONICAL_DIR.exists():
        print(f"[ERROR] canonical traces directory not found: {CANONICAL_DIR}")
        print("Run scripts.stage_canonical_traces first.")
        return 1

    delta_by_policy: dict[str, np.ndarray] = {}
    n = T = dt_s = orbital_period_s = None
    n_epochs_first = None

    for policy, fname in POLICIES.items():
        path = CANONICAL_DIR / fname
        if not path.exists():
            print(f"[WARN] missing trace: {path}")
            continue

        actions, n_epochs = load_matchings(path)
        if n is None:
            n = actions.shape[1]
            # Pull metadata from the trace for axis labels.
            _, manifest = load_trace(path)
            user = manifest.get("user", {})
            policy_params = user.get("policy_params", {})
            T = int(policy_params.get("T", user.get("T", 574)))
            dt_s = float(user.get("dt_s", 10.0))
            orbital_period_s = T * dt_s
            n_epochs_first = n_epochs
        elif n_epochs != n_epochs_first:
            print(f"[WARN] {policy}: n_epochs mismatch "
                  f"({n_epochs} vs {n_epochs_first}); skipping")
            continue

        print(f"Running consensus for {policy}: n={n}, n_epochs={n_epochs}, "
              f"{N_TRIALS} trials...")
        t0 = time.perf_counter()
        delta = run_consensus(actions, n, n_epochs)
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s. "
              f"Initial delta {delta[:, 0].mean():.3f}, "
              f"final delta {delta[:, -1].mean():.3e}")

        np.save(OUT_DIR / f"{policy}_disagreement.npy", delta)
        delta_by_policy[policy] = delta

    if not delta_by_policy:
        print("[ERROR] no policies loaded")
        return 1

    print()
    print("Summary (post-warmup disagreement reduction):")
    print(f"  {'policy':<14} {'delta(T)':>10} {'delta(end)':>12} {'reduction':>12}")
    for policy, delta in delta_by_policy.items():
        d_at_T = delta[:, T].mean()
        d_at_end = delta[:, -1].mean()
        ratio = d_at_end / max(d_at_T, 1e-15)
        print(f"  {policy:<14} {d_at_T:>10.4e} {d_at_end:>12.4e} {ratio:>12.4e}")

    out_path = figures_dir("paper") / "fig2_consensus.pdf"
    plot_consensus(delta_by_policy, T, n_epochs_first, dt_s, orbital_period_s, out_path)
    print(f"\nSaved figure to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
