# orbit-match/orbitmatch/plotting/diagnostic_plots.py
# Run: imported by sanity-check and dev scripts; not a runnable script.

"""Diagnostic plots for exploratory and sanity-check use.

These plots are *not* in the paper. They live here so that:

- ``run_sanity_checks.py`` can produce S5 (single-orbit visualization)
  consistently across runs.
- Dev iterations on policy code can compare lambda_2(Phi), edge counts,
  and deferral patterns visually without re-inventing plotting code.

Style
-----
Calls ``apply_theme(context='diagnostic')`` should be done by the
caller before these functions are invoked, so the figures use the
larger fonts and more generous sizing suited to exploration.

Every function returns ``(fig, ax)`` (or ``(fig, [ax, ...])`` when the
plot uses multiple axes). No file I/O.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.runner import SimulationResult
from orbitmatch.plotting.theme import COLORS
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Orbit-projection view
# ---------------------------------------------------------------------------


def plot_orbit_projection(
    positions: np.ndarray,
    *,
    epoch: int = 0,
    matchings: Optional[Sequence[np.ndarray]] = None,
    walker: Optional[WalkerDeltaConfig] = None,
    show_trails: bool = True,
    trail_epochs: int = 60,
    figsize: tuple[float, float] = (5.0, 5.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Top-down (equatorial-plane) projection of satellite positions.

    Plots an x-y projection of the constellation at one epoch, with
    optional trails over the previous ``trail_epochs`` epochs to show
    orbital motion. If a sequence of matchings is given, the realized
    matching at ``epoch`` is overlaid as line segments.

    Parameters
    ----------
    positions
        ``(n_epochs, n, 3)`` ECI position tensor in km.
    epoch
        Epoch to draw. Default 0.
    matchings
        Optional sequence of ``(k_t, 2)`` int arrays, one per epoch.
        If given, the matching at ``epoch`` is overlaid as edges.
    walker
        Optional WalkerDeltaConfig. If given, satellites are color-coded
        by plane.
    show_trails
        If True, draw a fading trail behind each satellite over the
        previous ``trail_epochs`` epochs.
    trail_epochs
        Number of past epochs to include in the trail.

    Returns
    -------
    fig, ax
    """
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"positions must have shape (n_epochs, n, 3); got {positions.shape}")
    n_epochs, n, _ = positions.shape
    if not (0 <= epoch < n_epochs):
        raise ValueError(f"epoch {epoch} out of range [0, {n_epochs}).")

    fig, ax = plt.subplots(figsize=figsize)

    # Earth disc.
    R_earth = 6378.137
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(
        R_earth * np.cos(theta), R_earth * np.sin(theta), color=COLORS.parchment, alpha=0.5, zorder=1, label="Earth"
    )
    ax.plot(R_earth * np.cos(theta), R_earth * np.sin(theta), color=COLORS.warm_gray, linewidth=0.5, zorder=2)

    # Plane colors.
    if walker is not None:
        plane_colors = [COLORS.burgundy, COLORS.copper, COLORS.olive, COLORS.warmbrown, COLORS.forest, COLORS.deep_red]
        sat_colors = [plane_colors[walker.plane_index(i) % len(plane_colors)] for i in range(n)]
    else:
        sat_colors = [COLORS.burgundy] * n

    # Trails.
    if show_trails and epoch > 0:
        t0 = max(0, epoch - trail_epochs)
        for i in range(n):
            xs = positions[t0 : epoch + 1, i, 0]
            ys = positions[t0 : epoch + 1, i, 1]
            ax.plot(xs, ys, color=sat_colors[i], linewidth=0.4, alpha=0.5, zorder=3)

    # Satellites at the current epoch.
    xs = positions[epoch, :, 0]
    ys = positions[epoch, :, 1]
    ax.scatter(xs, ys, c=sat_colors, s=18, zorder=5, edgecolors=COLORS.near_black, linewidths=0.4)

    # Matching overlay.
    if matchings is not None and epoch < len(matchings):
        m = matchings[epoch]
        if m.size > 0:
            segments = [
                [(positions[epoch, i, 0], positions[epoch, i, 1]), (positions[epoch, j, 0], positions[epoch, j, 1])]
                for i, j in m
            ]
            lc = LineCollection(segments, colors=COLORS.near_black, linewidths=0.8, zorder=4)
            ax.add_collection(lc)

    ax.set_aspect("equal")
    ax.set_xlabel("ECI x (km)")
    ax.set_ylabel("ECI y (km)")
    title = f"Constellation snapshot (epoch {epoch})"
    if matchings is not None and epoch < len(matchings):
        title += f", {matchings[epoch].shape[0]} links"
    ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------------
# Feasibility heatmap
# ---------------------------------------------------------------------------


def plot_feasibility_heatmap(
    feasibility: np.ndarray,
    *,
    fraction: bool = True,  # noqa: ARG001
    figsize: tuple[float, float] = (5.5, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Pair-vs-time heatmap of the feasibility tensor.

    Reshapes the ``(n_epochs, n, n)`` boolean tensor into an
    ``(n_epochs, n*(n-1)/2)`` matrix of upper-triangle pairs, then
    plots it as a heatmap with time on the x-axis and pair index on
    the y-axis. White means infeasible, dark means feasible.

    Parameters
    ----------
    feasibility
        Boolean ``(n_epochs, n, n)`` tensor.
    fraction
        If True, label the x-axis colorbar by feasibility fraction
        (sum / total) rather than raw boolean. Not currently used
        beyond labeling.

    Returns
    -------
    fig, ax
    """
    n_epochs, n, _ = feasibility.shape
    # Upper-triangle pair indices.
    iu, ju = np.triu_indices(n, k=1)
    # Shape: (n_epochs, n_pairs).
    pair_view = feasibility[:, iu, ju].astype(np.float32)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pair_view.T,
        aspect="auto",
        origin="lower",
        cmap="bone_r",
        interpolation="nearest",
    )
    ax.set_xlabel("Epoch $t$")
    ax.set_ylabel(f"Pair index (1..{n * (n - 1) // 2})")
    ax.set_title(f"Feasibility, mean = {feasibility.sum() / (n_epochs * n * (n - 1)):.3f}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Feasible (1) / Infeasible (0)")
    return fig, ax


# ---------------------------------------------------------------------------
# lambda_2(Phi) trace
# ---------------------------------------------------------------------------


def plot_lambda2_phi_trace(
    results_by_policy: dict[str, list[SimulationResult]],
    *,
    figsize: tuple[float, float] = (6.0, 3.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot lambda_2(Phi(t)) over time for one or more policies.

    Mirror of paper_plots.plot_lambda2_traces but for the windowed
    Laplacian Phi (the policy's internal smoothing object), not the
    realized union graph. Phi includes edge multiplicities, so its
    lambda_2 can grow significantly larger than the union's once the
    window fills.

    Used for debugging: if lambda_2(Phi) is doing something weird
    (oscillating, dropping unexpectedly), it shows up here clearly.

    Parameters
    ----------
    results_by_policy
        Same shape as paper_plots.plot_lambda2_traces.

    Returns
    -------
    fig, ax
    """
    from orbitmatch.plotting.theme import POLICY_COLORS, policy_style  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=figsize)

    first = next(iter(results_by_policy.values()))[0]
    T = first.T
    n_epochs = first.n_epochs
    x = np.arange(n_epochs)

    ax.axvspan(0, T, color=COLORS.parchment, alpha=0.35, zorder=0, label=f"warmup ($t < T={T}$)")

    for policy_name, results in results_by_policy.items():
        if policy_name in POLICY_COLORS:
            style = policy_style(policy_name)
            style["marker"] = None
        else:
            style = {"color": COLORS.near_black, "linestyle": "-", "marker": None}

        stacked = np.stack([r.lambda2_phi for r in results], axis=0)
        mean = stacked.mean(axis=0)
        ax.plot(x, mean, label=policy_name, zorder=3, **style)
        if len(results) > 1:
            std = stacked.std(axis=0)
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15, zorder=2)

    ax.set_xlabel("Epoch $t$")
    ax.set_ylabel(r"$\lambda_2(\Phi(t))$")
    ax.set_xlim(0, n_epochs - 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    return fig, ax


# ---------------------------------------------------------------------------
# Deferral histogram
# ---------------------------------------------------------------------------


def plot_deferral_histogram(
    result: SimulationResult,
    *,
    figsize: tuple[float, float] = (5.0, 3.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Distribution of per-epoch deferrals for a predictive-policy run.

    Histogram of the ``deferrals`` diagnostic (count of satellites that
    deferred, per epoch). Used by sanity check S3: the deferral
    mechanism should fire on at least 5% of epochs.

    Parameters
    ----------
    result
        A SimulationResult from a predictive (or equilibrium) policy run
        with the ``deferrals`` diagnostic recorded.

    Returns
    -------
    fig, ax

    Raises
    ------
    KeyError
        If the result has no ``deferrals`` diagnostic.
    """
    if "deferrals" not in result.policy_diagnostics:
        raise KeyError(
            f"Result has no 'deferrals' diagnostic; "
            f"only policies derived from PredictiveMatching record it. "
            f"Available diagnostics: {list(result.policy_diagnostics.keys())}"
        )

    deferrals = result.policy_diagnostics["deferrals"]
    n_epochs = len(deferrals)
    fired_epochs = int((deferrals > 0).sum())

    fig, ax = plt.subplots(figsize=figsize)
    max_d = int(deferrals.max()) if deferrals.size > 0 else 0
    bins = np.arange(0, max_d + 2) - 0.5
    ax.hist(deferrals, bins=bins, color=COLORS.burgundy, edgecolor=COLORS.near_black, linewidth=0.5)

    ax.set_xlabel("Deferrals per epoch")
    ax.set_ylabel("Number of epochs")
    ax.set_title(
        f"{result.policy_name}: {fired_epochs}/{n_epochs} epochs "
        f"deferred ({100 * fired_epochs / n_epochs:.1f}%), "
        f"total = {int(deferrals.sum())}"
    )
    return fig, ax


# ---------------------------------------------------------------------------
# Edge count over time
# ---------------------------------------------------------------------------


def plot_edge_count_trace(
    results_by_policy: dict[str, list[SimulationResult]],
    *,
    show_union: bool = True,
    figsize: tuple[float, float] = (6.0, 3.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Per-epoch edge counts: realized matching size and (optionally) union size.

    Two solid lines per policy: matching size (edges at epoch t) and
    union size (edges in [t-T+1, t]). The gap between them tells you
    how much the rolling window is decaying.

    Parameters
    ----------
    results_by_policy
        Per-policy lists of SimulationResults.
    show_union
        If True (default), plot both the matching size and the union
        size. If False, just the matching size.

    Returns
    -------
    fig, ax
    """
    from orbitmatch.plotting.theme import POLICY_COLORS, policy_style  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=figsize)

    first = next(iter(results_by_policy.values()))[0]
    n_epochs = first.n_epochs
    x = np.arange(n_epochs)

    for policy_name, results in results_by_policy.items():
        if policy_name in POLICY_COLORS:
            style = policy_style(policy_name)
            style["marker"] = None
        else:
            style = {"color": COLORS.near_black, "linestyle": "-", "marker": None}

        # Matching-size trace (mean across seeds).
        match_stack = np.stack([r.n_edges_per_epoch for r in results], axis=0)
        match_mean = match_stack.mean(axis=0)
        ax.plot(x, match_mean, label=f"{policy_name} (matching)", zorder=3, **style)

        if show_union:
            # Union-size trace, lighter and thinner so the matching trace dominates visually.
            union_stack = np.stack([r.n_union_edges_per_epoch for r in results], axis=0)
            union_mean = union_stack.mean(axis=0)
            style_union = dict(style)
            style_union["linestyle"] = ":"
            ax.plot(x, union_mean, label=f"{policy_name} (union)", alpha=0.6, zorder=2, **style_union)

    ax.set_xlabel("Epoch $t$")
    ax.set_ylabel("Number of edges")
    ax.set_xlim(0, n_epochs - 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", ncol=2 if show_union else 1)
    return fig, ax


# ---------------------------------------------------------------------------
# BR-rounds trace (equilibrium only)
# ---------------------------------------------------------------------------


def plot_br_rounds_trace(
    result: SimulationResult,
    *,
    figsize: tuple[float, float] = (5.5, 2.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Number of best-response rounds per epoch for an equilibrium run.

    Plots the ``br_rounds`` diagnostic over time. Useful for confirming
    that the equilibrium policy is not always trivially converging in
    one round (would suggest the warm-start is already a fixed point;
    not necessarily wrong but not interesting either).

    Parameters
    ----------
    result
        SimulationResult from an equilibrium-policy run.

    Returns
    -------
    fig, ax
    """
    if "br_rounds" not in result.policy_diagnostics:
        raise KeyError(
            f"Result has no 'br_rounds' diagnostic; "
            f"only the EquilibriumMatching policy records it. "
            f"Available diagnostics: {list(result.policy_diagnostics.keys())}"
        )

    rounds = result.policy_diagnostics["br_rounds"]
    converged = result.policy_diagnostics.get("br_converged")

    fig, ax = plt.subplots(figsize=figsize)
    n_epochs = len(rounds)
    x = np.arange(n_epochs)
    ax.plot(x, rounds, color=COLORS.copper, linewidth=1.0, marker=".", markersize=3)

    # Highlight non-converged epochs.
    if converged is not None:
        non_conv = np.flatnonzero(converged == 0)
        if non_conv.size > 0:
            ax.scatter(
                x[non_conv],
                rounds[non_conv],
                color=COLORS.deep_red,
                s=18,
                label=f"max_rounds hit ({non_conv.size})",
                zorder=5,
            )

    ax.set_xlabel("Epoch $t$")
    ax.set_ylabel("BR rounds")
    ax.set_title(f"{result.policy_name}: mean {rounds.mean():.2f}, max {int(rounds.max())}")
    if converged is not None and non_conv.size > 0:
        ax.legend(loc="best")
    ax.set_ylim(bottom=0)
    return fig, ax
