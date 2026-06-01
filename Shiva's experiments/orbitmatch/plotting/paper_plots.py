# orbit-match/orbitmatch/plotting/paper_plots.py
# Run: imported by render scripts; not a runnable script.

"""Paper figure functions.

Four functions, one per paper figure:

- :func:`plot_lambda2_traces` — Fig 1, the headline. lambda_2 of the
  realized union graph over time, for all four policies, with the
  rho * alpha_0 horizon line from Theorem 6.
- :func:`plot_horizon_ablation` — Fig 2. Time-averaged lambda_2 vs.
  the lookahead horizon H, with seed-averaged error bars.
- :func:`plot_scaling` — Fig 3. Realized efficiency ratio vs. number
  of satellites n.
- :func:`plot_robustness` — Fig 4 (optional). lambda_2 trajectory
  through a satellite-dropout event, with recovery comparison.

Each function takes data in memory and returns a ``(fig, ax)`` pair.
Saving to disk is the caller's job; this module never writes files.

All colors, line widths, font sizes, and figure dimensions come from
:mod:`orbitmatch.plotting.theme`. The caller is expected to have called
``apply_theme()`` once at the top of its script.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from orbitmatch.experiments.runner import SimulationResult
from orbitmatch.experiments.sweep import SweepRecord
from orbitmatch.plotting.theme import (
    COLORS,
    FIG_HEIGHT_DEFAULT,
    FIG_WIDTH_DOUBLE_COL,
    FIG_WIDTH_SINGLE_COL,
    POLICY_COLORS,
    POLICY_LINESTYLES,
    policy_style,
)
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stack_seeds(results: Sequence[SimulationResult], attr: str) -> np.ndarray:
    """Stack a per-epoch trace across seeds into a (n_seeds, n_epochs) array."""
    arrays = [getattr(r, attr) for r in results]
    n_epochs = arrays[0].shape[0]
    if any(a.shape[0] != n_epochs for a in arrays):
        raise ValueError(
            f"Inconsistent n_epochs across seeds: {[a.shape[0] for a in arrays]}"
        )
    return np.stack(arrays, axis=0)


def _epochs_to_orbital(n_epochs: int, dt_s: float, orbital_period_s: float) -> np.ndarray:
    """Return a length-n_epochs array of times in units of orbital periods."""
    return np.arange(n_epochs) * dt_s / orbital_period_s


# ---------------------------------------------------------------------------
# Fig 1: lambda_2 traces
# ---------------------------------------------------------------------------


def plot_lambda2_traces(
    results_by_policy: dict[str, list[SimulationResult]],
    alpha_0: float,
    *,
    rho_bound: Optional[float] = None,
    x_in_periods: bool = True,
    orbital_period_s: Optional[float] = None,
    show_warmup: bool = True,
    figsize: tuple[float, float] = (FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot lambda_2(L_union_G(t; T)) over time for one or more policies.

    For each policy, plots the mean trace (solid line) and a +/- 1 std
    shaded band over seeds.

    Parameters
    ----------
    results_by_policy
        Map ``policy_name -> [SimulationResult per seed]``. Each list
        must contain at least one result; all results within a list
        must have the same length and dt_s.
    alpha_0
        The persistent-feasibility constant from Assumption 5. Used
        for the horizontal reference line at ``rho_bound * alpha_0``
        if ``rho_bound`` is given; otherwise just used to label the
        y-axis.
    rho_bound
        If given, draws a horizontal dashed line at ``rho_bound *
        alpha_0`` representing the certificate from Theorem 6. If None,
        the line is omitted.
    x_in_periods
        If True (default), x-axis is in units of orbital periods. If
        False, x-axis is in epochs.
    orbital_period_s
        Required when ``x_in_periods=True``. Orbital period in seconds.
    show_warmup
        If True, shade the warmup region [0, T) in a light parchment
        tint so readers see when the certificate kicks in.

    Returns
    -------
    fig, ax
        The matplotlib figure and axes. Caller saves via savefig.
    """
    if not results_by_policy:
        raise ValueError("results_by_policy is empty.")
    if x_in_periods and orbital_period_s is None:
        raise ValueError("orbital_period_s required when x_in_periods=True.")

    fig, ax = plt.subplots(figsize=figsize)

    # All policies share dt_s, T, and n_epochs by construction; pull from the
    # first one we see.
    first_result = next(iter(results_by_policy.values()))[0]
    dt_s = first_result.dt_s
    T = first_result.T
    n_epochs = first_result.n_epochs

    if x_in_periods:
        x = _epochs_to_orbital(n_epochs, dt_s, orbital_period_s)
        x_label = "Time (orbital periods)"
        x_warmup_end = T * dt_s / orbital_period_s
    else:
        x = np.arange(n_epochs)
        x_label = "Epoch $t$"
        x_warmup_end = float(T)

    # Warmup band first so it sits behind the curves.
    if show_warmup:
        ax.axvspan(0, x_warmup_end, color=COLORS.parchment, alpha=0.35, zorder=0,
                   label="warmup ($t < T$)")

    # Plot each policy: mean + shaded band over seeds.
    for policy_name, results in results_by_policy.items():
        if policy_name not in POLICY_COLORS:
            log.warning("plot_lambda2_traces: unknown policy %r; using default style", policy_name)
            style = {"color": COLORS.near_black, "linestyle": "-", "marker": None}
        else:
            style = policy_style(policy_name)
            style["marker"] = None  # no markers on dense traces

        stacked = _stack_seeds(results, "lambda2_union")
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        ax.plot(x, mean, label=_pretty_policy_name(policy_name), zorder=3, **style)
        if len(results) > 1:
            ax.fill_between(
                x, mean - std, mean + std,
                color=style["color"], alpha=0.15, zorder=2, linewidth=0,
            )

    # Theorem guarantee line.
    if rho_bound is not None:
        bound_value = rho_bound * alpha_0
        ax.axhline(
            bound_value,
            color=COLORS.near_black,
            linestyle="--",
            linewidth=1.0,
            zorder=4,
            label=fr"$\rho \cdot \alpha_0 = {bound_value:.2f}$",
        )

    # Alpha_0 reference line (always shown; the geometric ceiling).
    ax.axhline(
        alpha_0,
        color=COLORS.forest,
        linestyle=":",
        linewidth=0.8,
        zorder=4,
        label=fr"$\alpha_0 = {alpha_0:.2f}$",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\lambda_2(L^\cup_{\mathcal{G}}(t; T))$")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", ncol=1)

    return fig, ax


def _pretty_policy_name(name: str) -> str:
    """Convert internal policy name to a paper-friendly label."""
    pretty = {
        "predictive": "Predictive",
        "equilibrium": "Equilibrium (NE)",
        "greedy": "Greedy",
        "random": "Random",
    }
    return pretty.get(name, name)


# ---------------------------------------------------------------------------
# Fig 2: horizon ablation
# ---------------------------------------------------------------------------


def plot_horizon_ablation(
    records: Sequence[SweepRecord],
    *,
    metric: str = "rho_realized_mean",
    figsize: tuple[float, float] = (FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a sweep-averaged metric vs. the horizon H.

    Groups records by the H value in their override, then plots
    mean +/- std across seeds. All records must have a single-key
    override with key 'H'.

    Parameters
    ----------
    records
        SweepRecords from a run_sweep call with ``overrides`` of the
        form ``{"H": h_value}``.
    metric
        Which DiagnosticsReport field to plot. Default is
        ``rho_realized_mean`` (the post-warmup mean of the realized
        efficiency ratio).

    Returns
    -------
    fig, ax
        The matplotlib figure and axes.
    """
    h_values, means, stds = _aggregate_records(records, axis_key="H_", metric=metric)

    fig, ax = plt.subplots(figsize=figsize)

    color = POLICY_COLORS.get(records[0].policy_name, COLORS.burgundy)
    ax.errorbar(
        h_values, means, yerr=stds,
        color=color, marker="o", linestyle="-", linewidth=1.2,
        markersize=4, capsize=2, elinewidth=0.8,
        label=_pretty_policy_name(records[0].policy_name),
    )

    ax.set_xlabel(r"Lookahead horizon $H$")
    ax.set_ylabel(_metric_label(metric))
    ax.set_xscale("log")
    ax.set_xticks(h_values)
    ax.set_xticklabels([str(h) for h in h_values])
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    return fig, ax


# ---------------------------------------------------------------------------
# Fig 3: scaling
# ---------------------------------------------------------------------------


def plot_scaling(
    records_by_policy: dict[str, list[SweepRecord]],
    *,
    metric: str = "rho_realized_mean",
    figsize: tuple[float, float] = (FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a metric vs. constellation size n, one curve per policy.

    Each policy's records use overrides of the form
    ``{"walker": WalkerDeltaConfig(M=n, ...)}``. The override_label is
    expected to be of the form ``"n=<int>"``.

    Parameters
    ----------
    records_by_policy
        Map ``policy_name -> list of SweepRecord`` (one record set per
        policy, spanning all values of n and all seeds).
    metric
        DiagnosticsReport field to plot.

    Returns
    -------
    fig, ax
        The matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for policy_name, records in records_by_policy.items():
        n_values, means, stds = _aggregate_records(records, axis_key="n_", metric=metric)
        if policy_name in POLICY_COLORS:
            style = policy_style(policy_name)
        else:
            style = {"color": COLORS.burgundy, "linestyle": "-", "marker": "o"}
        ax.errorbar(
            n_values, means, yerr=stds,
            label=_pretty_policy_name(policy_name),
            markersize=4, capsize=2, elinewidth=0.8, linewidth=1.2,
            **style,
        )

    ax.set_xlabel(r"Number of satellites $n$")
    ax.set_ylabel(_metric_label(metric))
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    return fig, ax


# ---------------------------------------------------------------------------
# Fig 4: robustness
# ---------------------------------------------------------------------------


def plot_robustness(
    results_by_policy: dict[str, list[SimulationResult]],
    dropout_epoch: int,
    *,
    x_in_periods: bool = True,
    orbital_period_s: Optional[float] = None,
    figsize: tuple[float, float] = (FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot lambda_2 trajectories around a satellite-dropout event.

    The data layout is identical to :func:`plot_lambda2_traces`; the
    only addition is a vertical line marking the dropout epoch.

    Parameters
    ----------
    results_by_policy
        Per-policy lists of SimulationResults, one per seed.
    dropout_epoch
        Epoch index at which satellites are dropped. Drawn as a
        vertical line.

    Returns
    -------
    fig, ax
    """
    fig, ax = plot_lambda2_traces(
        results_by_policy,
        alpha_0=1.0,  # placeholder; robustness plot doesn't show the bound line
        rho_bound=None,
        x_in_periods=x_in_periods,
        orbital_period_s=orbital_period_s,
        show_warmup=False,
        figsize=figsize,
    )

    # Remove the alpha_0 reference line that plot_lambda2_traces draws
    # (irrelevant here -- alpha_0 changes when satellites drop).
    for line in list(ax.get_lines()):
        if line.get_linestyle() == ":":
            ydata = np.asarray(line.get_ydata())
            if ydata.size > 0 and abs(float(ydata.mean()) - 1.0) < 1e-9:
                line.remove()

    if x_in_periods:
        x_drop = dropout_epoch * results_by_policy[next(iter(results_by_policy))][0].dt_s / orbital_period_s
    else:
        x_drop = dropout_epoch

    ax.axvline(
        x_drop, color=COLORS.deep_red, linestyle="-", linewidth=0.8, zorder=5,
        label="dropout",
    )
    ax.legend(loc="best")
    return fig, ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_records(
    records: Sequence[SweepRecord],
    axis_key: str,
    metric: str,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Group records by axis value, return (values, means, stds).

    Parameters
    ----------
    records
        Sweep records.
    axis_key
        Prefix in the override_key identifying the axis (e.g. "H_",
        "T_", "n_"). The remainder is parsed as an int.
    metric
        DiagnosticsReport field to aggregate.

    Returns
    -------
    values
        Sorted list of axis values (ints).
    means
        Per-value means across seeds.
    stds
        Per-value stds across seeds.
    """
    by_value: dict[int, list[float]] = {}
    for r in records:
        if not r.override_key.startswith(axis_key):
            raise ValueError(f"Record {r.override_key} does not start with {axis_key!r}")
        v = int(r.override_key[len(axis_key):])
        m = getattr(r.report, metric)
        by_value.setdefault(v, []).append(float(m))

    values = sorted(by_value.keys())
    means = np.array([np.mean(by_value[v]) for v in values])
    stds = np.array([np.std(by_value[v]) for v in values])
    return values, means, stds


def _metric_label(metric: str) -> str:
    """Pretty-print a DiagnosticsReport field name for use as a y-axis label."""
    labels = {
        "rho_realized_final": r"$\rho_{\mathrm{realized}}$ (final)",
        "rho_realized_max": r"$\rho_{\mathrm{realized}}$ (max)",
        "rho_realized_mean": r"$\bar{\rho}_{\mathrm{realized}}$",
        "rho_cover_final": r"$\rho_{\mathrm{cover}}$",
    }
    return labels.get(metric, metric)