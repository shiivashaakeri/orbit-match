# orbit-match/scripts/render_paper_figures.py
# Run: python -m scripts.render_paper_figures

"""Read saved traces from results/canonical/ and render paper figures.

Pure consumer of the canonical traces staged by
`scripts/stage_canonical_traces.py`. Does not run any simulations;
only loads .npz files and calls the plotting functions.

Current figures
---------------
- Fig 1: lambda_2 traces. Inputs from results/canonical/fig1/.
  Four policies: predictive (k1_sync), greedy, random, k1_gs.
- Fig 3: update-order ablation. Inputs from results/canonical/fig3/.
  Two-panel: headline comparison + ordering remark.

Output
------
PDFs in figures/paper/.

Behavior on missing data
------------------------
If a canonical subdirectory is empty or missing, the script logs a
warning and skips that figure. It does not fail.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbitmatch.experiments.diagnostics import DiagnosticsReport
from orbitmatch.experiments.runner import SimulationResult
from orbitmatch.experiments.sweep import SweepRecord
from orbitmatch.plotting.paper_plots import (
    plot_horizon_ablation,
    plot_lambda2_traces,
    plot_scaling,
)
from orbitmatch.plotting.theme import (
    COLORS,
    FIG_HEIGHT_DEFAULT,
    FIG_WIDTH_DOUBLE_COL,
    FIG_WIDTH_SINGLE_COL,
    POLICY_COLORS,
    apply_theme,
)
from orbitmatch.utils.io import RESULTS_ROOT, figures_dir, load_trace
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


CANONICAL_ROOT = RESULTS_ROOT / "canonical"


# ---------------------------------------------------------------------------
# Trace -> SimulationResult reconstruction
# ---------------------------------------------------------------------------


def load_simulation_result(path: Path) -> tuple[SimulationResult, DiagnosticsReport]:
    """Reconstruct a SimulationResult and its DiagnosticsReport from disk."""
    arrays, manifest = load_trace(path)
    user = manifest.get("user", {})

    rep_dict = user.get("report")
    if rep_dict is None:
        raise ValueError(f"Trace {path} has no 'report' in manifest")
    report = DiagnosticsReport(**rep_dict)

    diag: dict[str, np.ndarray] = {}
    for k in list(arrays.keys()):
        if k.startswith("diag_"):
            diag[k[len("diag_"):]] = arrays.pop(k)

    matchings_obj = arrays.pop("matchings")
    matchings = tuple(np.asarray(m, dtype=np.int64) for m in matchings_obj)

    # Some manifests have nested walker dict; some don't.
    walker_meta = user.get("walker", {})
    n_from_meta = walker_meta.get("M") if walker_meta else None
    n = int(n_from_meta) if n_from_meta is not None else int(arrays["actions"].shape[1])

    # T comes from policy_params if present, else from manifest top-level.
    pp = user.get("policy_params", {})
    T = int(pp.get("T", user.get("T", 0)))

    result = SimulationResult(
        policy_name=user.get("policy_name", "unknown"),
        config_label=user.get("config_label", "unknown"),
        n=n,
        n_epochs=int(user.get("n_epochs", arrays["lambda2_union"].shape[0])),
        T=T,
        dt_s=float(user.get("dt_s", 10.0)),
        lambda2_phi=arrays["lambda2_phi"],
        lambda2_union=arrays["lambda2_union"],
        n_edges_per_epoch=arrays["n_edges_per_epoch"],
        n_union_edges_per_epoch=arrays["n_union_edges_per_epoch"],
        actions=arrays["actions"],
        matchings=matchings,
        final_phi=arrays["final_phi"],
        final_union_adjacency=arrays["final_union_adjacency"],
        policy_diagnostics=diag,
        metadata=user,
    )
    return result, report


# ---------------------------------------------------------------------------
# Fig 1: lambda_2 traces (canonical layout)
# ---------------------------------------------------------------------------


# Filenames in canonical/fig1/ follow {policy_label}_seed{N}.npz.
FIG1_PATTERN = re.compile(r"^(?P<policy>.+?)_seed(?P<seed>\d+)\.npz$")


def render_fig1(out_dir: Path) -> bool:
    """Build Fig 1 from results/canonical/fig1/."""
    data_dir = CANONICAL_ROOT / "fig1"
    if not data_dir.exists():
        log.warning("Fig 1: no data at %s; skipping. Run stage_canonical_traces.", data_dir)
        return False

    traces = sorted(data_dir.glob("*.npz"))
    if not traces:
        log.warning("Fig 1: no .npz files in %s; skipping.", data_dir)
        return False

    # Group by the policy_label from the filename.
    results_by_policy: dict[str, list[SimulationResult]] = {}
    reports_by_policy: dict[str, list[DiagnosticsReport]] = {}
    for path in traces:
        m = FIG1_PATTERN.match(path.name)
        if m is None:
            log.warning("Fig 1: skipping unrecognized filename %s", path.name)
            continue
        policy_label = m.group("policy")
        result, report = load_simulation_result(path)
        results_by_policy.setdefault(policy_label, []).append(result)
        reports_by_policy.setdefault(policy_label, []).append(report)

    first_report = next(iter(reports_by_policy.values()))[0]
    alpha_0 = first_report.alpha_0

    first_result = next(iter(results_by_policy.values()))[0]
    orbital_period_s = first_result.T * first_result.dt_s

    print(f"Fig 1: loaded {len(traces)} traces, {len(results_by_policy)} policies")
    for policy_label, results in sorted(results_by_policy.items()):
        reports = reports_by_policy[policy_label]
        rho_max = float(np.mean([r.rho_realized_max for r in reports]))
        rho_mean = float(np.mean([r.rho_realized_mean for r in reports]))
        print(f"  {policy_label:<14} ({len(results)} seeds): "
              f"rho_max={rho_max:.4f}, rho_mean={rho_mean:.4f}")

    # Canonical ordering of curves: predictive, k1_gs, greedy, random.
    # (Equilibrium is intentionally NOT in Fig 1 anymore; per FINDINGS F13
    # it's a strictly worse NE than k1_gs.)
    canonical_order = ["predictive", "k1_gs", "greedy", "random"]
    ordered = {p: results_by_policy[p] for p in canonical_order if p in results_by_policy}
    for p in results_by_policy:
        if p not in ordered:
            ordered[p] = results_by_policy[p]

    # plot_lambda2_traces expects names matching POLICY_COLORS. k1_gs isn't
    # there; we patch the function call by remapping the dict to use a name
    # the theme recognizes, then overlaying the k1_gs color manually after.
    # Simpler: just remap k1_gs -> a name that we can color via fall-through.
    # The function falls back to near_black for unknown policies, which we
    # explicitly want to override -- so we wrap it.
    fig, ax = _plot_fig1_curves(ordered, alpha_0, orbital_period_s)

    out_path = out_dir / "fig1_lambda2_traces.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return True


def _plot_fig1_curves(
    results_by_policy: dict[str, list[SimulationResult]],
    alpha_0: float,
    orbital_period_s: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Custom Fig 1 plotter that knows about k1_gs in addition to the four
    standard policies. Mirrors plot_lambda2_traces but assigns a color to
    k1_gs explicitly."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT))

    # Pull standard timing from any result.
    first = next(iter(results_by_policy.values()))[0]
    dt_s = first.dt_s
    T = first.T
    n_epochs = first.n_epochs
    x = np.arange(n_epochs) * dt_s / orbital_period_s
    x_warmup_end = T * dt_s / orbital_period_s

    ax.axvspan(0, x_warmup_end, color=COLORS.parchment, alpha=0.35, zorder=0,
               label=r"warmup ($t < T$)")

    # Style table: policy_label -> (color, linestyle).
    style_table = {
        "predictive":  (POLICY_COLORS["predictive"],  "-"),
        "k1_gs":       (COLORS.copper,                "-"),
        "greedy":      (POLICY_COLORS["greedy"],      "--"),
        "random":      (POLICY_COLORS["random"],      ":"),
    }
    pretty = {
        "predictive": "Predictive (sync)",
        "k1_gs":      "Predictive (Gauss-Seidel)",
        "greedy":     "Greedy",
        "random":     "Random",
    }

    for policy_label, results in results_by_policy.items():
        color, ls = style_table.get(policy_label, (COLORS.near_black, "-"))
        stacked = np.stack([r.lambda2_union for r in results], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0) if len(results) > 1 else None
        ax.plot(x, mean, label=pretty.get(policy_label, policy_label),
                color=color, linestyle=ls, zorder=3)
        if std is not None:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15,
                            zorder=2, linewidth=0)

    ax.axhline(alpha_0, color=COLORS.forest, linestyle=":", linewidth=0.8, zorder=4,
               label=rf"$\alpha_0 = {alpha_0:.2f}$")

    ax.set_xlabel("Time (orbital periods)")
    ax.set_ylabel(r"$\lambda_2(L^\cup_{\mathcal{G}}(t; T))$")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=7, ncol=1)
    return fig, ax


# ---------------------------------------------------------------------------
# Fig 3: update-order ablation
# ---------------------------------------------------------------------------


def render_fig3(out_dir: Path) -> bool:
    """Build Fig 3 from results/canonical/fig3/.

    Two-panel layout:
      Left: headline comparison (k1_sync, k1_gs identity, adaptive, equilibrium)
            showing requests/ep, edges/ep, waste% as grouped bars.
      Right: edges-per-epoch time series for the same four policies over
             the full simulation, with a smoothed mean overlay.
    """
    data_dir = CANONICAL_ROOT / "fig3"
    if not data_dir.exists():
        log.warning("Fig 3: no data at %s; skipping. Run stage_canonical_traces.", data_dir)
        return False

    headline_labels = ["k1_sync", "k1_gs_identity", "adaptive", "equilibrium"]

    headline_data = {}    # label -> summary dict
    headline_traces = {}  # label -> SimulationResult (for time series)
    for label in headline_labels:
        path = data_dir / f"{label}_seed42.npz"
        if not path.exists():
            log.warning("Fig 3: missing headline trace %s", path.name)
            continue
        result, report = load_simulation_result(path)
        headline_data[label] = _summarize(result, report)
        headline_traces[label] = result

    if not headline_data:
        log.warning("Fig 3: no headline data; skipping.")
        return False

    print(f"Fig 3: loaded {len(headline_data)} headline traces")

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_DOUBLE_COL, FIG_HEIGHT_DEFAULT * 1.4),
        gridspec_kw={"width_ratios": [1.1, 1.3]},
    )

    # ---- Left panel: headline summary bars ----
    _plot_headline_panel(ax_left, headline_data)
    # ---- Right panel: edges/ep time series ----
    _plot_edge_timeseries_panel(ax_right, headline_traces)

    fig.tight_layout()
    out_path = out_dir / "fig3_update_order.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return True


def _summarize(result: SimulationResult, report: DiagnosticsReport) -> dict[str, float]:
    """Per-epoch averages used by Fig 3."""
    requests = (result.actions != -1).sum(axis=1).mean()
    edges = result.n_edges_per_epoch.mean()
    waste = requests - 2 * edges
    return {
        "requests_per_epoch": float(requests),
        "edges_per_epoch": float(edges),
        "waste_per_epoch": float(waste),
        "waste_pct": 100.0 * float(waste) / max(float(requests), 1e-9),
        "rho_cover": float(report.rho_cover_final),
        "rho_mean": float(report.rho_realized_mean),
    }


def _plot_headline_panel(ax: plt.Axes, headline_data: dict[str, dict]) -> None:
    """Grouped bars: requests/ep, edges/ep, waste% for four policies.

    Three metrics are plotted as side-by-side groups along the x-axis.
    Each group has one bar per policy.
    """
    labels = list(headline_data.keys())
    pretty = {
        "k1_sync":         "Predictive\n(sync)",
        "k1_gs_identity":  "Predictive\n(GS)",
        "adaptive":        "Adaptive",
        "equilibrium":     "Equilibrium",
    }
    colors_table = {
        "k1_sync":         POLICY_COLORS["predictive"],
        "k1_gs_identity":  COLORS.copper,
        "adaptive":        COLORS.warmbrown,
        "equilibrium":     COLORS.olive,
    }

    metrics = ["requests_per_epoch", "edges_per_epoch", "waste_pct"]
    metric_names = ["Requests / ep", "Edges / ep", "Waste %"]

    n_policies = len(labels)
    n_metrics = len(metrics)
    group_width = 0.8
    bar_width = group_width / n_policies
    x_groups = np.arange(n_metrics)

    for j, label in enumerate(labels):
        values = [headline_data[label][m] for m in metrics]
        positions = x_groups + (j - n_policies / 2 + 0.5) * bar_width
        ax.bar(
            positions, values, width=bar_width,
            color=colors_table.get(label, COLORS.near_black),
            edgecolor=COLORS.near_black, linewidth=0.4,
            label=pretty.get(label, label),
        )

    ax.set_xticks(x_groups)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Value (count or %)")
    ax.set_title("Update order: synchronous vs Gauss-Seidel", fontsize=9)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        fontsize=7, ncol=4, frameon=False,
        handletextpad=0.4, columnspacing=1.0,
    )
    ax.set_ylim(bottom=0)

    if "k1_gs_identity" in headline_data:
        edges = headline_data["k1_gs_identity"]["edges_per_epoch"]
        ax.annotate(
            f"{edges:.1f}",
            xy=(1 + (1 - n_policies / 2 + 0.5) * bar_width, edges),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=7, color=COLORS.copper,
        )


def _plot_edge_timeseries_panel(ax: plt.Axes, traces: dict[str, SimulationResult]) -> None:
    """Edges per epoch over time, smoothed with a rolling mean.

    Plots the raw trace as a thin transparent line and the smoothed mean
    (rolling window of 20 epochs) as the foreground curve. This makes the
    steady-state separation between the four policies visible despite the
    integer-valued per-epoch noise.
    """
    if not traces:
        ax.text(0.5, 0.5, "No time-series data", ha="center", va="center",
                transform=ax.transAxes, color=COLORS.warm_gray)
        return

    pretty = {
        "k1_sync":         "Predictive (sync)",
        "k1_gs_identity":  "Predictive (GS)",
        "adaptive":        "Adaptive",
        "equilibrium":     "Equilibrium",
    }
    colors_table = {
        "k1_sync":         POLICY_COLORS["predictive"],
        "k1_gs_identity":  COLORS.copper,
        "adaptive":        COLORS.warmbrown,
        "equilibrium":     COLORS.olive,
    }

    # Use orbital periods on x-axis if any trace has T*dt > 0.
    first = next(iter(traces.values()))
    orbital_period_s = first.T * first.dt_s
    use_periods = orbital_period_s > 0
    if use_periods:
        x_base = np.arange(first.n_epochs) * first.dt_s / orbital_period_s
        x_label = "Time (orbital periods)"
        warmup_end = first.T * first.dt_s / orbital_period_s
    else:
        x_base = np.arange(first.n_epochs)
        x_label = "Epoch $t$"
        warmup_end = float(first.T)

    ax.axvspan(0, warmup_end, color=COLORS.parchment, alpha=0.35, zorder=0)

    window = 20  # rolling mean window in epochs
    final_positions: list[tuple[str, float, str]] = []  # (label, y_final, color)

    for label, result in traces.items():
        edges = result.n_edges_per_epoch.astype(np.float64)
        color = colors_table.get(label, COLORS.near_black)

        # Raw trace: thin and transparent.
        ax.plot(x_base, edges, color=color, linewidth=0.4, alpha=0.25, zorder=2)

        # Smoothed trace: rolling mean.
        if len(edges) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(edges, kernel, mode="same")
            ax.plot(x_base, smoothed, color=color, linewidth=1.2, zorder=3)
            # Average over the last window epochs for a stable label position.
            y_final = float(np.mean(smoothed[-window:]))
        else:
            ax.plot(x_base, edges, color=color, linewidth=1.2, zorder=3)
            y_final = float(edges[-1])

        final_positions.append((pretty.get(label, label), y_final, color))

    ax.set_xlabel(x_label)
    ax.set_ylabel("Edges per epoch")
    ax.set_title("Realized matching density over time", fontsize=9)
    ax.set_ylim(bottom=0)

    # Extend the x-axis slightly past the last data point so inline labels
    # have room to the right of the curves without being clipped.
    x_end = float(x_base[-1])
    x_pad = (x_end - float(x_base[0])) * 0.20
    ax.set_xlim(x_base[0], x_end + x_pad)

    # Inline labels at the right edge of each curve. Sort by y so we can
    # nudge any overlapping labels apart.
    final_positions.sort(key=lambda t: t[1])
    min_gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06  # ~6% of axis range
    last_y = -np.inf
    for label, y, color in final_positions:
        y_text = max(y, last_y + min_gap)
        ax.annotate(
            label,
            xy=(x_end, y), xytext=(x_end + x_pad * 0.10, y_text),
            color=color, fontsize=7, va="center", ha="left",
        )
        last_y = y_text

    # Warmup label placed where the data is, not in the bottom-right.
    ax.text(
        warmup_end / 2, ax.get_ylim()[1] * 0.04,
        "warmup", color=COLORS.warm_gray, fontsize=7, ha="center", va="bottom",
    )




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", type=str, default=None,
        help="Render only one figure: 'fig1' or 'fig3'.",
    )
    args = parser.parse_args()

    configure(level="WARNING")
    apply_theme(context="paper")

    out_dir = figures_dir("paper")
    print(f"Output directory: {out_dir}\n")

    rendered = 0
    figures = [
        ("fig1", render_fig1),
        ("fig3", render_fig3),
    ]
    for name, fn in figures:
        if args.only is not None and args.only != name:
            continue
        print(f"--- {name} ---")
        if fn(out_dir):
            rendered += 1
        print()

    print(f"Rendered {rendered} figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
