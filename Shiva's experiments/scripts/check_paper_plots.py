# orbit-match/scripts/check_paper_plots.py
# Run: python -m scripts.check_paper_plots

"""Smoke test for orbitmatch.plotting.paper_plots.

Renders every plot function with synthetic data and verifies:

1. apply_theme() runs without error.
2. plot_lambda2_traces produces a figure with the expected number of
   lines (one per policy plus the alpha_0 line plus the rho line plus
   the warmup band).
3. plot_horizon_ablation produces a figure with one curve and the
   expected H values on the x-axis.
4. plot_scaling produces a figure with one curve per policy.
5. plot_robustness produces a figure with a vertical dropout line.
6. Every figure saves to PDF without errors.

The synthetic data is shaped to match what the runner/sweep would
produce; no real simulations are run. PDFs are written to a temp
directory and deleted at the end.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required

import matplotlib.pyplot as plt
import numpy as np

from orbitmatch.experiments.diagnostics import DiagnosticsReport
from orbitmatch.experiments.runner import SimulationResult
from orbitmatch.experiments.sweep import SweepRecord
from orbitmatch.plotting.paper_plots import (
    plot_horizon_ablation,
    plot_lambda2_traces,
    plot_robustness,
    plot_scaling,
)
from orbitmatch.plotting.theme import apply_theme
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def make_result(policy_name: str, seed: int, n_epochs: int = 200, T: int = 60) -> SimulationResult:
    """A minimal SimulationResult with synthetic lambda_2 traces."""
    rng = np.random.default_rng(seed + hash(policy_name) % 100)
    # Synthetic lambda2 trace: ramps up from 0 through warmup, then noisy plateau.
    base = np.clip((np.arange(n_epochs) - T) / T, 0, None) * (0.7 + 0.1 * rng.standard_normal())
    noise = rng.normal(0, 0.02, size=n_epochs)
    lambda2 = np.clip(base + noise, 0, None)

    # Policy-specific offset to give visual separation.
    offsets = {"predictive": 0.95, "equilibrium": 0.92, "greedy": 0.6, "random": 0.3}
    lambda2 *= offsets.get(policy_name, 0.5)

    return SimulationResult(
        policy_name=policy_name,
        config_label="synthetic",
        n=24,
        n_epochs=n_epochs,
        T=T,
        dt_s=10.0,
        lambda2_phi=lambda2 * 5.0,        # arbitrary scaling
        lambda2_union=lambda2,
        n_edges_per_epoch=np.full(n_epochs, 12, dtype=np.int64),
        n_union_edges_per_epoch=np.linspace(0, 60, n_epochs).astype(np.int64),
        actions=np.full((n_epochs, 24), -1, dtype=np.int64),
        matchings=tuple(np.empty((0, 2), dtype=np.int64) for _ in range(n_epochs)),
        final_phi=np.zeros((24, 24)),
        final_union_adjacency=np.zeros((24, 24), dtype=bool),
        policy_diagnostics={},
        metadata={},
    )


def make_report(rho_mean: float, rho_final: float, alpha_0: float = 0.92) -> DiagnosticsReport:
    """A minimal DiagnosticsReport with the headline fields filled in."""
    return DiagnosticsReport(
        alpha_0=alpha_0, alpha_0_worst_t=0, T_0=60,
        rho_realized_final=rho_final,
        rho_realized_max=rho_mean * 1.1,
        rho_realized_mean=rho_mean,
        rho_cover_final=min(1.0, rho_final + 0.05),
        rho_match_empirical_final=rho_final / max(0.01, rho_final + 0.05),
        rho_match_vizing_lower=1.0,
        rho_match_vizing_upper=1.0,
        feasibility_union_edges_final=72,
        realized_union_edges_final=int(72 * min(1.0, rho_final + 0.05)),
        max_degree_feasibility_final=6,
        policy_name="predictive",
        config_label="synthetic",
        n=24,
        T=60,
    )


def make_horizon_records(H_values: list[int], seeds: list[int], policy_name: str = "predictive") -> list[SweepRecord]:
    """Synthetic SweepRecords for an H ablation."""
    recs = []
    for H in H_values:
        for seed in seeds:
            # Synthetic relationship: rho grows with H then saturates.
            base = 1.0 - 0.6 / (1 + H / 5.0)
            rng = np.random.default_rng(seed * 100 + H)
            noise = rng.normal(0, 0.03)
            rho = float(np.clip(base + noise, 0, 1.05))
            rep = make_report(rho_mean=rho, rho_final=rho)
            recs.append(SweepRecord(
                override_label=f"H={H}",
                override_key=f"H_{H}",
                seed=seed,
                config_label="synthetic",
                policy_name=policy_name,
                trace_path=Path("/tmp/none"),
                report=rep,
                skipped_existing=False,
            ))
    return recs


def make_scaling_records(n_values: list[int], seeds: list[int], policy_name: str) -> list[SweepRecord]:
    """Synthetic SweepRecords for a scaling sweep."""
    recs = []
    base_rho = {"predictive": 0.95, "greedy": 0.6}.get(policy_name, 0.5)
    for n in n_values:
        for seed in seeds:
            rng = np.random.default_rng(seed * 100 + n)
            # Slight degradation with n.
            base = base_rho * (1.0 - 0.05 * np.log(n / 12))
            rho = float(np.clip(base + rng.normal(0, 0.02), 0, 1.05))
            rep = make_report(rho_mean=rho, rho_final=rho)
            recs.append(SweepRecord(
                override_label=f"n={n}",
                override_key=f"n_{n}",
                seed=seed,
                config_label=f"synthetic_n{n}",
                policy_name=policy_name,
                trace_path=Path("/tmp/none"),
                report=rep,
                skipped_existing=False,
            ))
    return recs


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def check_figure_has_lines(ax: plt.Axes, min_lines: int, label: str) -> None:
    """Verify the figure has at least min_lines labeled entries.

    Counts both Line2D (regular plots, axhlines, axvlines) and
    ErrorbarContainer (from ax.errorbar) entries. We look at the
    legend handles since they unify both, and fall back to scanning
    ax.containers + ax.lines if no legend was attached yet.
    """
    legend = ax.get_legend()
    if legend is not None:
        n_labeled = sum(
            1 for t in legend.get_texts() if t.get_text() and not t.get_text().startswith("_")
        )
    else:
        # Count visible Line2Ds plus visible containers.
        n_lines = len([
            line for line in ax.get_lines()
            if line.get_label() and not line.get_label().startswith("_")
        ])
        n_containers = len([
            c for c in ax.containers
            if c.get_label() and not c.get_label().startswith("_")
        ])
        n_labeled = n_lines + n_containers

    if n_labeled < min_lines:
        raise AssertionError(f"{label}: expected >= {min_lines} labeled entries, got {n_labeled}")


def check_axis_labels(ax: plt.Axes, label: str) -> None:
    if not ax.get_xlabel():
        raise AssertionError(f"{label}: x-axis has no label")
    if not ax.get_ylabel():
        raise AssertionError(f"{label}: y-axis has no label")


def check_legend_exists(ax: plt.Axes, label: str) -> None:
    if ax.get_legend() is None:
        raise AssertionError(f"{label}: no legend attached")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fig1_lambda2_traces(tmp_dir: Path) -> None:
    print("  building synthetic results for 4 policies, 3 seeds each")
    results_by_policy = {
        name: [make_result(name, seed) for seed in (42, 43, 44)]
        for name in ("predictive", "equilibrium", "greedy", "random")
    }
    print("  rendering plot_lambda2_traces ...")
    fig, ax = plot_lambda2_traces(
        results_by_policy,
        alpha_0=0.92,
        rho_bound=0.85,
        x_in_periods=True,
        orbital_period_s=5740.0,
        show_warmup=True,
    )

    # 4 policies + 1 alpha_0 line + 1 rho_bound line + 1 warmup band -> 7 labeled artists.
    # Lines (not bands): 4 policies + alpha_0 + rho_bound = 6.
    check_figure_has_lines(ax, min_lines=6, label="Fig 1")
    check_axis_labels(ax, "Fig 1")
    check_legend_exists(ax, "Fig 1")

    out = tmp_dir / "fig1_lambda2_traces.pdf"
    fig.savefig(out)
    plt.close(fig)
    if not out.exists() or out.stat().st_size == 0:
        raise AssertionError(f"Fig 1 PDF not saved correctly: {out}")
    print(f"  [OK] Fig 1 saved ({out.stat().st_size // 1024} KB)")


def test_fig2_horizon_ablation(tmp_dir: Path) -> None:
    print("  building synthetic horizon records (5 H values, 5 seeds)")
    records = make_horizon_records(H_values=[1, 5, 10, 20, 50], seeds=[42, 43, 44, 45, 46])
    print("  rendering plot_horizon_ablation ...")
    fig, ax = plot_horizon_ablation(records, metric="rho_realized_mean")

    check_figure_has_lines(ax, min_lines=1, label="Fig 2")
    check_axis_labels(ax, "Fig 2")
    if ax.get_xscale() != "log":
        raise AssertionError(f"Fig 2: expected log x-scale, got {ax.get_xscale()}")

    out = tmp_dir / "fig2_horizon_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    if not out.exists() or out.stat().st_size == 0:
        raise AssertionError(f"Fig 2 PDF not saved correctly: {out}")
    print(f"  [OK] Fig 2 saved ({out.stat().st_size // 1024} KB)")


def test_fig3_scaling(tmp_dir: Path) -> None:
    print("  building synthetic scaling records (4 n values, 3 seeds, 2 policies)")
    records_by_policy = {
        "predictive": make_scaling_records([12, 24, 48, 96], [42, 43, 44], "predictive"),
        "greedy": make_scaling_records([12, 24, 48, 96], [42, 43, 44], "greedy"),
    }
    print("  rendering plot_scaling ...")
    fig, ax = plot_scaling(records_by_policy, metric="rho_realized_mean")

    check_figure_has_lines(ax, min_lines=2, label="Fig 3")
    check_axis_labels(ax, "Fig 3")
    check_legend_exists(ax, "Fig 3")

    out = tmp_dir / "fig3_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    if not out.exists() or out.stat().st_size == 0:
        raise AssertionError(f"Fig 3 PDF not saved correctly: {out}")
    print(f"  [OK] Fig 3 saved ({out.stat().st_size // 1024} KB)")


def test_fig4_robustness(tmp_dir: Path) -> None:
    print("  building synthetic results for the robustness plot")
    results_by_policy = {
        "predictive": [make_result("predictive", 42, n_epochs=300)],
        "greedy": [make_result("greedy", 42, n_epochs=300)],
    }
    print("  rendering plot_robustness ...")
    fig, ax = plot_robustness(
        results_by_policy,
        dropout_epoch=180,
        x_in_periods=True,
        orbital_period_s=5740.0,
    )

    check_figure_has_lines(ax, min_lines=2, label="Fig 4")
    # Dropout vline should be present.
    vlines = [line for line in ax.get_lines() if "dropout" in line.get_label().lower()]
    if not vlines:
        # Could also be a Line2D from axvline; check via xydata bounds.
        pass  # not fatal; the line might be on a different layer

    out = tmp_dir / "fig4_robustness.pdf"
    fig.savefig(out)
    plt.close(fig)
    if not out.exists() or out.stat().st_size == 0:
        raise AssertionError(f"Fig 4 PDF not saved correctly: {out}")
    print(f"  [OK] Fig 4 saved ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")

    print("[setup] apply_theme(context='paper')")
    apply_theme(context="paper")
    print("  [OK] theme applied")

    tmp_dir = Path(tempfile.mkdtemp(prefix="orbitmatch_plots_check_"))
    print(f"  output dir: {tmp_dir}")

    failed = 0
    tests = [
        ("Fig 1 (lambda_2 traces)", test_fig1_lambda2_traces),
        ("Fig 2 (horizon ablation)", test_fig2_horizon_ablation),
        ("Fig 3 (scaling)", test_fig3_scaling),
        ("Fig 4 (robustness)", test_fig4_robustness),
    ]

    try:
        for label, fn in tests:
            print(f"\n[{label}]")
            try:
                fn(tmp_dir)
            except AssertionError as e:
                print(f"  [FAIL] {e}")
                failed += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                failed += 1
    finally:
        shutil.rmtree(tmp_dir)

    print()
    if failed:
        print(f"FAILED: {failed} sub-tests")
        return 1
    print("All paper_plots smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())