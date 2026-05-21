# orbit-match/scripts/render_paper_figures.py
# Run: python -m scripts.render_paper_figures

"""Read saved traces from results/ and render paper figures.

Pure consumer of the run_*.py scripts' output. Does not run any
simulations; only loads .npz files and calls the plotting functions.

Current figures
---------------
- Fig 1: lambda_2 traces. Inputs come from results/lambda2_traces/.
- Fig 2, 3, 4: stubbed in. They will populate once the corresponding
  run_*.py scripts (run_horizon_ablation, run_scaling, run_robustness)
  have written their traces.

Output
------
PDFs in figures/paper/. Filenames follow PROJECT_PLAN.md Sec 4.2:
fig1_lambda2_traces.pdf, fig2_horizon_ablation.pdf, fig3_scaling.pdf,
fig4_robustness.pdf.

Behavior on missing data
------------------------
If a results directory is empty or missing for a particular figure,
the script logs a warning and skips that figure. It does not fail.
This way it can be run incrementally as the run_*.py scripts produce
data.
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
from orbitmatch.plotting.theme import apply_theme
from orbitmatch.utils.io import RESULTS_ROOT, figures_dir, load_trace
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Trace -> SimulationResult reconstruction
# ---------------------------------------------------------------------------


def load_simulation_result(path: Path) -> tuple[SimulationResult, DiagnosticsReport]:
    """Reconstruct a SimulationResult and its DiagnosticsReport from disk.

    The trace file contains arrays plus a manifest with the report
    under ``manifest["user"]["report"]`` and the original metadata
    under ``manifest["user"]``. Both are needed to rebuild the
    SimulationResult dataclass.
    """
    arrays, manifest = load_trace(path)
    user = manifest.get("user", {})

    # Reconstruct the report.
    rep_dict = user.get("report")
    if rep_dict is None:
        raise ValueError(f"Trace {path} has no 'report' in manifest")
    report = DiagnosticsReport(**rep_dict)

    # Recover the policy diagnostics (stored as "diag_<name>" keys).
    diag: dict[str, np.ndarray] = {}
    for k in list(arrays.keys()):
        if k.startswith("diag_"):
            diag[k[len("diag_"):]] = arrays.pop(k)

    # Matchings come back as a (n_epochs,) object-dtype array of (k_t, 2) arrays.
    matchings_obj = arrays.pop("matchings")
    matchings = tuple(np.asarray(m, dtype=np.int64) for m in matchings_obj)

    result = SimulationResult(
        policy_name=user["policy_name"],
        config_label=user["config_label"],
        n=int(user["walker"]["M"]),
        n_epochs=int(user["n_epochs"]),
        T=int(user["policy_params"]["T"]),
        dt_s=float(user["dt_s"]),
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


def load_sweep_record_from_trace(path: Path) -> SweepRecord:
    """Reconstruct a SweepRecord from a sweep-produced trace file."""
    _, manifest = load_trace(path)
    user = manifest.get("user", {})
    rep_dict = user.get("report")
    if rep_dict is None:
        raise ValueError(f"Trace {path} has no 'report' in manifest")
    report = DiagnosticsReport(**rep_dict)
    override_label = user.get("override_label", "?")
    # Recover override_key from the filename: trace_..._{key}_seed{seed}.npz
    # The override key is the segment between the policy name and seed.
    return SweepRecord(
        override_label=override_label,
        override_key=_override_key_from_filename(path.name),
        seed=int(user["seed"]),
        config_label=user["config_label"],
        policy_name=user["policy_name"],
        trace_path=path,
        report=report,
        skipped_existing=True,
    )


def _override_key_from_filename(fname: str) -> str:
    """Extract the override_key segment from a sweep trace filename.

    Expected pattern: trace_{config}_{policy}_{override_key}_seed{N}.npz
    """
    m = re.match(r"^trace_.+?_.+?_(.+)_seed\d+\.npz$", fname)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Fig 1: lambda_2 traces
# ---------------------------------------------------------------------------


def render_fig1(out_dir: Path, *, results_subdir: str = "lambda2_traces") -> bool:
    """Build Fig 1 from the saved lambda2_traces directory.

    Returns True on success, False if no data found.
    """
    data_dir = RESULTS_ROOT / results_subdir
    if not data_dir.exists():
        log.warning("Fig 1: no data at %s; skipping.", data_dir)
        return False

    traces = sorted(data_dir.glob("trace_*.npz"))
    if not traces:
        log.warning("Fig 1: no .npz files in %s; skipping.", data_dir)
        return False

    # Group by policy.
    results_by_policy: dict[str, list[SimulationResult]] = {}
    reports_by_policy: dict[str, list[DiagnosticsReport]] = {}
    for path in traces:
        result, report = load_simulation_result(path)
        results_by_policy.setdefault(result.policy_name, []).append(result)
        reports_by_policy.setdefault(result.policy_name, []).append(report)

    # alpha_0 is shared (constellation-only); take from any report.
    first_report = next(iter(reports_by_policy.values()))[0]
    alpha_0 = first_report.alpha_0

    # Pull orbital period from the metadata of any result.
    first_result = next(iter(results_by_policy.values()))[0]
    walker_meta = first_result.metadata.get("walker", {})
    altitude_km = float(walker_meta.get("altitude_km", 550.0))
    # T_orb_s = 2*pi*sqrt(a^3/mu), but we have it indirectly via T = T_orb in epochs
    # for these runs. Derive it from T * dt_s.
    orbital_period_s = first_result.T * first_result.dt_s

    print(f"Fig 1: loaded {len(traces)} traces, {len(results_by_policy)} policies "
          f"({', '.join(sorted(results_by_policy))})")
    for policy_name, results in results_by_policy.items():
        n_seeds = len(results)
        reports = reports_by_policy[policy_name]
        rho_max = float(np.mean([r.rho_realized_max for r in reports]))
        rho_mean = float(np.mean([r.rho_realized_mean for r in reports]))
        print(f"  {policy_name:<14} ({n_seeds} seeds): rho_max={rho_max:.4f}, rho_mean={rho_mean:.4f}")

    # Order policies for consistent legend: predictive, equilibrium, greedy, random.
    canonical_order = ["predictive", "equilibrium", "greedy", "random"]
    ordered = {
        p: results_by_policy[p]
        for p in canonical_order if p in results_by_policy
    }
    # Any other policies appended at the end.
    for p in results_by_policy:
        if p not in ordered:
            ordered[p] = results_by_policy[p]

    fig, ax = plot_lambda2_traces(
        ordered,
        alpha_0=alpha_0,
        rho_bound=None,  # the theorem bound rho*alpha_0 is intentionally omitted;
                        # rho_match is essentially 1 in our regime (Vizing-vacuous),
                        # and rho_realized empirically saturates at 1.
        x_in_periods=True,
        orbital_period_s=orbital_period_s,
        show_warmup=True,
    )

    out_path = out_dir / "fig1_lambda2_traces.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Fig 2: horizon ablation
# ---------------------------------------------------------------------------


def render_fig2(out_dir: Path, *, results_subdir: str = "horizon_ablation") -> bool:
    data_dir = RESULTS_ROOT / results_subdir
    if not data_dir.exists():
        log.warning("Fig 2: no data at %s; skipping.", data_dir)
        return False

    traces = sorted(data_dir.glob("trace_*.npz"))
    if not traces:
        log.warning("Fig 2: no .npz files in %s; skipping.", data_dir)
        return False

    records = [load_sweep_record_from_trace(p) for p in traces]
    print(f"Fig 2: loaded {len(records)} records")

    fig, ax = plot_horizon_ablation(records, metric="rho_realized_mean")
    out_path = out_dir / "fig2_horizon_ablation.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Fig 3: scaling
# ---------------------------------------------------------------------------


def render_fig3(out_dir: Path, *, results_subdir: str = "scaling") -> bool:
    data_dir = RESULTS_ROOT / results_subdir
    if not data_dir.exists():
        log.warning("Fig 3: no data at %s; skipping.", data_dir)
        return False

    traces = sorted(data_dir.glob("trace_*.npz"))
    if not traces:
        log.warning("Fig 3: no .npz files in %s; skipping.", data_dir)
        return False

    records_by_policy: dict[str, list[SweepRecord]] = {}
    for path in traces:
        rec = load_sweep_record_from_trace(path)
        records_by_policy.setdefault(rec.policy_name, []).append(rec)

    print(f"Fig 3: loaded {sum(len(v) for v in records_by_policy.values())} records, "
          f"{len(records_by_policy)} policies")

    fig, ax = plot_scaling(records_by_policy, metric="rho_realized_mean")
    out_path = out_dir / "fig3_scaling.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", type=str, default=None,
        help="Render only one figure: 'fig1', 'fig2', or 'fig3'.",
    )
    args = parser.parse_args()

    configure(level="WARNING")
    apply_theme(context="paper")

    out_dir = figures_dir("paper")
    print(f"Output directory: {out_dir}\n")

    rendered = 0
    figures = [
        ("fig1", render_fig1),
        ("fig2", render_fig2),
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