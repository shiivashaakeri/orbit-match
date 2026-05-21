# orbit-match/scripts/run_lambda2_traces.py
# Run: python -m scripts.run_lambda2_traces [--force]

"""Headline experiment: produce Fig 1 data.

Runs all four policies (predictive, greedy, random, equilibrium) on
the medium Walker constellation, with the certificate window set to
one orbital period (T = T_orb). One simulation per (policy, seed)
combination; the per-policy traces are saved as separate .npz files
so the rendering script can load them independently.

Configuration is read from configs/experiment_main.yaml. Output traces
live in results/lambda2_traces/.

Per PROJECT_PLAN.md Sec 5.2 ("never re-run a simulation that already
has a saved trace"), this script checks for an existing trace file
before running each (policy, seed) job. Use --force to override.

Wall-clock estimate (medium constellation, n=60, 3 periods, 3 seeds,
4 policies = 12 jobs): roughly 30-60 minutes on a laptop. The medium
config feasibility tensor is cached after the first run, so re-runs
of individual jobs are faster than the initial cold run.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
from orbitmatch.experiments.diagnostics import build_report, compute_alpha_0, DiagnosticsReport
from orbitmatch.experiments.runner import SimulationResult, run_simulation
from orbitmatch.feasibility.compute import compute_feasibility
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.policy.base import PolicyParams
from orbitmatch.utils.io import PROJECT_ROOT, load_trace, results_dir, save_trace
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENT_NAME = "lambda2_traces"


# ---------------------------------------------------------------------------
# YAML loading (same conventions as scripts/load_configs.py)
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping; got {type(data).__name__}")
    return data


def build_walker(cfg: dict[str, Any]) -> WalkerDeltaConfig:
    return WalkerDeltaConfig(
        M=int(cfg["M"]),
        P=int(cfg["P"]),
        F=int(cfg["F"]),
        altitude_km=float(cfg["altitude_km"]),
        inclination_deg=float(cfg["inclination_deg"]),
        name=str(cfg.get("name", "")),
    )


def build_feasibility_params(cfg: dict[str, Any]) -> FeasibilityParams:
    if "rate_max_deg_per_s" in cfg:
        rate_rad = float(np.deg2rad(cfg["rate_max_deg_per_s"]))
    elif "rate_max_rad_per_s" in cfg:
        rate_rad = float(cfg["rate_max_rad_per_s"])
    else:
        rate_rad = float(np.deg2rad(1.0))
    return FeasibilityParams(
        atm_buffer_km=float(cfg.get("atm_buffer_km", 80.0)),
        range_max_km=float(cfg.get("range_max_km", 8000.0)),
        rate_max_rad_per_s=rate_rad,
    )


def resolve_T(T_spec: Any, walker: WalkerDeltaConfig, dt_s: float) -> int:
    if isinstance(T_spec, int):
        return T_spec
    if isinstance(T_spec, str) and T_spec == "orbital_period":
        return int(math.ceil(walker.orbital_period_s / dt_s))
    raise ValueError(f"Invalid T spec: {T_spec!r}")


def build_policy_params(cfg: dict[str, Any], T: int) -> PolicyParams:
    return PolicyParams(
        H=int(cfg["H"]),
        T=T,
        switching_cost_scale=float(cfg["switching_cost_scale"]),
        epsilon_geometric_prior=float(cfg["epsilon_geometric_prior"]),
        tie_break=str(cfg["tie_break"]),
        seed=cfg.get("seed"),
    )


# ---------------------------------------------------------------------------
# Trace filename (PROJECT_PLAN.md Sec 5.3 convention)
# ---------------------------------------------------------------------------


def trace_filename(config_label: str, policy_name: str, T: int, H: int, seed: int) -> str:
    return f"trace_{config_label}_{policy_name}_T{T}_H{H}_seed{seed}.npz"


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


def run_job(
    *,
    walker: WalkerDeltaConfig,
    feasibility_params: FeasibilityParams,
    policy_name: str,
    policy_params: PolicyParams,
    n_epochs: int,
    dt_s: float,
    seed: int,
    config_label: str,
    out_path: Path,
    precomputed_alpha_0: tuple[float, int] | None,
) -> tuple[SimulationResult, DiagnosticsReport]:
    """Run one (policy, seed) job and save the trace to out_path."""
    result = run_simulation(
        walker=walker,
        feasibility_params=feasibility_params,
        policy_name=policy_name,
        policy_params=policy_params,
        n_epochs=n_epochs,
        dt_s=dt_s,
        seed=seed,
        config_label=config_label,
    )

    # Build the diagnostics report. The feasibility tensor is loaded
    # again here (cached on disk, so the second call is cheap), and
    # alpha_0 is shared across policies for the same constellation.
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * dt_s, dt_s=dt_s)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, dt_s, feasibility_params)

    report = build_report(
        result, feasibility,
        T_0=policy_params.T,
        precomputed_alpha_0=precomputed_alpha_0,
    )

    # Save the trace with the report folded into metadata.
    metadata = dict(result.metadata)
    metadata["report"] = report.to_metadata()
    metadata["experiment"] = EXPERIMENT_NAME
    save_trace(out_path, arrays=result.to_arrays(), metadata=metadata)

    return result, report


def load_existing_report(out_path: Path) -> DiagnosticsReport:
    """Reconstruct a DiagnosticsReport from a cached trace's manifest."""
    _, manifest = load_trace(out_path)
    rep_dict = manifest.get("user", {}).get("report")
    if rep_dict is None:
        raise ValueError(f"Trace {out_path} has no 'report' in its manifest")
    return DiagnosticsReport(**rep_dict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run every job even if a trace already exists on disk.",
    )
    parser.add_argument(
        "--experiment-yaml", type=Path, default=CONFIGS_DIR / "experiment_main.yaml",
        help="Path to the experiment YAML.",
    )
    args = parser.parse_args()

    configure(level="WARNING")

    # ---- Load and resolve config -------------------------------------------
    print(f"Loading experiment: {args.experiment_yaml.relative_to(PROJECT_ROOT)}")
    exp_cfg = load_yaml(args.experiment_yaml)

    constellation_cfg = load_yaml(CONFIGS_DIR / exp_cfg["constellation"])
    walker = build_walker(constellation_cfg["walker"])
    feas_params = build_feasibility_params(constellation_cfg["feasibility"])
    config_label = constellation_cfg.get("label", "unknown")

    policy_cfg = load_yaml(CONFIGS_DIR / exp_cfg["policy"])
    policies = list(exp_cfg["policies"])

    sim = exp_cfg["simulation"]
    dt_s = float(sim["dt_s"])
    duration_periods = float(sim["duration_periods"])
    n_epochs = int(math.ceil(duration_periods * walker.orbital_period_s / dt_s))
    T = resolve_T(sim["T"], walker, dt_s)
    seeds = list(sim["seeds"])

    policy_params = build_policy_params(policy_cfg["policy"], T)

    out_dir = results_dir(EXPERIMENT_NAME)

    print(f"  constellation: {config_label} (n={walker.M}, P={walker.P}, F={walker.F})")
    print(f"  policies:      {policies}")
    print(f"  n_epochs:      {n_epochs}  ({duration_periods} periods at dt={dt_s}s)")
    print(f"  T:             {T} epochs  ({T*dt_s/60:.2f} min)")
    print(f"  H:             {policy_params.H}")
    print(f"  seeds:         {seeds}")
    print(f"  total jobs:    {len(policies) * len(seeds)}")
    print(f"  output dir:    {out_dir}")
    print()

    # ---- Pre-compute alpha_0 once (shared across policies) ----------------
    print("Computing alpha_0 (shared across all policies for this constellation)...")
    t0 = time.perf_counter()
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * dt_s, dt_s=dt_s)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, dt_s, feas_params)
    alpha_0, t_worst = compute_alpha_0(feasibility, T_0=T)
    print(f"  alpha_0 = {alpha_0:.4f} (argmin t = {t_worst}, took {time.perf_counter() - t0:.1f}s)")
    print()

    # ---- Run all (policy, seed) jobs --------------------------------------
    summary: list[tuple[str, int, DiagnosticsReport, str]] = []
    t_start = time.perf_counter()

    for policy_name in policies:
        for seed in seeds:
            fname = trace_filename(config_label, policy_name, T, policy_params.H, seed)
            out_path = out_dir / fname

            if out_path.exists() and not args.force:
                print(f"[skip] {policy_name}/seed={seed}: trace exists ({fname})")
                report = load_existing_report(out_path)
                summary.append((policy_name, seed, report, "cached"))
                continue

            print(f"[run]  {policy_name}/seed={seed} -> {fname}")
            tj = time.perf_counter()
            _, report = run_job(
                walker=walker,
                feasibility_params=feas_params,
                policy_name=policy_name,
                policy_params=policy_params,
                n_epochs=n_epochs,
                dt_s=dt_s,
                seed=seed,
                config_label=config_label,
                out_path=out_path,
                precomputed_alpha_0=(alpha_0, t_worst),
            )
            print(f"       wall-clock: {time.perf_counter() - tj:.1f}s, "
                  f"rho_realized(max)={report.rho_realized_max:.4f}, "
                  f"rho_cover(final)={report.rho_cover_final:.4f}")
            summary.append((policy_name, seed, report, "fresh"))

    total = time.perf_counter() - t_start
    print()
    print("=" * 76)
    print(f"All {len(summary)} jobs done in {total:.1f}s")
    print("=" * 76)

    # ---- Per-policy summary table -----------------------------------------
    print()
    print(f"{'policy':<14} {'seeds':<6} {'rho_max':<10} {'rho_mean':<10} {'rho_cover':<10}")
    print("-" * 60)
    by_policy: dict[str, list[DiagnosticsReport]] = {}
    for policy_name, seed, report, _ in summary:
        by_policy.setdefault(policy_name, []).append(report)
    for policy_name, reps in by_policy.items():
        rho_max = float(np.mean([r.rho_realized_max for r in reps]))
        rho_mean = float(np.mean([r.rho_realized_mean for r in reps]))
        rho_cover = float(np.mean([r.rho_cover_final for r in reps]))
        print(f"{policy_name:<14} {len(reps):<6} {rho_max:<10.4f} {rho_mean:<10.4f} {rho_cover:<10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())