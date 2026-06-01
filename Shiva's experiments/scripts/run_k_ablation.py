# orbit-match/scripts/run_k_ablation.py
# Run: python -m scripts.run_k_ablation [--force]

"""k-ablation experiment: how does strategic recursion depth affect connectivity?

Runs eight policies on the medium Walker constellation:

  k1_sync       k=1 synchronous BR (= predictive, one-step indicator)
  k3_sync       k=3 synchronous BR
  k8_sync       k=8 synchronous BR (deep recursion in sync mode)
  k1_gs         k=1 Gauss-Seidel BR (= predictive on most epochs)
  k3_gs         k=3 Gauss-Seidel BR
  k8_gs         k=8 Gauss-Seidel BR (proxy for iterated-BR fixed point)
  equilibrium   Gauss-Seidel BR to convergence (NE), warm-started from level-1
  adaptive      k=1 default, escalate to k=3 only on close decisions

Output: results/k_ablation/*.npz plus a console table.

The headline questions:
  Q1. Does k > 1 improve the realized union connectivity (rho_realized)?
  Q2. Does k > 1 reduce wasted requests?
  Q3. Does Gauss-Seidel beat synchronous BR at the same depth?
  Q4. Can adaptive get k_max quality at k=1 cost?
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

from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.diagnostics import build_report, compute_alpha_0, DiagnosticsReport
from orbitmatch.experiments.runner import SimulationResult, get_policy_class
from orbitmatch.feasibility.compute import compute_feasibility
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.graph.laplacian import actions_to_edges
from orbitmatch.graph.windowed import WindowedLaplacian, WindowedUnion
from orbitmatch.policy.base import PolicyParams
from orbitmatch.utils.io import PROJECT_ROOT, load_trace, results_dir, save_trace
from orbitmatch.utils.logging_setup import configure, get_logger
from orbitmatch.utils.seeding import make_rng
from orbitmatch.utils.timing import timed

log = get_logger(__name__)


EXPERIMENT_NAME = "k_ablation"
DT_S = 10.0
SEED = 42

# Eight policies to ablate. Each entry: (label, policy_name, extra_kwargs).
POLICIES: list[tuple[str, str, dict]] = [
    ("k1_sync",     "k_step",      {"k": 1, "mode": "sync"}),
    ("k3_sync",     "k_step",      {"k": 3, "mode": "sync"}),
    ("k8_sync",     "k_step",      {"k": 8, "mode": "sync"}),
    ("k1_gs",       "k_step",      {"k": 1, "mode": "gauss_seidel"}),
    ("k3_gs",       "k_step",      {"k": 3, "mode": "gauss_seidel"}),
    ("k8_gs",       "k_step",      {"k": 8, "mode": "gauss_seidel"}),
    ("equilibrium", "equilibrium", {}),
    ("adaptive",    "adaptive",    {"delta_threshold": 0.1, "k_max": 3}),
]


def make_walker() -> WalkerDeltaConfig:
    return WalkerDeltaConfig(M=60, P=6, F=2, altitude_km=550.0, inclination_deg=53.0, name="medium")


def make_feas() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0, range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def make_pp(T: int, H: int = 10) -> PolicyParams:
    return PolicyParams(
        H=H, T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )


def trace_filename(label: str) -> str:
    return f"trace_medium_{label}_seed{SEED}.npz"


def run_one(
    label: str,
    policy_name: str,
    policy_kwargs: dict,
    walker: WalkerDeltaConfig,
    feasibility: np.ndarray,
    positions: np.ndarray,
    n_epochs: int,
    T: int,
    out_dir: Path,
    alpha_0_and_t: tuple[float, int],
) -> tuple[SimulationResult, DiagnosticsReport]:
    """Build the policy directly and drive the loop.

    We do NOT use run_simulation here because it doesn't accept
    extra policy_kwargs (k, delta_threshold, k_max). The cost is a few
    extra lines of glue.
    """
    PolicyCls = get_policy_class(policy_name)
    pp = make_pp(T=T)
    wl = WindowedLaplacian(walker.M, window_length=T)
    wu = WindowedUnion(walker.M, window_length=T)
    rng = make_rng(seed=SEED)

    policy = PolicyCls(
        n=walker.M, feasibility=feasibility, positions=positions, dt_s=DT_S,
        params=pp, rng=rng, windowed_laplacian=wl,
        **policy_kwargs,
    )

    n = walker.M
    lambda2_phi = np.zeros(n_epochs, dtype=np.float64)
    lambda2_union = np.zeros(n_epochs, dtype=np.float64)
    n_edges = np.zeros(n_epochs, dtype=np.int64)
    n_union_edges = np.zeros(n_epochs, dtype=np.int64)
    actions_hist = np.full((n_epochs, n), -1, dtype=np.int64)
    matchings: list[np.ndarray] = []

    with timed(f"k_ablation[{label}]"):
        for t in range(n_epochs):
            a = policy.decide(t)
            m = actions_to_edges(a)
            wl.push(m)
            wu.push(m)
            lambda2_phi[t] = wl.lambda_2()
            lambda2_union[t] = wu.lambda_2()
            n_edges[t] = m.shape[0]
            n_union_edges[t] = wu.n_edges
            actions_hist[t] = a
            matchings.append(m)
            policy.step(t, a)

    # Pack diagnostics.
    diag = {k: np.asarray(v) for k, v in policy.diagnostics().items()}

    result = SimulationResult(
        policy_name=policy_name,
        config_label=f"medium__{label}",
        n=n,
        n_epochs=n_epochs,
        T=T,
        dt_s=DT_S,
        lambda2_phi=lambda2_phi,
        lambda2_union=lambda2_union,
        n_edges_per_epoch=n_edges,
        n_union_edges_per_epoch=n_union_edges,
        actions=actions_hist,
        matchings=tuple(matchings),
        final_phi=wl.phi.copy(),
        final_union_adjacency=wu.adjacency.copy(),
        policy_diagnostics=diag,
        metadata={
            "policy_name": policy_name,
            "policy_kwargs": policy_kwargs,
            "label": label,
            "walker": {
                "M": walker.M, "P": walker.P, "F": walker.F,
                "altitude_km": walker.altitude_km, "inclination_deg": walker.inclination_deg,
            },
            "n_epochs": n_epochs,
            "dt_s": DT_S,
            "seed": SEED,
            "policy_params": asdict(pp),
        },
    )

    report = build_report(result, feasibility, T_0=T, precomputed_alpha_0=alpha_0_and_t)

    # Save to disk.
    metadata = dict(result.metadata)
    metadata["report"] = report.to_metadata()
    metadata["experiment"] = EXPERIMENT_NAME
    out_path = out_dir / trace_filename(label)
    save_trace(out_path, arrays=result.to_arrays(), metadata=metadata)

    return result, report


def request_waste_stats(result: SimulationResult) -> dict[str, float]:
    """Compute requests/edges/waste per epoch."""
    requests = (result.actions != -1).sum(axis=1)
    edges = result.n_edges_per_epoch
    successful = 2 * edges
    wasted = requests - successful
    return {
        "requests_per_epoch": float(requests.mean()),
        "edges_per_epoch": float(edges.mean()),
        "waste_per_epoch": float(wasted.mean()),
        "waste_pct": 100.0 * float(wasted.mean()) / max(float(requests.mean()), 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Re-run even if a trace exists.")
    args = parser.parse_args()

    configure(level="WARNING")

    walker = make_walker()
    feas_params = make_feas()

    # 2 orbital periods to match the existing lambda2_traces run scope.
    duration_periods = 2.0
    n_epochs = int(math.ceil(duration_periods * walker.orbital_period_s / DT_S))
    T = int(math.ceil(walker.orbital_period_s / DT_S))

    out_dir = results_dir(EXPERIMENT_NAME)
    print(f"k-ablation experiment")
    print(f"  constellation: medium (n={walker.M})")
    print(f"  n_epochs:      {n_epochs}  ({duration_periods} periods at dt={DT_S}s)")
    print(f"  T:             {T} epochs")
    print(f"  seed:          {SEED}")
    print(f"  output dir:    {out_dir}")
    print(f"  policies:      {[p[0] for p in POLICIES]}")
    print()

    # Pre-compute feasibility once and alpha_0 once.
    print("Precomputing feasibility tensor + alpha_0 ...")
    t0 = time.perf_counter()
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * DT_S, dt_s=DT_S)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, DT_S, feas_params)
    alpha_0, alpha_0_t = compute_alpha_0(feasibility, T_0=T)
    print(f"  alpha_0 = {alpha_0:.4f} ({time.perf_counter() - t0:.1f}s)")
    print()

    # Run all six policies.
    results: dict[str, tuple[SimulationResult, DiagnosticsReport]] = {}
    t_start = time.perf_counter()

    for label, policy_name, kwargs in POLICIES:
        out_path = out_dir / trace_filename(label)
        if out_path.exists() and not args.force:
            print(f"[skip] {label}: trace exists ({trace_filename(label)})")
            # We could reconstruct from disk here for the summary, but since
            # the run is fresh we just skip the summary for cached items.
            continue
        print(f"[run]  {label} ({policy_name}, {kwargs})")
        tj = time.perf_counter()
        res, rep = run_one(
            label, policy_name, kwargs,
            walker, feasibility, positions, n_epochs, T, out_dir,
            (alpha_0, alpha_0_t),
        )
        results[label] = (res, rep)
        stats = request_waste_stats(res)
        print(f"       wall-clock: {time.perf_counter() - tj:.1f}s, "
              f"requests/ep: {stats['requests_per_epoch']:.2f}, "
              f"edges/ep: {stats['edges_per_epoch']:.2f}, "
              f"waste: {stats['waste_pct']:.1f}%, "
              f"rho_max: {rep.rho_realized_max:.4f}, "
              f"rho_mean: {rep.rho_realized_mean:.4f}")

    total = time.perf_counter() - t_start
    print()
    print("=" * 90)
    print(f"All ablation jobs done in {total:.1f}s")
    print("=" * 90)
    print()

    # ---- Summary table -----------------------------------------------------
    print(f"{'policy':<14} {'req/ep':>8} {'edges/ep':>9} {'waste/ep':>9} {'waste%':>7} "
          f"{'rho_max':>9} {'rho_mean':>9} {'rho_cover':>10}")
    print("-" * 90)
    for label, _, _ in POLICIES:
        if label not in results:
            continue
        res, rep = results[label]
        s = request_waste_stats(res)
        print(f"{label:<14} {s['requests_per_epoch']:>8.2f} {s['edges_per_epoch']:>9.2f} "
              f"{s['waste_per_epoch']:>9.2f} {s['waste_pct']:>6.1f}% "
              f"{rep.rho_realized_max:>9.4f} {rep.rho_realized_mean:>9.4f} "
              f"{rep.rho_cover_final:>10.4f}")

    # ---- Adaptive-specific diagnostic --------------------------------------
    if "adaptive" in results:
        res, _ = results["adaptive"]
        amb_frac = res.policy_diagnostics.get("ambiguous_frac")
        if amb_frac is not None:
            amb = np.asarray(amb_frac)
            print()
            print(f"Adaptive: mean ambiguous fraction = {amb.mean():.2%} "
                  f"(max {amb.max():.2%}, min {amb.min():.2%})")
            print(f"Cost vs k=1: 1 + {amb.mean():.4f} * (k_max - 1) = "
                  f"{1 + amb.mean() * 2:.4f}x (with k_max=3)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
