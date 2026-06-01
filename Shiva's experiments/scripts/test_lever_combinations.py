# orbit-match/scripts/test_lever_combinations.py
# Run: python -m scripts.test_lever_combinations

"""Test combinations of (H, scarcity_beta, history_gamma) with small coefficients.

The single-knob sweeps in test_levers.py showed each lever alone either
does nothing or trades coverage for waste reduction. This script tests
whether a *combination* with small coefficients can dominate the
baseline on BOTH edges/ep AND rho_cover - a true Pareto improvement,
not a trade along the frontier.

The grid: nine variants sampling small values of each knob, plus the
baseline (H=10, beta=0, gamma=0) for direct comparison.

A combination is a Pareto improvement over baseline iff:
    edges  >= baseline.edges      (no worse on density)
    rho_cover >= baseline.rho_cover (no worse on coverage)
    waste_pct <= baseline.waste_pct (no worse on efficiency)
and at least one is strictly better.

If nothing in the grid is a Pareto improvement, we have strong evidence
the baseline policy is at the frontier on this geometry. That itself is
a paper-worthy observation.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

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
from orbitmatch.utils.logging_setup import configure, get_logger
from orbitmatch.utils.seeding import make_rng

log = get_logger(__name__)


DT_S = 10.0
SEED = 42


# Grid: (H, beta, gamma). Plus baseline (10, 0, 0).
GRID = [
    (10, 0.0, 0.0),     # baseline
    (10, 0.1, 0.0),
    (10, 0.3, 0.0),
    (10, 0.5, 0.0),
    (30, 0.1, 0.0),
    (30, 0.3, 0.0),
    (30, 0.5, 0.0),
    (10, 0.3, 0.1),
    (30, 0.3, 0.1),
]


def make_walker() -> WalkerDeltaConfig:
    return WalkerDeltaConfig(M=60, P=6, F=2, altitude_km=550.0, inclination_deg=53.0, name="medium")


def make_feas() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0, range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def make_pp(T: int, H: int) -> PolicyParams:
    return PolicyParams(
        H=H, T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )


def run_combination(
    walker: WalkerDeltaConfig,
    feasibility: np.ndarray,
    positions: np.ndarray,
    n_epochs: int,
    T: int,
    H: int,
    beta: float,
    gamma: float,
    alpha_and_t: tuple[float, int],
) -> tuple[SimulationResult, DiagnosticsReport, float]:
    cls = get_policy_class("lever")
    pp = make_pp(T=T, H=H)
    wl = WindowedLaplacian(walker.M, window_length=T)
    wu = WindowedUnion(walker.M, window_length=T)
    rng = make_rng(seed=SEED)
    policy = cls(
        n=walker.M, feasibility=feasibility, positions=positions, dt_s=DT_S,
        params=pp, rng=rng, windowed_laplacian=wl,
        scarcity_beta=beta, history_gamma=gamma,
    )

    n = walker.M
    lambda2_phi = np.zeros(n_epochs, dtype=np.float64)
    lambda2_union = np.zeros(n_epochs, dtype=np.float64)
    n_edges = np.zeros(n_epochs, dtype=np.int64)
    n_union_edges = np.zeros(n_epochs, dtype=np.int64)
    actions_hist = np.full((n_epochs, n), -1, dtype=np.int64)
    matchings: list[np.ndarray] = []

    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0

    result = SimulationResult(
        policy_name="lever",
        config_label=f"medium__H{H}_b{beta}_g{gamma}",
        n=n, n_epochs=n_epochs, T=T, dt_s=DT_S,
        lambda2_phi=lambda2_phi, lambda2_union=lambda2_union,
        n_edges_per_epoch=n_edges, n_union_edges_per_epoch=n_union_edges,
        actions=actions_hist, matchings=tuple(matchings),
        final_phi=wl.phi.copy(), final_union_adjacency=wu.adjacency.copy(),
        policy_diagnostics={k: np.asarray(v) for k, v in policy.diagnostics().items()},
        metadata={"H": H, "scarcity_beta": beta, "history_gamma": gamma},
    )
    report = build_report(result, feasibility, T_0=T, precomputed_alpha_0=alpha_and_t)
    return result, report, elapsed


def stats(result: SimulationResult, report: DiagnosticsReport) -> dict:
    req = float((result.actions != -1).sum(axis=1).mean())
    edges = float(result.n_edges_per_epoch.mean())
    waste = req - 2 * edges
    waste_pct = 100.0 * waste / max(req, 1e-9)
    return {
        "req": req, "edges": edges, "waste": waste, "waste_pct": waste_pct,
        "rho_mean": float(report.rho_realized_mean),
        "rho_cover": float(report.rho_cover_final),
    }


def main() -> int:
    configure(level="WARNING")

    walker = make_walker()
    feas_params = make_feas()
    duration_periods = 2.0
    n_epochs = int(math.ceil(duration_periods * walker.orbital_period_s / DT_S))
    T = int(math.ceil(walker.orbital_period_s / DT_S))

    print(f"Multi-knob combination search on medium Walker, seed {SEED}")
    print(f"  n={walker.M}, n_epochs={n_epochs}, T={T}")
    print()

    print("Precomputing feasibility + alpha_0...")
    t0 = time.perf_counter()
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * DT_S, dt_s=DT_S)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, DT_S, feas_params)
    alpha_0, alpha_0_t = compute_alpha_0(feasibility, T_0=T)
    a0t = (alpha_0, alpha_0_t)
    print(f"  alpha_0 = {alpha_0:.4f} ({time.perf_counter() - t0:.1f}s)")
    print()

    print(f"{'variant':<22} {'req':>7} {'edges':>7} {'waste':>7} {'waste%':>7} "
          f"{'rho_mean':>9} {'rho_cover':>10} {'time':>6}")
    print("-" * 88)

    results: dict[tuple[int, float, float], dict] = {}
    for (H, beta, gamma) in GRID:
        label = f"H={H} b={beta:g} g={gamma:g}"
        res, rep, elapsed = run_combination(
            walker, feasibility, positions, n_epochs, T, H, beta, gamma, a0t,
        )
        s = stats(res, rep)
        results[(H, beta, gamma)] = s
        print(f"{label:<22} {s['req']:>7.2f} {s['edges']:>7.2f} {s['waste']:>7.2f} "
              f"{s['waste_pct']:>6.1f}% {s['rho_mean']:>9.4f} {s['rho_cover']:>10.4f} "
              f"{elapsed:>5.0f}s")

    print()

    # ---- Pareto analysis ---------------------------------------------------
    baseline_key = (10, 0.0, 0.0)
    if baseline_key not in results:
        print("[ERROR] baseline missing from results")
        return 1
    baseline = results[baseline_key]

    print("=" * 88)
    print("Pareto analysis vs baseline (10, 0, 0)")
    print("=" * 88)
    print(f"baseline:  edges={baseline['edges']:.4f}  rho_cover={baseline['rho_cover']:.4f}  "
          f"waste%={baseline['waste_pct']:.2f}%")
    print()
    print(f"{'variant':<22} {'Δedges':>9} {'Δrho_cover':>12} {'Δwaste%':>9} {'Pareto?':>10}")
    print("-" * 70)
    for key, s in results.items():
        if key == baseline_key:
            continue
        H, beta, gamma = key
        label = f"H={H} b={beta:g} g={gamma:g}"
        d_edges = s["edges"] - baseline["edges"]
        d_cover = s["rho_cover"] - baseline["rho_cover"]
        d_waste_pct = s["waste_pct"] - baseline["waste_pct"]
        # Pareto improvement: weakly better on all three, strictly better on at least one.
        no_worse = (d_edges >= -0.01) and (d_cover >= -0.001) and (d_waste_pct <= 0.5)
        strict = (d_edges > 0.5) or (d_cover > 0.005) or (d_waste_pct < -1.0)
        pareto = "YES" if (no_worse and strict) else ""
        print(f"{label:<22} {d_edges:>+9.4f} {d_cover:>+12.6f} {d_waste_pct:>+8.2f}% "
              f"{pareto:>10}")

    print()

    # Identify the best variant on each metric.
    best_edges_key = max(results, key=lambda k: results[k]["edges"])
    best_cover_key = max(results, key=lambda k: results[k]["rho_cover"])
    best_waste_key = min(results, key=lambda k: results[k]["waste_pct"])

    print("Per-metric leaders:")
    print(f"  best edges/ep:    {best_edges_key}  ({results[best_edges_key]['edges']:.4f})")
    print(f"  best rho_cover:   {best_cover_key}  ({results[best_cover_key]['rho_cover']:.4f})")
    print(f"  best waste%:      {best_waste_key}  ({results[best_waste_key]['waste_pct']:.2f}%)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
