# orbit-match/scripts/check_level_k.py
# Run: python -m scripts.check_level_k

"""Verify LevelKPredictive boundary cases + run the level-k ablation.

Two phases:

Phase 1 (smoke tests, small Walker, ~30 epochs):
  - level=0 must match GreedyMatching exactly.
  - level=1 must match PredictiveMatching exactly.
  - level >= 2 must terminate (no infinite recursion).

Phase 2 (ablation, medium Walker, 2 orbital periods):
  - level in {1, 2, 3, 5}. Compare edges/ep, requests/ep, waste, rho_cover.
  - Question: does level-2 actually predict better than level-1?
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


def make_pp(T: int) -> PolicyParams:
    return PolicyParams(
        H=10, T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )


def make_feas() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0, range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def run_with(policy_name: str, walker, feas_params, n_epochs: int, T: int,
             seed: int = 42, **policy_kwargs):
    """Run a policy; return (action_history, edges_per_epoch, diagnostics, report)."""
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * DT_S, dt_s=DT_S)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, DT_S, feas_params)
    alpha_0, alpha_0_t = compute_alpha_0(feasibility, T_0=T)
    wl = WindowedLaplacian(walker.M, window_length=T)
    wu = WindowedUnion(walker.M, window_length=T)
    rng = make_rng(seed=seed)
    cls = get_policy_class(policy_name)
    policy = cls(
        n=walker.M, feasibility=feasibility, positions=positions, dt_s=DT_S,
        params=make_pp(T=T), rng=rng, windowed_laplacian=wl,
        **policy_kwargs,
    )
    actions_history = np.full((n_epochs, walker.M), -1, dtype=np.int64)
    edges_per_epoch = np.zeros(n_epochs, dtype=np.int64)
    union_edges_per_epoch = np.zeros(n_epochs, dtype=np.int64)
    lambda2_phi = np.zeros(n_epochs, dtype=np.float64)
    lambda2_union = np.zeros(n_epochs, dtype=np.float64)
    matchings = []
    for tt in range(n_epochs):
        a = policy.decide(tt)
        m = actions_to_edges(a)
        wl.push(m)
        wu.push(m)
        lambda2_phi[tt] = wl.lambda_2()
        lambda2_union[tt] = wu.lambda_2()
        actions_history[tt] = a
        edges_per_epoch[tt] = m.shape[0]
        union_edges_per_epoch[tt] = wu.n_edges
        matchings.append(m)
        policy.step(tt, a)

    result = SimulationResult(
        policy_name=policy_name,
        config_label=f"{walker.name}__{policy_name}",
        n=walker.M, n_epochs=n_epochs, T=T, dt_s=DT_S,
        lambda2_phi=lambda2_phi, lambda2_union=lambda2_union,
        n_edges_per_epoch=edges_per_epoch,
        n_union_edges_per_epoch=union_edges_per_epoch,
        actions=actions_history,
        matchings=tuple(matchings),
        final_phi=wl.phi.copy(),
        final_union_adjacency=wu.adjacency.copy(),
        policy_diagnostics={k: np.asarray(v) for k, v in policy.diagnostics().items()},
        metadata={"policy_name": policy_name, "policy_kwargs": policy_kwargs},
    )
    report = build_report(result, feasibility, T_0=T, precomputed_alpha_0=(alpha_0, alpha_0_t))
    return result, report


def request_waste(result):
    req = float((result.actions != -1).sum(axis=1).mean())
    edges = float(result.n_edges_per_epoch.mean())
    waste = req - 2 * edges
    waste_pct = 100.0 * waste / max(req, 1e-9)
    return req, edges, waste, waste_pct


def phase1_smoke_tests() -> int:
    """Boundary cases on small Walker."""
    print("=" * 80)
    print("Phase 1: smoke tests on small Walker (M=24), 30 epochs")
    print("=" * 80)

    walker = WalkerDeltaConfig(M=24, P=4, F=1, altitude_km=550.0, inclination_deg=53.0, name="small")
    feas_params = make_feas()
    n_epochs = 30
    T = 15  # < n_epochs so compute_alpha_0 succeeds; smoke tests only check actions

    # 1. level=0 == greedy.
    print("[1/3] level=0 == greedy")
    res_greedy, _ = run_with("greedy", walker, feas_params, n_epochs, T, seed=42)
    res_l0, _ = run_with("level_k", walker, feas_params, n_epochs, T, seed=42, level=0)
    if not np.array_equal(res_greedy.actions, res_l0.actions):
        n_diff = (res_greedy.actions != res_l0.actions).any(axis=1).sum()
        first = int(np.argmax((res_greedy.actions != res_l0.actions).any(axis=1)))
        print(f"  [FAIL] level=0 differs from greedy on {n_diff}/{n_epochs} epochs (first at t={first})")
        return 1
    print("  [OK] level=0 == greedy on every epoch")

    # 2. level=1 == predictive.
    print("[2/3] level=1 == predictive")
    res_pred, _ = run_with("predictive", walker, feas_params, n_epochs, T, seed=42)
    res_l1, _ = run_with("level_k", walker, feas_params, n_epochs, T, seed=42, level=1)
    if not np.array_equal(res_pred.actions, res_l1.actions):
        n_diff = (res_pred.actions != res_l1.actions).any(axis=1).sum()
        first = int(np.argmax((res_pred.actions != res_l1.actions).any(axis=1)))
        diff_idx = np.flatnonzero(res_pred.actions[first] != res_l1.actions[first])
        print(f"  [FAIL] level=1 differs from predictive on {n_diff}/{n_epochs} epochs")
        print(f"         first diff at t={first}, satellites: {diff_idx[:5].tolist()}")
        print(f"         predictive: {res_pred.actions[first, diff_idx[:5]].tolist()}")
        print(f"         level_k 1:  {res_l1.actions[first, diff_idx[:5]].tolist()}")
        return 1
    print("  [OK] level=1 == predictive on every epoch")

    # 3. level >= 2 terminates.
    print("[3/3] level=2 terminates")
    t0 = time.perf_counter()
    res_l2, _ = run_with("level_k", walker, feas_params, n_epochs, T, seed=42, level=2)
    print(f"  [OK] level=2 finished in {time.perf_counter() - t0:.2f}s")

    print()
    return 0


def phase2_ablation() -> int:
    """Main ablation on medium Walker."""
    print("=" * 80)
    print("Phase 2: level-k ablation on medium Walker (M=60), 2 orbital periods")
    print("=" * 80)

    walker = WalkerDeltaConfig(M=60, P=6, F=2, altitude_km=550.0, inclination_deg=53.0, name="medium")
    feas_params = make_feas()
    duration_periods = 2.0
    n_epochs = int(math.ceil(duration_periods * walker.orbital_period_s / DT_S))
    T = int(math.ceil(walker.orbital_period_s / DT_S))

    print(f"Config: n={walker.M}, n_epochs={n_epochs}, T={T}, seed=42")
    print()
    print(f"{'policy':<14} {'level':>5} {'req/ep':>7} {'edges/ep':>9} {'waste':>7} "
          f"{'waste%':>7} {'rho_max':>8} {'rho_mean':>9} {'rho_cover':>10}")
    print("-" * 90)

    levels = [1, 2, 3, 5]
    results = {}
    for level in levels:
        t0 = time.perf_counter()
        res, rep = run_with("level_k", walker, feas_params, n_epochs, T, seed=42, level=level)
        req, edges, waste, waste_pct = request_waste(res)
        results[level] = (res, rep, time.perf_counter() - t0)
        print(f"{'level_k':<14} {level:>5} {req:>7.2f} {edges:>9.2f} {waste:>7.2f} "
              f"{waste_pct:>6.1f}% {rep.rho_realized_max:>8.4f} {rep.rho_realized_mean:>9.4f} "
              f"{rep.rho_cover_final:>10.4f}  ({results[level][2]:.1f}s)")

    print()
    print("Improvements over level=1:")
    print(f"  {'level':<8} {'Δedges':>9} {'Δwaste':>9} {'Δrho_cover':>12}")
    baseline_res, baseline_rep, _ = results[1]
    base_req, base_edges, base_waste, _ = request_waste(baseline_res)
    base_cover = baseline_rep.rho_cover_final
    for level in levels[1:]:
        res, rep, _ = results[level]
        _, edges, waste, _ = request_waste(res)
        d_edges = edges - base_edges
        d_waste = waste - base_waste
        d_cover = rep.rho_cover_final - base_cover
        print(f"  level={level:<2} {d_edges:>+9.4f} {d_waste:>+9.4f} {d_cover:>+12.6f}")

    print()
    return 0


def main() -> int:
    configure(level="WARNING")
    rc = phase1_smoke_tests()
    if rc != 0:
        return rc
    return phase2_ablation()


if __name__ == "__main__":
    sys.exit(main())
