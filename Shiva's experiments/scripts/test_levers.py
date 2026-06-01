# orbit-match/scripts/test_levers.py
# Run: python -m scripts.test_levers

"""Lever-1a + 1b + 2a ablation: bigger H, scarcity weighting, history.

Four isolated phases, each holding the other parameters at baseline:

Phase 1: H sweep.
  H in {10, 30, 60, 100}, beta=0, gamma=0.
  Tests whether longer lookahead in the value function helps.

Phase 2: scarcity beta sweep.
  H=10, beta in {0, 1, 3, 10}, gamma=0.
  Tests whether weighting rare edges more heavily helps.

Phase 3: history gamma sweep.
  H=10, beta=0, gamma in {0, 1, 3, 10}.
  Tests whether boosting historically-reciprocating partners helps.

Phase 4: combined.
  Best H, best beta, best gamma together; compared to baseline.

Each run takes ~1-3 min on medium config. Total wall-clock: ~30-40 min.

All runs use seed 42 since the predictive family is largely deterministic
on this geometry; F11/F12 from earlier validation work establishes that
seed-spread is negligible.
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


def run_lever(
    label: str,
    walker: WalkerDeltaConfig,
    feasibility: np.ndarray,
    positions: np.ndarray,
    n_epochs: int,
    T: int,
    H: int,
    alpha_and_t: tuple[float, int],
    scarcity_beta: float = 0.0,
    history_gamma: float = 0.0,
) -> tuple[SimulationResult, DiagnosticsReport, float]:
    """Run a single LeverPredictive variant."""
    cls = get_policy_class("lever")
    pp = make_pp(T=T, H=H)
    wl = WindowedLaplacian(walker.M, window_length=T)
    wu = WindowedUnion(walker.M, window_length=T)
    rng = make_rng(seed=SEED)

    policy = cls(
        n=walker.M, feasibility=feasibility, positions=positions, dt_s=DT_S,
        params=pp, rng=rng, windowed_laplacian=wl,
        scarcity_beta=scarcity_beta,
        history_gamma=history_gamma,
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
        config_label=f"medium__{label}",
        n=n, n_epochs=n_epochs, T=T, dt_s=DT_S,
        lambda2_phi=lambda2_phi, lambda2_union=lambda2_union,
        n_edges_per_epoch=n_edges, n_union_edges_per_epoch=n_union_edges,
        actions=actions_hist, matchings=tuple(matchings),
        final_phi=wl.phi.copy(), final_union_adjacency=wu.adjacency.copy(),
        policy_diagnostics={k: np.asarray(v) for k, v in policy.diagnostics().items()},
        metadata={
            "label": label, "H": H,
            "scarcity_beta": scarcity_beta,
            "history_gamma": history_gamma,
        },
    )
    report = build_report(result, feasibility, T_0=T, precomputed_alpha_0=alpha_and_t)
    return result, report, elapsed


def stats(result: SimulationResult, report: DiagnosticsReport) -> dict[str, float]:
    req = float((result.actions != -1).sum(axis=1).mean())
    edges = float(result.n_edges_per_epoch.mean())
    waste = req - 2 * edges
    waste_pct = 100.0 * waste / max(req, 1e-9)
    return {
        "req": req, "edges": edges, "waste": waste, "waste_pct": waste_pct,
        "rho_mean": float(report.rho_realized_mean),
        "rho_cover": float(report.rho_cover_final),
    }


def header() -> None:
    print(f"{'variant':<22} {'req':>7} {'edges':>7} {'waste':>7} {'waste%':>7} "
          f"{'rho_mean':>9} {'rho_cover':>10} {'time':>6}")
    print("-" * 88)


def print_row(label: str, s: dict, elapsed: float) -> None:
    print(f"{label:<22} {s['req']:>7.2f} {s['edges']:>7.2f} {s['waste']:>7.2f} "
          f"{s['waste_pct']:>6.1f}% {s['rho_mean']:>9.4f} {s['rho_cover']:>10.4f} "
          f"{elapsed:>5.0f}s")


def main() -> int:
    configure(level="WARNING")

    walker = make_walker()
    feas_params = make_feas()
    duration_periods = 2.0
    n_epochs = int(math.ceil(duration_periods * walker.orbital_period_s / DT_S))
    T = int(math.ceil(walker.orbital_period_s / DT_S))

    print(f"Lever ablation on medium Walker, 2 orbital periods, seed {SEED}")
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

    all_results: dict[str, tuple[dict, float]] = {}

    # ---- Phase 1: H sweep ----------------------------------------------------
    print("=" * 88)
    print("Phase 1: H sweep (beta=0, gamma=0)")
    print("=" * 88)
    header()

    h_values = [10, 30, 60, 100]
    h_stats = {}
    for H in h_values:
        label = f"H={H}"
        res, rep, elapsed = run_lever(label, walker, feasibility, positions,
                                       n_epochs, T, H, a0t)
        s = stats(res, rep)
        h_stats[H] = s
        all_results[f"phase1_H{H}"] = (s, elapsed)
        print_row(label, s, elapsed)
    print()

    # Pick the best H by edges/ep (tie-break on rho_cover).
    best_H = max(h_values, key=lambda h: (h_stats[h]["edges"], h_stats[h]["rho_cover"]))
    print(f"  best H by edges/ep: {best_H} (edges/ep {h_stats[best_H]['edges']:.2f})")
    print()

    # ---- Phase 2: scarcity beta sweep ---------------------------------------
    print("=" * 88)
    print("Phase 2: scarcity_beta sweep (H=10, gamma=0)")
    print("=" * 88)
    header()

    beta_values = [0.0, 1.0, 3.0, 10.0]
    beta_stats = {}
    for beta in beta_values:
        label = f"beta={beta:g}"
        res, rep, elapsed = run_lever(label, walker, feasibility, positions,
                                       n_epochs, T, H=10, alpha_and_t=a0t,
                                       scarcity_beta=beta)
        s = stats(res, rep)
        beta_stats[beta] = s
        all_results[f"phase2_beta{beta}"] = (s, elapsed)
        print_row(label, s, elapsed)
    print()

    best_beta = max(beta_values, key=lambda b: (beta_stats[b]["edges"], beta_stats[b]["rho_cover"]))
    print(f"  best beta by edges/ep: {best_beta} (edges/ep {beta_stats[best_beta]['edges']:.2f})")
    print()

    # ---- Phase 3: history gamma sweep ---------------------------------------
    print("=" * 88)
    print("Phase 3: history_gamma sweep (H=10, beta=0)")
    print("=" * 88)
    header()

    gamma_values = [0.0, 1.0, 3.0, 10.0]
    gamma_stats = {}
    for gamma in gamma_values:
        label = f"gamma={gamma:g}"
        res, rep, elapsed = run_lever(label, walker, feasibility, positions,
                                       n_epochs, T, H=10, alpha_and_t=a0t,
                                       history_gamma=gamma)
        s = stats(res, rep)
        gamma_stats[gamma] = s
        all_results[f"phase3_gamma{gamma}"] = (s, elapsed)
        print_row(label, s, elapsed)
    print()

    best_gamma = max(gamma_values, key=lambda g: (gamma_stats[g]["edges"], gamma_stats[g]["rho_cover"]))
    print(f"  best gamma by edges/ep: {best_gamma} (edges/ep {gamma_stats[best_gamma]['edges']:.2f})")
    print()

    # ---- Phase 4: combined ---------------------------------------------------
    print("=" * 88)
    print(f"Phase 4: combined (H={best_H}, beta={best_beta:g}, gamma={best_gamma:g})")
    print("=" * 88)
    header()

    res, rep, elapsed = run_lever("baseline", walker, feasibility, positions,
                                   n_epochs, T, H=10, alpha_and_t=a0t,
                                   scarcity_beta=0.0, history_gamma=0.0)
    baseline_s = stats(res, rep)
    print_row("baseline", baseline_s, elapsed)

    res, rep, elapsed = run_lever("combined", walker, feasibility, positions,
                                   n_epochs, T, H=best_H, alpha_and_t=a0t,
                                   scarcity_beta=best_beta, history_gamma=best_gamma)
    combined_s = stats(res, rep)
    print_row("combined", combined_s, elapsed)
    print()

    print("Improvement of combined over baseline:")
    print(f"  Δedges   : {combined_s['edges'] - baseline_s['edges']:+.4f}")
    print(f"  Δwaste   : {combined_s['waste'] - baseline_s['waste']:+.4f}")
    print(f"  Δwaste%  : {combined_s['waste_pct'] - baseline_s['waste_pct']:+.2f}%")
    print(f"  Δρ_cover : {combined_s['rho_cover'] - baseline_s['rho_cover']:+.6f}")
    print(f"  Δρ_mean  : {combined_s['rho_mean'] - baseline_s['rho_mean']:+.6f}")
    print()

    # ---- Final summary table -------------------------------------------------
    print("=" * 88)
    print("Final summary (all phases)")
    print("=" * 88)
    header()
    for key, (s, elapsed) in all_results.items():
        print_row(key, s, elapsed)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
