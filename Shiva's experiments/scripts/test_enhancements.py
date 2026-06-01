# orbit-match/scripts/test_enhancements.py
# Run: python -m scripts.test_enhancements

"""Test ideas 2 (temporal warm-start) and 5 (most-constrained-first ordering).

Ablates four variants of k1_gs on medium config:

  baseline_gs        identity ordering, level-0 start
  feasibility_order  feasibility ordering, level-0 start
  warmstart          identity ordering, temporal warm-start
  both               feasibility ordering + temporal warm-start

Compares against the original k1_gs identity (loaded from canonical
trace) to verify the baseline_gs matches exactly. Reports edges/ep,
requests/ep, waste, rho_cover, rho_mean for each variant.
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
from orbitmatch.utils.io import results_dir, save_trace
from orbitmatch.utils.logging_setup import configure, get_logger
from orbitmatch.utils.seeding import make_rng

log = get_logger(__name__)


DT_S = 10.0


def make_walker() -> WalkerDeltaConfig:
    return WalkerDeltaConfig(M=60, P=6, F=2, altitude_km=550.0, inclination_deg=53.0, name="medium")


def make_feas() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0, range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def make_pp(T: int) -> PolicyParams:
    return PolicyParams(
        H=10, T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )


def run_one(label, walker, feasibility, positions, n_epochs, T, alpha0, **policy_kwargs):
    PolicyCls = get_policy_class("k_step")
    pp = make_pp(T=T)
    wl = WindowedLaplacian(walker.M, window_length=T)
    wu = WindowedUnion(walker.M, window_length=T)
    rng = make_rng(seed=42)

    policy = PolicyCls(
        n=walker.M, feasibility=feasibility, positions=positions, dt_s=DT_S,
        params=pp, rng=rng, windowed_laplacian=wl,
        k=1, mode="gauss_seidel",
        **policy_kwargs,
    )

    n_edges = np.zeros(n_epochs, dtype=np.int64)
    n_union_edges = np.zeros(n_epochs, dtype=np.int64)
    lambda2_phi = np.zeros(n_epochs, dtype=np.float64)
    lambda2_union = np.zeros(n_epochs, dtype=np.float64)
    actions_hist = np.full((n_epochs, walker.M), -1, dtype=np.int64)
    matchings = []

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

    result = SimulationResult(
        policy_name="k_step", config_label=f"medium__{label}",
        n=walker.M, n_epochs=n_epochs, T=T, dt_s=DT_S,
        lambda2_phi=lambda2_phi, lambda2_union=lambda2_union,
        n_edges_per_epoch=n_edges, n_union_edges_per_epoch=n_union_edges,
        actions=actions_hist, matchings=tuple(matchings),
        final_phi=wl.phi.copy(), final_union_adjacency=wu.adjacency.copy(),
        policy_diagnostics={k: np.asarray(v) for k, v in policy.diagnostics().items()},
        metadata={"label": label, "policy_kwargs": policy_kwargs},
    )
    report = build_report(result, feasibility, T_0=T, precomputed_alpha_0=alpha0)
    return result, report


def stats(result, report):
    req = (result.actions != -1).sum(axis=1).mean()
    edges = result.n_edges_per_epoch.mean()
    waste = req - 2 * edges
    waste_pct = 100.0 * waste / max(req, 1e-9)
    return {
        "req": float(req),
        "edges": float(edges),
        "waste": float(waste),
        "waste_pct": float(waste_pct),
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

    print(f"Testing enhancements (medium n={walker.M}, n_epochs={n_epochs}, T={T})")
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

    variants = [
        ("baseline_gs",         dict(order_by="identity",    temporal_warmstart=False,
                                     partial_observation=False, update_pointing_on_request=False,
                                     dynamic_prior=False)),
        ("feasibility_order",   dict(order_by="feasibility", temporal_warmstart=False,
                                     partial_observation=False, update_pointing_on_request=False,
                                     dynamic_prior=False)),
        ("warmstart",           dict(order_by="identity",    temporal_warmstart=True,
                                     partial_observation=False, update_pointing_on_request=False,
                                     dynamic_prior=False)),
        ("partial_obs",         dict(order_by="identity",    temporal_warmstart=False,
                                     partial_observation=True,  update_pointing_on_request=False,
                                     dynamic_prior=False)),
        ("pointing_on_req",     dict(order_by="identity",    temporal_warmstart=False,
                                     partial_observation=False, update_pointing_on_request=True,
                                     dynamic_prior=False)),
        ("dynamic_prior",       dict(order_by="identity",    temporal_warmstart=False,
                                     partial_observation=False, update_pointing_on_request=False,
                                     dynamic_prior=True)),
        ("all_on",              dict(order_by="feasibility", temporal_warmstart=True,
                                     partial_observation=True,  update_pointing_on_request=True,
                                     dynamic_prior=True)),
    ]

    print(f"{'variant':<22} {'req':>7} {'edges':>7} {'waste':>7} {'waste%':>7} "
          f"{'rho_mean':>9} {'rho_cover':>10}")
    print("-" * 80)

    results = {}
    for label, kwargs in variants:
        t0 = time.perf_counter()
        res, rep = run_one(label, walker, feasibility, positions, n_epochs, T, a0t, **kwargs)
        s = stats(res, rep)
        results[label] = (res, rep, s)
        print(f"{label:<22} {s['req']:>7.2f} {s['edges']:>7.2f} {s['waste']:>7.2f} "
              f"{s['waste_pct']:>6.1f}% {s['rho_mean']:>9.4f} {s['rho_cover']:>10.4f}  "
              f"({time.perf_counter() - t0:.1f}s)")

    print()
    # Sanity: baseline_gs should match the canonical k1_gs identity exactly.
    canonical_path = Path("results/canonical/fig3/k1_gs_identity_seed42.npz")
    if canonical_path.exists():
        from orbitmatch.utils.io import load_trace
        arrays, _ = load_trace(canonical_path)
        canonical_edges_mean = float(arrays["n_edges_per_epoch"].mean())
        baseline_edges = results["baseline_gs"][2]["edges"]
        if abs(canonical_edges_mean - baseline_edges) < 0.01:
            print(f"  [OK] baseline_gs matches canonical k1_gs identity "
                  f"(edges/ep {baseline_edges:.4f} vs {canonical_edges_mean:.4f})")
        else:
            print(f"  [WARN] baseline_gs diverged from canonical: "
                  f"{baseline_edges:.4f} vs {canonical_edges_mean:.4f}")

    # Headline comparison: improvements relative to baseline.
    baseline = results["baseline_gs"][2]
    print()
    print(f"Improvements over baseline_gs:")
    print(f"  {'variant':<22} {'Δedges':>9} {'Δwaste':>9} {'Δrho_cover':>12}")
    for label, _ in variants[1:]:
        s = results[label][2]
        d_edges = s["edges"] - baseline["edges"]
        d_waste = s["waste"] - baseline["waste"]
        d_cover = s["rho_cover"] - baseline["rho_cover"]
        print(f"  {label:<22} {d_edges:>+9.4f} {d_waste:>+9.4f} {d_cover:>+12.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
