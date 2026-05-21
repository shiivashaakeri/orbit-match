# orbit-match/scripts/validate_k1_gs.py
# Run: python -m scripts.validate_k1_gs

"""Validate the k1_gs finding before reframing the paper around it.

The k_ablation showed k1_gs (k=1 Gauss-Seidel BR from level-0) is the
densest, zero-waste policy on medium config, beating equilibrium on
edges-per-epoch. Before committing to this as the headline, we want
to know:

Validation A. Is the result robust across seeds?
    Run k1_gs on seeds {42, 43, 44}. Compare metrics.

Validation B. Does the satellite update order matter?
    Run k1_gs on seed 42 with three orderings:
      - identity (0, 1, ..., n-1)
      - reversed (n-1, ..., 1, 0)
      - random (with fixed permutation seed)
    Compare metrics.

Output: console table per validation. Traces saved under
results/k_gs_validation/ for inspection.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict
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


EXPERIMENT_NAME = "k_gs_validation"
DT_S = 10.0


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


def run_one(
    label: str,
    walker: WalkerDeltaConfig,
    feasibility: np.ndarray,
    positions: np.ndarray,
    n_epochs: int,
    T: int,
    seed: int,
    update_order: np.ndarray | None,
    alpha_0_and_t: tuple[float, int],
    out_dir: Path,
) -> tuple[SimulationResult, DiagnosticsReport]:
    """Run one k1_gs job with the given seed and update_order."""
    PolicyCls = get_policy_class("k_step")
    pp = make_pp(T=T)
    wl = WindowedLaplacian(walker.M, window_length=T)
    wu = WindowedUnion(walker.M, window_length=T)
    rng = make_rng(seed=seed)

    policy = PolicyCls(
        n=walker.M, feasibility=feasibility, positions=positions, dt_s=DT_S,
        params=pp, rng=rng, windowed_laplacian=wl,
        k=1, mode="gauss_seidel", update_order=update_order,
    )

    n = walker.M
    lambda2_phi = np.zeros(n_epochs, dtype=np.float64)
    lambda2_union = np.zeros(n_epochs, dtype=np.float64)
    n_edges = np.zeros(n_epochs, dtype=np.int64)
    n_union_edges = np.zeros(n_epochs, dtype=np.int64)
    actions_hist = np.full((n_epochs, n), -1, dtype=np.int64)
    matchings: list[np.ndarray] = []

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

    diag = {k: np.asarray(v) for k, v in policy.diagnostics().items()}

    result = SimulationResult(
        policy_name="k_step",
        config_label=f"medium__{label}",
        n=n, n_epochs=n_epochs, T=T, dt_s=DT_S,
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
            "policy_name": "k_step",
            "policy_kwargs": {"k": 1, "mode": "gauss_seidel"},
            "label": label,
            "seed": seed,
            "update_order": (update_order.tolist() if update_order is not None else None),
        },
    )

    report = build_report(result, feasibility, T_0=T, precomputed_alpha_0=alpha_0_and_t)

    metadata = dict(result.metadata)
    metadata["report"] = report.to_metadata()
    metadata["experiment"] = EXPERIMENT_NAME
    out_path = out_dir / f"trace_{label}.npz"
    save_trace(out_path, arrays=result.to_arrays(), metadata=metadata)

    return result, report


def request_waste(result: SimulationResult) -> tuple[float, float, float, float]:
    requests = (result.actions != -1).sum(axis=1).mean()
    edges = result.n_edges_per_epoch.mean()
    wasted = requests - 2 * edges
    waste_pct = 100.0 * wasted / max(requests, 1e-9)
    return float(requests), float(edges), float(wasted), float(waste_pct)


def print_row(label: str, res: SimulationResult, rep: DiagnosticsReport) -> None:
    req, edges, waste, waste_pct = request_waste(res)
    print(f"{label:<24} {req:>7.2f} {edges:>9.2f} {waste:>8.2f} {waste_pct:>6.1f}% "
          f"{rep.rho_realized_max:>9.4f} {rep.rho_realized_mean:>9.4f} "
          f"{rep.rho_cover_final:>10.4f}")


def header() -> None:
    print(f"{'label':<24} {'req/ep':>7} {'edges/ep':>9} {'waste':>8} {'waste%':>7} "
          f"{'rho_max':>9} {'rho_mean':>9} {'rho_cover':>10}")
    print("-" * 90)


def main() -> int:
    configure(level="WARNING")

    walker = make_walker()
    feas_params = make_feas()
    duration_periods = 2.0
    n_epochs = int(math.ceil(duration_periods * walker.orbital_period_s / DT_S))
    T = int(math.ceil(walker.orbital_period_s / DT_S))

    out_dir = results_dir(EXPERIMENT_NAME)
    print(f"k1_gs validation")
    print(f"  constellation: medium, T={T}, n_epochs={n_epochs}")
    print(f"  output dir:    {out_dir}")
    print()

    print("Precomputing feasibility tensor + alpha_0 ...")
    t0 = time.perf_counter()
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * DT_S, dt_s=DT_S)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, DT_S, feas_params)
    alpha_0, alpha_0_t = compute_alpha_0(feasibility, T_0=T)
    a0t = (alpha_0, alpha_0_t)
    print(f"  alpha_0 = {alpha_0:.4f} ({time.perf_counter() - t0:.1f}s)")
    print()

    # ---- Validation A: seed robustness ------------------------------------
    print("=" * 90)
    print("Validation A: seed robustness (identity ordering, three seeds)")
    print("=" * 90)
    header()

    seed_results = {}
    for seed in [42, 43, 44]:
        label = f"k1_gs_seed{seed}_identity"
        t0 = time.perf_counter()
        res, rep = run_one(label, walker, feasibility, positions, n_epochs, T,
                           seed=seed, update_order=None, alpha_0_and_t=a0t, out_dir=out_dir)
        print_row(f"seed={seed}", res, rep)
        seed_results[seed] = (res, rep, time.perf_counter() - t0)

    # Compute per-seed agreement.
    edges_by_seed = {s: r[0].n_edges_per_epoch.mean() for s, r in seed_results.items()}
    cover_by_seed = {s: r[1].rho_cover_final for s, r in seed_results.items()}
    edges_spread = max(edges_by_seed.values()) - min(edges_by_seed.values())
    cover_spread = max(cover_by_seed.values()) - min(cover_by_seed.values())
    print()
    print(f"  edges/ep spread across seeds: {edges_spread:.4f} (max - min)")
    print(f"  rho_cover spread across seeds: {cover_spread:.4f} (max - min)")
    if edges_spread < 0.5 and cover_spread < 0.01:
        print(f"  [OK] tight agreement: k1_gs is seed-robust")
    else:
        print(f"  [WARN] significant variation; investigate")
    print()

    # ---- Validation B: ordering sensitivity --------------------------------
    print("=" * 90)
    print("Validation B: ordering sensitivity (seed 42, three permutations)")
    print("=" * 90)
    header()

    n = walker.M
    orderings = {
        "identity":  np.arange(n, dtype=np.int64),
        "reversed":  np.arange(n - 1, -1, -1, dtype=np.int64),
        "random_p7": np.random.default_rng(7).permutation(n).astype(np.int64),
    }

    order_results = {}
    for name, perm in orderings.items():
        label = f"k1_gs_seed42_{name}"
        t0 = time.perf_counter()
        res, rep = run_one(label, walker, feasibility, positions, n_epochs, T,
                           seed=42, update_order=perm, alpha_0_and_t=a0t, out_dir=out_dir)
        print_row(name, res, rep)
        order_results[name] = (res, rep, time.perf_counter() - t0)

    edges_by_order = {n: r[0].n_edges_per_epoch.mean() for n, r in order_results.items()}
    cover_by_order = {n: r[1].rho_cover_final for n, r in order_results.items()}
    edges_spread = max(edges_by_order.values()) - min(edges_by_order.values())
    cover_spread = max(cover_by_order.values()) - min(cover_by_order.values())
    print()
    print(f"  edges/ep spread across orderings: {edges_spread:.4f} (max - min)")
    print(f"  rho_cover spread across orderings: {cover_spread:.4f} (max - min)")
    if edges_spread < 0.5 and cover_spread < 0.01:
        print(f"  [OK] tight agreement: k1_gs is order-invariant on this game")
    else:
        print(f"  [INFO] meaningful order-dependence: GS converges to different NEs")
        print(f"         this would deserve a remark in the paper")
    print()

    print("Validation done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
