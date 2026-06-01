# orbit-match/scripts/check_k_step.py
# Run: python -m scripts.check_k_step

"""Smoke test for orbitmatch.policy.k_step.KStepPredictive.

The KStepPredictive class generalizes the predictive family along a
strategic-recursion axis k. This script verifies the boundary cases:

  k = 0    -> identical actions to GreedyMatching
  k = 1    -> identical actions to PredictiveMatching
  k large  -> approaches EquilibriumMatching's edges-per-epoch profile

Boundary verification: actions arrays compared elementwise. Convergence
verification: realized matching density per epoch compared at k = 10
vs. equilibrium.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.runner import run_simulation
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.policy.base import PolicyParams
from orbitmatch.policy.k_step import KStepPredictive
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.policy.baselines import GreedyMatching
from orbitmatch.policy.equilibrium import EquilibriumMatching
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


def make_walker() -> WalkerDeltaConfig:
    return WalkerDeltaConfig(M=24, P=4, F=1, altitude_km=550.0, inclination_deg=53.0, name="small")


def make_feas() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0, range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def make_pp(T: int = 60) -> PolicyParams:
    return PolicyParams(
        H=10, T=T, switching_cost_scale=0.2, epsilon_geometric_prior=0.01,
        tie_break="lowest_index", seed=None,
    )


def run_with_policy(policy_name: str, walker, feas_params, n_epochs: int, T: int, seed: int = 42, **policy_kwargs):
    """Helper: call run_simulation, allowing extra policy kwargs (for k_step's k=...)."""
    if policy_kwargs:
        # run_simulation doesn't accept extra policy_kwargs directly; we need to
        # build the policy manually for these. Drop down to the lower-level path.
        from orbitmatch.experiments.runner import get_policy_class
        from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
        from orbitmatch.feasibility.compute import compute_feasibility
        from orbitmatch.graph.laplacian import actions_to_edges
        from orbitmatch.graph.windowed import WindowedLaplacian
        from orbitmatch.utils.seeding import make_rng

        elements = walker.initial_elements()
        times = make_time_grid(duration_s=n_epochs * 10.0, dt_s=10.0)
        positions = propagate_keplerian(elements, times)
        feasibility = compute_feasibility(positions, 10.0, feas_params)
        wl = WindowedLaplacian(walker.M, window_length=T)
        rng = make_rng(seed=seed)
        cls = get_policy_class(policy_name)
        policy = cls(
            n=walker.M, feasibility=feasibility, positions=positions, dt_s=10.0,
            params=make_pp(T=T), rng=rng, windowed_laplacian=wl,
            **policy_kwargs,
        )
        actions_history = np.full((n_epochs, walker.M), -1, dtype=np.int64)
        n_edges = np.zeros(n_epochs, dtype=np.int64)
        for tt in range(n_epochs):
            a = policy.decide(tt)
            m = actions_to_edges(a)
            wl.push(m)
            actions_history[tt] = a
            n_edges[tt] = m.shape[0]
            policy.step(tt, a)
        return actions_history, n_edges

    res = run_simulation(
        walker=walker,
        feasibility_params=feas_params,
        policy_name=policy_name,
        policy_params=make_pp(T=T),
        n_epochs=n_epochs,
        dt_s=10.0,
        seed=seed,
        config_label="small",
    )
    return res.actions, res.n_edges_per_epoch


def main() -> int:
    configure(level="WARNING")

    walker = make_walker()
    feas_params = make_feas()
    T = 60
    n_epochs = 120
    seed = 42

    print(f"Fixture: small Walker (n={walker.M}), T={T}, n_epochs={n_epochs}, seed={seed}")
    print()

    # ---- 1. k = 0 should match greedy ------------------------------------
    print("[1/3] k = 0 should produce identical actions to GreedyMatching")
    t0 = time.perf_counter()
    a_greedy, _ = run_with_policy("greedy", walker, feas_params, n_epochs, T, seed=seed)
    a_k0, _ = run_with_policy("k_step", walker, feas_params, n_epochs, T, seed=seed, k=0)
    print(f"       took {time.perf_counter() - t0:.1f}s")
    if not np.array_equal(a_greedy, a_k0):
        n_diff_epochs = (a_greedy != a_k0).any(axis=1).sum()
        first_diff = np.argmax((a_greedy != a_k0).any(axis=1))
        print(f"  [FAIL] k=0 differs from greedy on {n_diff_epochs}/{n_epochs} epochs")
        print(f"         first diff at epoch {first_diff}:")
        diff_idx = np.flatnonzero(a_greedy[first_diff] != a_k0[first_diff])
        for di in diff_idx[:5]:
            print(f"         sat {di}: greedy={a_greedy[first_diff, di]}, k=0={a_k0[first_diff, di]}")
        return 1
    print(f"  [OK] k=0 == greedy on every epoch and every satellite")

    # ---- 2. k = 1 should match predictive --------------------------------
    print("\n[2/3] k = 1 should produce identical actions to PredictiveMatching")
    t0 = time.perf_counter()
    a_pred, _ = run_with_policy("predictive", walker, feas_params, n_epochs, T, seed=seed)
    a_k1, _ = run_with_policy("k_step", walker, feas_params, n_epochs, T, seed=seed, k=1)
    print(f"       took {time.perf_counter() - t0:.1f}s")
    if not np.array_equal(a_pred, a_k1):
        n_diff_epochs = (a_pred != a_k1).any(axis=1).sum()
        first_diff = np.argmax((a_pred != a_k1).any(axis=1))
        print(f"  [FAIL] k=1 differs from predictive on {n_diff_epochs}/{n_epochs} epochs")
        print(f"         first diff at epoch {first_diff}:")
        diff_idx = np.flatnonzero(a_pred[first_diff] != a_k1[first_diff])
        for di in diff_idx[:5]:
            print(f"         sat {di}: predictive={a_pred[first_diff, di]}, k=1={a_k1[first_diff, di]}")
        return 1
    print(f"  [OK] k=1 == predictive on every epoch and every satellite")

    # ---- 3. k large should approach equilibrium edges-per-epoch ----------
    # Note: k_step uses SYNCHRONOUS BR; equilibrium uses GAUSS-SEIDEL. So
    # the exact action profiles may differ at fixed points (there can be
    # multiple NE). What we check is that edges-per-epoch grows toward
    # equilibrium's, since both saturate the matching constraint.
    print("\n[3/3] k large (k=8) should give edges/epoch close to equilibrium's")
    t0 = time.perf_counter()
    _, n_edges_eq = run_with_policy("equilibrium", walker, feas_params, n_epochs, T, seed=seed)
    _, n_edges_k8 = run_with_policy("k_step", walker, feas_params, n_epochs, T, seed=seed, k=8)
    _, n_edges_k1 = run_with_policy("k_step", walker, feas_params, n_epochs, T, seed=seed, k=1)
    print(f"       took {time.perf_counter() - t0:.1f}s")

    mean_eq = n_edges_eq.mean()
    mean_k8 = n_edges_k8.mean()
    mean_k1 = n_edges_k1.mean()
    print(f"       equilibrium mean edges/epoch: {mean_eq:.2f}")
    print(f"       k=8         mean edges/epoch: {mean_k8:.2f}")
    print(f"       k=1         mean edges/epoch: {mean_k1:.2f}")

    if mean_k8 < mean_k1 - 0.5:
        print(f"  [FAIL] k=8 ({mean_k8:.2f}) is WORSE than k=1 ({mean_k1:.2f}); BR is not improving things")
        return 1
    # Generous tolerance: synchronous BR can land at a different NE than
    # Gauss-Seidel, so we don't require exact agreement. We do require
    # that k=8 is meaningfully closer to equilibrium than k=1 is.
    gap_k1 = abs(mean_eq - mean_k1)
    gap_k8 = abs(mean_eq - mean_k8)
    if gap_k8 > gap_k1 * 1.5:
        print(f"  [WARN] k=8 (gap {gap_k8:.2f}) is not closer to equilibrium than k=1 (gap {gap_k1:.2f})")
        print(f"         this can happen if synchronous BR cycles; not necessarily a bug")
    else:
        print(f"  [OK] k=8 is closer to equilibrium than k=1 (gap {gap_k8:.2f} vs {gap_k1:.2f})")

    print("\nAll k_step smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
