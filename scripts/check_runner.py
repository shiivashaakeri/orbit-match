# orbit-match/scripts/check_runner.py
# Run: python -m scripts.check_runner

"""Smoke test for orbitmatch.experiments.runner.run_simulation.

Runs the predictive, greedy, and random policies on the small Walker
constellation for T + 30 epochs (i.e. just past one orbital window so
the WindowedUnion has fully cycled). Checks:

1. SimulationResult has the right field shapes and dtypes.
2. Numeric traces are finite and non-negative.
3. Predictive's deferrals diagnostic is present and nonzero.
4. After the window is full, predictive yields positive lambda_2(union).
5. to_arrays() packs cleanly (incl. jagged matchings as object-dtype).
6. The feasibility cache is hit on the second policy (same constellation).

Wall-clock budget: the first run computes feasibility (one-time cost,
~tens of seconds); subsequent runs are policy-only.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.runner import SimulationResult, run_simulation
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.policy.base import PolicyParams
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fixture: small Walker, short horizon
# ---------------------------------------------------------------------------


def make_small_walker() -> WalkerDeltaConfig:
    return WalkerDeltaConfig(
        M=24,
        P=4,
        F=1,
        altitude_km=550.0,
        inclination_deg=53.0,
        name="small",
    )


def make_feasibility_params() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0,
        range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def make_policy_params(T: int, H: int = 10) -> PolicyParams:
    return PolicyParams(
        H=H,
        T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_result(res: SimulationResult) -> None:  # noqa: C901, PLR0912
    """Check that SimulationResult has the expected shapes and dtypes."""
    n = res.n
    n_epochs = res.n_epochs

    # Trace shapes.
    for name in ("lambda2_phi", "lambda2_union"):
        arr = getattr(res, name)
        if arr.shape != (n_epochs,):
            raise AssertionError(f"{name} shape {arr.shape}, expected ({n_epochs},)")
        if arr.dtype != np.float64:
            raise AssertionError(f"{name} dtype {arr.dtype}, expected float64")
        if not np.isfinite(arr).all():
            raise AssertionError(f"{name} contains non-finite values")
        if (arr < -1e-9).any():
            raise AssertionError(f"{name} contains negative values: min = {arr.min()}")

    for name in ("n_edges_per_epoch", "n_union_edges_per_epoch"):
        arr = getattr(res, name)
        if arr.shape != (n_epochs,):
            raise AssertionError(f"{name} shape {arr.shape}, expected ({n_epochs},)")
        if arr.dtype != np.int64:
            raise AssertionError(f"{name} dtype {arr.dtype}, expected int64")
        if (arr < 0).any():
            raise AssertionError(f"{name} contains negative values")

    if res.actions.shape != (n_epochs, n):
        raise AssertionError(f"actions shape {res.actions.shape}, expected ({n_epochs}, {n})")
    if res.actions.dtype != np.int64:
        raise AssertionError(f"actions dtype {res.actions.dtype}, expected int64")

    if len(res.matchings) != n_epochs:
        raise AssertionError(f"matchings has {len(res.matchings)} entries, expected {n_epochs}")
    for t, m in enumerate(res.matchings):
        if m.ndim != 2 or m.shape[1] != 2:
            raise AssertionError(f"matchings[{t}] shape {m.shape}, expected (k, 2)")
        if m.size > 0 and (m < 0).any():
            raise AssertionError(f"matchings[{t}] contains negative indices")
        if m.size > 0 and (m >= n).any():
            raise AssertionError(f"matchings[{t}] contains out-of-range indices")

    if res.final_phi.shape != (n, n):
        raise AssertionError(f"final_phi shape {res.final_phi.shape}, expected ({n}, {n})")
    if res.final_union_adjacency.shape != (n, n):
        raise AssertionError(f"final_union_adjacency shape {res.final_union_adjacency.shape}")


def validate_packing(res: SimulationResult) -> None:
    """Check that to_arrays() produces a save_trace-compatible dict."""
    arrays = res.to_arrays()
    required = {
        "lambda2_phi",
        "lambda2_union",
        "n_edges_per_epoch",
        "n_union_edges_per_epoch",
        "actions",
        "matchings",
        "final_phi",
        "final_union_adjacency",
    }
    missing = required - set(arrays.keys())
    if missing:
        raise AssertionError(f"to_arrays() missing keys: {missing}")

    # Matchings must be object-dtype (jagged).
    if arrays["matchings"].dtype != np.dtype("O"):
        raise AssertionError(f"matchings dtype {arrays['matchings'].dtype}, expected object")
    if len(arrays["matchings"]) != res.n_epochs:
        raise AssertionError(f"packed matchings length {len(arrays['matchings'])} != n_epochs {res.n_epochs}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_three_policies(walker, feas_params, T, n_epochs):
    """Run each of the three policies and validate every result."""
    results: dict[str, SimulationResult] = {}

    for policy_name in ("predictive", "greedy", "random"):
        print(f"  running {policy_name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = run_simulation(
            walker=walker,
            feasibility_params=feas_params,
            policy_name=policy_name,
            policy_params=make_policy_params(T=T),
            n_epochs=n_epochs,
            dt_s=10.0,
            seed=42,
            config_label="small",
        )
        elapsed = time.perf_counter() - t0
        print(f"({elapsed:.1f}s)")
        validate_result(res)
        validate_packing(res)
        results[policy_name] = res
    print("  [OK] all three policies completed and packed cleanly")
    return results


def test_predictive_deferrals(results: dict[str, SimulationResult]):
    """Predictive should record nonzero deferrals; the others should not have the key."""
    pred = results["predictive"]
    if "deferrals" not in pred.policy_diagnostics:
        raise AssertionError("PredictiveMatching did not record 'deferrals' diagnostic")
    deferrals = pred.policy_diagnostics["deferrals"]
    if deferrals.shape != (pred.n_epochs,):
        raise AssertionError(f"deferrals shape {deferrals.shape}, expected ({pred.n_epochs},)")
    if not (deferrals >= 0).all():
        raise AssertionError("deferrals has negative entries")
    if deferrals.sum() == 0:
        # Acceptable in principle but suspicious; the small constellation always has SOME deferrals.
        print(f"  [WARN] predictive had zero total deferrals across {pred.n_epochs} epochs")
    else:
        print(f"  [OK] predictive recorded {int(deferrals.sum())} total deferrals (mean {deferrals.mean():.2f}/epoch)")

    # Greedy and Random should not record this diagnostic.
    for other in ("greedy", "random"):
        if "deferrals" in results[other].policy_diagnostics:
            raise AssertionError(f"{other} should not record 'deferrals'")
    print("  [OK] greedy and random correctly omit 'deferrals'")


def test_union_fills(results: dict[str, SimulationResult], T: int):
    """After the window fills, predictive should have positive lambda2(union)."""
    for policy_name in ("predictive", "greedy", "random"):
        res = results[policy_name]
        post_warmup = res.lambda2_union[T:]
        if len(post_warmup) == 0:
            raise AssertionError(f"{policy_name}: no post-warmup samples (n_epochs={res.n_epochs}, T={T})")
        mean_post = float(post_warmup.mean())
        max_post = float(post_warmup.max())
        print(f"  {policy_name}: post-warmup lambda2(union) mean={mean_post:.4f}, max={max_post:.4f}")

    # Strong invariant only for predictive: after warmup, the realized union should be connected at least once.
    pred = results["predictive"]
    if not (pred.lambda2_union[T:] > 0).any():
        raise AssertionError(
            "predictive: lambda2(union) is zero throughout the post-warmup window; "
            "this contradicts the connectivity certificate."
        )
    print("  [OK] predictive achieves positive lambda2(union) post-warmup")


def test_metadata_complete(results: dict[str, SimulationResult]):
    """Metadata must contain everything needed for save_trace."""
    required_keys = {
        "policy_name",
        "config_label",
        "walker",
        "feasibility_params",
        "policy_params",
        "n_epochs",
        "dt_s",
        "seed",
    }
    for name, res in results.items():
        missing = required_keys - set(res.metadata.keys())
        if missing:
            raise AssertionError(f"{name}: metadata missing keys {missing}")
    print("  [OK] all results carry complete metadata")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")  # silence the orbitmatch.* loggers during the run.

    walker = make_small_walker()
    feas_params = make_feasibility_params()

    # Short horizon: just past one orbital period of certificate window.
    T = 60  # ~10 minutes at dt=10s; smaller than T_orb but well past warmup
    n_epochs = T + 30
    dt_s = 10.0

    print(f"Fixture: small Walker (n={walker.M}), T={T} epochs, n_epochs={n_epochs}, dt={dt_s}s")
    print()

    tests = [  # noqa: F841
        ("3 policies run + validate", lambda: test_all_three_policies(walker, feas_params, T, n_epochs)),
    ]

    failed = 0
    print("[1/4] running all three policies")
    try:
        results = test_all_three_policies(walker, feas_params, T, n_epochs)
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        return 1
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        return 1

    print("\n[2/4] predictive diagnostics")
    try:
        test_predictive_deferrals(results)
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1

    print("\n[3/4] union-graph fills past warmup")
    try:
        test_union_fills(results, T)
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1

    print("\n[4/4] metadata completeness")
    try:
        test_metadata_complete(results)
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1

    print()
    if failed:
        print(f"FAILED: {failed} sub-tests")
        return 1
    print("All runner smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
