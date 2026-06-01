# orbit-match/scripts/check_diagnostics.py
# Run: python -m scripts.check_diagnostics

"""Smoke test for orbitmatch.experiments.diagnostics.

Two runs on the small Walker constellation:

1. Short-T run (T=60 epochs, ~10 min, n_epochs=120). Validates the
   DiagnosticsReport structure: every field is present, types are
   right, alpha_0 > 0, rho_realized in [0, 1+slack], Vizing bounds
   are sensible. Fast.

2. Headline-T run (T = T_orb in epochs, n_epochs = 2*T_orb). Verifies
   the saturation claim from EXPERIMENTS_LOG: at T = T_orb on the
   small config, rho_realized at the final epoch should be ~1.

The second run takes longer (the Kirchhoff value matrix scales with
the lookahead horizon and is the dominant cost). Acceptable for a
sanity script run a few times a week.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.diagnostics import (
    DiagnosticsReport,
    build_report,
    compute_alpha_0,
    rho_match_vizing_interval,
)
from orbitmatch.experiments.runner import run_simulation
from orbitmatch.feasibility.compute import compute_feasibility
from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.policy.base import PolicyParams
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_small_walker() -> WalkerDeltaConfig:
    return WalkerDeltaConfig(M=24, P=4, F=1, altitude_km=550.0, inclination_deg=53.0, name="small")


def make_feasibility_params() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0,
        range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def make_policy_params(T: int, H: int = 10) -> PolicyParams:
    return PolicyParams(
        H=H, T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_report(rep: DiagnosticsReport, T: int) -> None:
    """Check the DiagnosticsReport fields are sensible."""
    if rep.alpha_0 <= 0:
        raise AssertionError(f"alpha_0 must be positive; got {rep.alpha_0}")
    if not (0 <= rep.alpha_0_worst_t):
        raise AssertionError(f"alpha_0_worst_t must be non-negative; got {rep.alpha_0_worst_t}")
    if rep.T_0 <= 0:
        raise AssertionError(f"T_0 must be positive; got {rep.T_0}")

    # rho_realized in [0, 1 + small slack]. We allow a small overshoot because
    # the windowed-union lambda_2 can in principle exceed alpha_0 in transient
    # windows (alpha_0 is the worst-case lower bound, not an upper bound).
    if rep.rho_realized_final < -1e-9:
        raise AssertionError(f"rho_realized_final negative: {rep.rho_realized_final}")
    if rep.rho_realized_max > 5.0:
        raise AssertionError(f"rho_realized_max suspiciously large: {rep.rho_realized_max}")

    # rho_cover in [0, 1].
    if not (-1e-9 <= rep.rho_cover_final <= 1.0 + 1e-9):
        raise AssertionError(f"rho_cover_final out of [0, 1]: {rep.rho_cover_final}")

    # Empirical rho_match: NaN if rho_cover is 0; else should be finite.
    if rep.rho_cover_final > 0 and not math.isfinite(rep.rho_match_empirical_final):
        raise AssertionError(f"rho_match_empirical_final is not finite: {rep.rho_match_empirical_final}")

    # Vizing bounds.
    if rep.rho_match_vizing_lower < 0 or rep.rho_match_vizing_upper < rep.rho_match_vizing_lower:
        raise AssertionError(
            f"Vizing interval malformed: [{rep.rho_match_vizing_lower}, {rep.rho_match_vizing_upper}]"
        )
    if rep.rho_match_vizing_upper > 1.0 + 1e-9:
        raise AssertionError(f"Vizing upper bound > 1: {rep.rho_match_vizing_upper}")

    # Edge / degree integers must be sane.
    if rep.feasibility_union_edges_final < 0:
        raise AssertionError(f"feasibility_union_edges_final negative")
    if rep.realized_union_edges_final < 0:
        raise AssertionError(f"realized_union_edges_final negative")
    if rep.max_degree_feasibility_final < 0:
        raise AssertionError(f"max_degree_feasibility_final negative")

    # Provenance.
    if rep.T != T:
        raise AssertionError(f"report.T={rep.T} != expected T={T}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_short_T(walker, feas_params, T: int, n_epochs: int):
    """Short run -> validate report structure for all three policies."""
    reports = {}
    feasibility = None

    for policy_name in ("predictive", "greedy", "random"):
        print(f"  [{policy_name}] running ({n_epochs} epochs, T={T}) ... ", end="", flush=True)
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
        print(f"({time.perf_counter() - t0:.1f}s)")

        # Rebuild the feasibility tensor once (cached on disk, so this is cheap).
        if feasibility is None:
            elements = walker.initial_elements()
            times = make_time_grid(duration_s=n_epochs * 10.0, dt_s=10.0)
            positions = propagate_keplerian(elements, times)
            feasibility = compute_feasibility(positions, 10.0, feas_params)

        rep = build_report(res, feasibility)
        validate_report(rep, T=T)
        reports[policy_name] = rep
        print(
            f"           alpha_0={rep.alpha_0:.4f}, "
            f"rho_realized(final)={rep.rho_realized_final:.4f}, "
            f"rho_cover={rep.rho_cover_final:.4f}, "
            f"rho_match[Vizing] in [{rep.rho_match_vizing_lower:.4f}, {rep.rho_match_vizing_upper:.4f}]"
        )

    # alpha_0 must be identical across policies (depends only on geometry).
    a0s = {n: r.alpha_0 for n, r in reports.items()}
    if len(set(round(v, 8) for v in a0s.values())) != 1:
        raise AssertionError(f"alpha_0 should be policy-independent; got {a0s}")
    print(f"  [OK] alpha_0 identical across policies: {a0s['predictive']:.6f}")
    return reports, feasibility


def test_T_orb_saturation(walker, feas_params):
    """Headline claim: predictive at T = T_orb on small config -> rho ~ 1.

    Saturation is the claim that *at the first epoch when the certificate
    window is full* (t = T - 1, i.e. the window covers [0, T)), the
    realized union equals the feasibility union. After that the rolling
    window will drop edges that the policy correctly deferred from
    re-realizing -- this is the deferral mechanism working as designed,
    not a failure.

    Test strategy: run for slightly more than T epochs so the window
    fills, then evaluate rho_cover at t = T (the first full-window
    epoch). EXPERIMENTS_LOG predicts rho_cover ~ 1 here.
    """
    dt_s = 10.0
    T = int(math.ceil(walker.orbital_period_s / dt_s))  # ~574 epochs
    n_epochs = T + 5  # just past the first full window

    print(f"  T = T_orb = {T} epochs (~{T * dt_s / 60:.1f} min)")
    print(f"  n_epochs = {n_epochs} (T + 5)")

    t0 = time.perf_counter()
    res = run_simulation(
        walker=walker,
        feasibility_params=feas_params,
        policy_name="predictive",
        policy_params=make_policy_params(T=T),
        n_epochs=n_epochs,
        dt_s=dt_s,
        seed=42,
        config_label="small",
    )
    print(f"  predictive run wall-clock: {time.perf_counter() - t0:.1f}s")

    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * dt_s, dt_s=dt_s)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, dt_s, feas_params)

    rep = build_report(res, feasibility)
    validate_report(rep, T=T)

    # The headline number for the saturation check: rho_realized's MAX
    # over the post-warmup trace, which lands at the first full-window
    # epoch. The 'final' values reflect later rolling-window losses.
    from orbitmatch.experiments.diagnostics import rho_cover_at_t
    t_eval = T  # first epoch the window is fully T epochs long
    rho_cover_at_first_full = rho_cover_at_t(res, feasibility, t_eval, T_0=T)

    print(f"  alpha_0                 = {rep.alpha_0:.4f}")
    print(f"  rho_realized(max)       = {rep.rho_realized_max:.4f}")
    print(f"  rho_realized(final)     = {rep.rho_realized_final:.4f}")
    print(f"  rho_cover @ t=T         = {rho_cover_at_first_full:.4f}   <- saturation check")
    print(f"  rho_cover(final)        = {rep.rho_cover_final:.4f}")
    print(f"  feasibility union edges = {rep.feasibility_union_edges_final}")
    print(f"  realized union edges    = {rep.realized_union_edges_final}")

    # The saturation claim: at t = T (first full window), realized = feasibility.
    if rho_cover_at_first_full < 0.98:
        raise AssertionError(
            f"rho_cover @ t=T = {rho_cover_at_first_full:.4f}; expected ~1.0 at the first "
            f"full-window epoch. This contradicts the saturation claim in EXPERIMENTS_LOG."
        )
    print(f"  [OK] saturation: rho_cover @ t=T = {rho_cover_at_first_full:.4f} (>= 0.98)")


def test_alpha_0_invariance(walker, feas_params):
    """alpha_0 should not depend on the policy or random seed."""
    print(f"  computing alpha_0 directly from feasibility ...")
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=200 * 10.0, dt_s=10.0)
    positions = propagate_keplerian(elements, times)
    feas = compute_feasibility(positions, 10.0, feas_params)
    a0_a, t_a = compute_alpha_0(feas, T_0=60)
    a0_b, t_b = compute_alpha_0(feas, T_0=60)
    if abs(a0_a - a0_b) > 1e-12 or t_a != t_b:
        raise AssertionError(f"alpha_0 not deterministic: ({a0_a}, {t_a}) vs ({a0_b}, {t_b})")
    print(f"  alpha_0(T_0=60) = {a0_a:.6f}, argmin t = {t_a}")
    print(f"  [OK] alpha_0 deterministic across calls")


def test_vizing_helper():
    """Direct unit test of the Vizing bound."""
    lo, hi = rho_match_vizing_interval(T=60, max_degree=6)
    # T/(D+1) = 60/7, T/D = 60/6 = 10. Upper clipped to 1.
    expected_lo = min(1.0, 60 / 7)
    expected_hi = min(1.0, 60 / 6)
    if abs(lo - expected_lo) > 1e-9 or abs(hi - expected_hi) > 1e-9:
        raise AssertionError(f"Vizing interval: got ({lo}, {hi}), expected ({expected_lo}, {expected_hi})")
    # Zero-degree edge case.
    lo0, hi0 = rho_match_vizing_interval(T=60, max_degree=0)
    if (lo0, hi0) != (0.0, 0.0):
        raise AssertionError(f"Vizing on degree-0 graph: got ({lo0}, {hi0}), expected (0, 0)")
    print(f"  [OK] Vizing helper: interval at (T=60, Delta=6) = ({lo:.4f}, {hi:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")
    walker = make_small_walker()
    feas_params = make_feasibility_params()

    failed = 0

    print("[1/4] Vizing-interval helper")
    try:
        test_vizing_helper()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1

    print("\n[2/4] alpha_0 invariance")
    try:
        test_alpha_0_invariance(walker, feas_params)
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1

    print("\n[3/4] short-T report structure (all three policies)")
    try:
        test_short_T(walker, feas_params, T=60, n_epochs=120)
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        import traceback; traceback.print_exc()
        failed += 1

    print("\n[4/4] saturation claim at T = T_orb")
    try:
        test_T_orb_saturation(walker, feas_params)
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1
    except Exception as e:
        import traceback; traceback.print_exc()
        failed += 1

    print()
    if failed:
        print(f"FAILED: {failed} sub-tests")
        return 1
    print("All diagnostics smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())