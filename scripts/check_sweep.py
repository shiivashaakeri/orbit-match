# orbit-match/scripts/check_sweep.py
# Run: python -m scripts.check_sweep

"""Smoke test for orbitmatch.experiments.sweep.

Exercises:

1. expand_overrides + override_label work on H-axis and walker-axis sweeps.
2. _apply_override correctly modifies (walker, policy_params, n_epochs).
3. A tiny end-to-end sweep (2 H values x 2 seeds = 4 jobs) on the small
   config completes and returns 4 SweepRecords.
4. Resumability: re-running the same sweep loads from disk; all records
   have skipped_existing=True the second time.
5. Reports reconstructed from disk match what build_report produced live.

Uses a temp directory for the results so we don't pollute results/.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.sweep import (
    SweepRecord,
    _apply_override,
    expand_overrides,
    override_label,
    run_sweep,
)
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.policy.base import PolicyParams
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_walker() -> WalkerDeltaConfig:
    return WalkerDeltaConfig(M=24, P=4, F=1, altitude_km=550.0, inclination_deg=53.0, name="small")


def make_feas() -> FeasibilityParams:
    return FeasibilityParams(
        atm_buffer_km=80.0,
        range_max_km=8000.0,
        rate_max_rad_per_s=float(np.deg2rad(1.0)),
    )


def make_pp(T: int = 60) -> PolicyParams:
    return PolicyParams(
        H=10,
        T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_expand_and_label() -> None:
    """expand_overrides + override_label produce the right strings."""
    ovs = expand_overrides({"H": [1, 5, 10]})
    if len(ovs) != 3:
        raise AssertionError(f"expand_overrides len {len(ovs)}, expected 3")
    if ovs[0] != {"H": 1} or ovs[1] != {"H": 5} or ovs[2] != {"H": 10}:
        raise AssertionError(f"expand_overrides values wrong: {ovs}")
    print(f"  [OK] expand_overrides({{H: [1, 5, 10]}}) -> {ovs}")

    h, c = override_label({"H": 10})
    if h != "H=10" or c != "H_10":
        raise AssertionError(f"override_label({{H:10}}) = ({h!r}, {c!r}), expected ('H=10', 'H_10')")

    # Walker-axis label uses M.
    w = WalkerDeltaConfig(M=48, P=6, F=2, altitude_km=550.0, inclination_deg=53.0, name="scaled")
    h2, c2 = override_label({"walker": w})
    if h2 != "n=48" or c2 != "n_48":
        raise AssertionError(f"override_label({{walker:n=48}}) = ({h2!r}, {c2!r}), expected ('n=48', 'n_48')")
    print(f"  [OK] override_label handles H and walker axes")


def test_apply_override() -> None:
    """_apply_override modifies the right fields and leaves others alone."""
    walker = make_walker()
    pp = make_pp(T=60)
    n_epochs = 120

    # H override.
    w1, pp1, n1 = _apply_override(
        base_walker=walker,
        base_policy_params=pp,
        base_n_epochs=n_epochs,
        override={"H": 5},
    )
    if pp1.H != 5 or pp1.T != 60:
        raise AssertionError(f"H override: pp.H={pp1.H}, pp.T={pp1.T}; expected H=5, T=60")
    if w1 is not walker:
        raise AssertionError(f"H override should not change walker")

    # Walker override.
    w_new = WalkerDeltaConfig(M=12, P=2, F=1, altitude_km=550.0, inclination_deg=53.0, name="tiny")
    w2, pp2, n2 = _apply_override(
        base_walker=walker,
        base_policy_params=pp,
        base_n_epochs=n_epochs,
        override={"walker": w_new},
    )
    if w2.M != 12:
        raise AssertionError(f"walker override: M={w2.M}, expected 12")
    if pp2.H != pp.H or pp2.T != pp.T:
        raise AssertionError(f"walker override should not change policy params")

    # n_epochs override.
    _, _, n3 = _apply_override(
        base_walker=walker,
        base_policy_params=pp,
        base_n_epochs=n_epochs,
        override={"n_epochs": 200},
    )
    if n3 != 200:
        raise AssertionError(f"n_epochs override: got {n3}, expected 200")

    # Unsupported key.
    try:
        _apply_override(
            base_walker=walker,
            base_policy_params=pp,
            base_n_epochs=n_epochs,
            override={"bogus": 1},
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Unsupported key should raise KeyError")

    print(f"  [OK] _apply_override handles H, walker, n_epochs, and rejects unknowns")


def test_tiny_sweep(tmp_results: Path) -> list[SweepRecord]:
    """End-to-end: 2 H values x 2 seeds = 4 records, all new."""
    walker = make_walker()
    feas = make_feas()
    pp = make_pp(T=60)

    t0 = time.perf_counter()
    records = run_sweep(
        base_walker=walker,
        feasibility_params=feas,
        base_policy_name="predictive",
        base_policy_params=pp,
        base_n_epochs=120,
        dt_s=10.0,
        overrides=expand_overrides({"H": [5, 20]}),
        seeds=[42, 43],
        config_label="small_check",
        results_dir=tmp_results,
        force=False,
        share_alpha_0=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [OK] 4 jobs completed in {elapsed:.1f}s")

    if len(records) != 4:
        raise AssertionError(f"Expected 4 records, got {len(records)}")
    if any(r.skipped_existing for r in records):
        raise AssertionError(f"First pass should run everything fresh; some were skipped")

    # Records should cover the Cartesian product.
    pairs = {(r.override_key, r.seed) for r in records}
    expected = {("H_5", 42), ("H_5", 43), ("H_20", 42), ("H_20", 43)}
    if pairs != expected:
        raise AssertionError(f"pairs {pairs} != expected {expected}")

    # Files should exist on disk.
    for r in records:
        if not r.trace_path.exists():
            raise AssertionError(f"Trace not saved: {r.trace_path}")
    print(f"  [OK] all 4 trace files present on disk")

    # Each record carries a populated report.
    for r in records:
        if r.report.alpha_0 <= 0:
            raise AssertionError(f"Record {r.override_label}/seed={r.seed} has alpha_0={r.report.alpha_0}")
        if r.report.rho_realized_final < 0:
            raise AssertionError(f"Record has negative rho_realized_final")
    print(f"  [OK] all 4 records carry valid DiagnosticsReports")

    # Show the numbers.
    for r in records:
        print(f"      {r.override_label:8s} seed={r.seed}: rho_realized(max)={r.report.rho_realized_max:.4f}")

    return records


def test_resumability(tmp_results: Path, first_run: list[SweepRecord]) -> None:
    """Re-running the same sweep loads from disk; everything is skipped_existing."""
    walker = make_walker()
    feas = make_feas()
    pp = make_pp(T=60)

    t0 = time.perf_counter()
    records = run_sweep(
        base_walker=walker,
        feasibility_params=feas,
        base_policy_name="predictive",
        base_policy_params=pp,
        base_n_epochs=120,
        dt_s=10.0,
        overrides=expand_overrides({"H": [5, 20]}),
        seeds=[42, 43],
        config_label="small_check",
        results_dir=tmp_results,
        force=False,
        share_alpha_0=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [OK] second pass completed in {elapsed:.1f}s")

    if len(records) != 4:
        raise AssertionError(f"Second pass: expected 4 records, got {len(records)}")
    if not all(r.skipped_existing for r in records):
        not_skipped = [(r.override_label, r.seed) for r in records if not r.skipped_existing]
        raise AssertionError(f"Second pass should skip all jobs; these were re-run: {not_skipped}")
    print(f"  [OK] all 4 records loaded from disk (skipped_existing=True)")

    # The reports from cache should match the original (within float precision).
    for fresh, cached in zip(first_run, records):
        if abs(fresh.report.alpha_0 - cached.report.alpha_0) > 1e-9:
            raise AssertionError(f"alpha_0 differs after roundtrip: {fresh.report.alpha_0} vs {cached.report.alpha_0}")
        if abs(fresh.report.rho_realized_final - cached.report.rho_realized_final) > 1e-9:
            raise AssertionError(
                f"rho_realized_final differs after roundtrip: "
                f"{fresh.report.rho_realized_final} vs {cached.report.rho_realized_final}"
            )
    print(f"  [OK] reports match original within float precision")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")

    failed = 0

    print("[1/4] expand_overrides + override_label")
    try:
        test_expand_and_label()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1

    print("\n[2/4] _apply_override")
    try:
        test_apply_override()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # Use a temp dir so we don't pollute results/.
    tmp_results = Path(tempfile.mkdtemp(prefix="orbitmatch_sweep_check_"))
    try:
        print(f"\n[3/4] tiny end-to-end sweep -> {tmp_results}")
        try:
            first_run = test_tiny_sweep(tmp_results)
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
            first_run = []
        except Exception as e:
            import traceback

            traceback.print_exc()
            failed += 1
            first_run = []

        if first_run:
            print(f"\n[4/4] resumability (re-run, expect all skipped_existing=True)")
            try:
                test_resumability(tmp_results, first_run)
            except AssertionError as e:
                print(f"  [FAIL] {e}")
                failed += 1
            except Exception as e:
                import traceback

                traceback.print_exc()
                failed += 1
    finally:
        shutil.rmtree(tmp_results)

    print()
    if failed:
        print(f"FAILED: {failed} sub-tests")
        return 1
    print("All sweep smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
