# orbit-match/scripts/run_sanity_checks.py
# Run: python -m scripts.run_sanity_checks

"""Production sanity checks (S1-S5 from PROJECT_PLAN.md Sec 4.1).

Runs before any paper experiment. Different from the per-module
check_*.py scripts (which test code correctness): this script
validates the *modeling assumptions and policy behavior* on the
canonical small constellation.

The five checks
---------------
S1. Feasibility-union graph is connected: lambda_2(L_union_F(t; T_0)) > 0
    for all t, with T_0 = T_orb.
S2. alpha_0 computed and saved to results/diagnostics/alpha_0.npz.
S3. Deferral mechanism fires on >= 5% of epochs in a predictive run.
S4. Best-response dynamics is monotone: W_t non-decreasing across
    rounds in EquilibriumMatching.
S5. Single-orbit visualization saved to figures/diagnostics/orbit_*.pdf.
    Done both for small (smoke) and medium (real) configs.

Exit status
-----------
Returns 0 if every check passes, 1 otherwise. Designed for CI use.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.diagnostics import compute_alpha_0
from orbitmatch.experiments.runner import run_simulation
from orbitmatch.feasibility.compute import compute_feasibility, feasibility_union
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.graph.laplacian import adjacency_to_laplacian
from orbitmatch.graph.spectral import lambda_2
from orbitmatch.graph.windowed import WindowedLaplacian
from orbitmatch.policy.base import PolicyParams
from orbitmatch.policy.equilibrium import EquilibriumMatching
from orbitmatch.plotting.diagnostic_plots import plot_orbit_projection
from orbitmatch.plotting.theme import apply_theme
from orbitmatch.utils.io import figures_dir, results_dir, save_trace
from orbitmatch.utils.logging_setup import configure, get_logger
from orbitmatch.utils.seeding import make_rng

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------


DT_S = 10.0
S3_DEFERRAL_THRESHOLD = 0.05  # 5% of epochs


def make_walker(label: str) -> WalkerDeltaConfig:
    """Return the small or medium Walker config."""
    if label == "small":
        return WalkerDeltaConfig(M=24, P=4, F=1, altitude_km=550.0, inclination_deg=53.0, name="small")
    if label == "medium":
        return WalkerDeltaConfig(M=60, P=6, F=2, altitude_km=550.0, inclination_deg=53.0, name="medium")
    raise ValueError(f"Unknown label {label!r}")


def make_feas_params() -> FeasibilityParams:
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


def build_feasibility(walker: WalkerDeltaConfig, n_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, feasibility) for a given walker and horizon."""
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * DT_S, dt_s=DT_S)
    positions = propagate_keplerian(elements, times)
    feasibility = compute_feasibility(positions, DT_S, make_feas_params())
    return positions, feasibility


# ---------------------------------------------------------------------------
# S1: feasibility-union connected
# ---------------------------------------------------------------------------


def s1_feasibility_union_connected(label: str) -> tuple[bool, dict]:
    walker = make_walker(label)
    T_orb_epochs = int(math.ceil(walker.orbital_period_s / DT_S))
    n_epochs = T_orb_epochs + 30  # one full window plus a margin
    _, feasibility = build_feasibility(walker, n_epochs)

    n = walker.M
    # Check lambda_2 of L_union_F(t; T_0) at every valid starting epoch.
    lam_min = math.inf
    lam_argmin_t = 0
    n_windows = n_epochs - T_orb_epochs + 1
    if n_windows <= 0:
        return False, {"reason": "no valid windows", "T_0": T_orb_epochs, "n_epochs": n_epochs}

    for t in range(n_windows):
        adj = feasibility_union(feasibility, window_start=t, window_length=T_orb_epochs)
        L = adjacency_to_laplacian(adj.astype(np.float64))
        lam = lambda_2(L)
        if lam < lam_min:
            lam_min = lam
            lam_argmin_t = t

    passed = lam_min > 0.0
    return passed, {
        "label": label,
        "n": n,
        "T_0_epochs": T_orb_epochs,
        "n_windows_checked": n_windows,
        "lambda_2_min": float(lam_min),
        "lambda_2_argmin_t": int(lam_argmin_t),
    }


# ---------------------------------------------------------------------------
# S2: alpha_0 computed and saved
# ---------------------------------------------------------------------------


def s2_compute_alpha_0(label: str, out_dir: Path) -> tuple[bool, dict]:
    walker = make_walker(label)
    T_orb_epochs = int(math.ceil(walker.orbital_period_s / DT_S))
    n_epochs = T_orb_epochs + 30
    _, feasibility = build_feasibility(walker, n_epochs)

    alpha_0, t_worst = compute_alpha_0(feasibility, T_0=T_orb_epochs)
    out = out_dir / f"alpha_0_{label}.npz"
    save_trace(
        out,
        arrays={"alpha_0": np.array([alpha_0]), "t_worst": np.array([t_worst])},
        metadata={
            "label": label,
            "T_0_epochs": T_orb_epochs,
            "n_epochs": n_epochs,
            "walker": {
                "M": walker.M,
                "P": walker.P,
                "F": walker.F,
                "altitude_km": walker.altitude_km,
                "inclination_deg": walker.inclination_deg,
            },
        },
    )

    passed = alpha_0 > 0.0
    return passed, {
        "label": label,
        "alpha_0": float(alpha_0),
        "t_worst": int(t_worst),
        "saved_to": str(out),
    }


# ---------------------------------------------------------------------------
# S3: deferral mechanism fires
# ---------------------------------------------------------------------------


def s3_deferral_fires() -> tuple[bool, dict]:
    walker = make_walker("small")
    T_orb_epochs = int(math.ceil(walker.orbital_period_s / DT_S))
    n_epochs = T_orb_epochs + 30

    res = run_simulation(
        walker=walker,
        feasibility_params=make_feas_params(),
        policy_name="predictive",
        policy_params=make_policy_params(T=T_orb_epochs),
        n_epochs=n_epochs,
        dt_s=DT_S,
        seed=42,
        config_label="small",
    )

    deferrals = res.policy_diagnostics.get("deferrals")
    if deferrals is None:
        return False, {"reason": "predictive run did not record 'deferrals' diagnostic"}

    fired_epochs = int((deferrals > 0).sum())
    fired_frac = fired_epochs / len(deferrals)
    passed = fired_frac >= S3_DEFERRAL_THRESHOLD

    return passed, {
        "n_epochs": int(len(deferrals)),
        "fired_epochs": fired_epochs,
        "fired_frac": float(fired_frac),
        "threshold": S3_DEFERRAL_THRESHOLD,
        "total_deferrals": int(deferrals.sum()),
    }


# ---------------------------------------------------------------------------
# S4: BR dynamics monotone in W_t
# ---------------------------------------------------------------------------


def s4_br_monotone() -> tuple[bool, dict]:
    """Verify W_t is non-decreasing across BR rounds within each epoch.

    Builds an EquilibriumMatching with record_potential_trace=True
    and runs it on the small constellation, then checks that every
    epoch's potential trace is non-decreasing. Tolerance is a small
    floating-point epsilon since the Kirchhoff drops are sums of
    eigenvalue inverses.
    """
    walker = make_walker("small")
    T_orb_epochs = int(math.ceil(walker.orbital_period_s / DT_S))
    n_epochs = T_orb_epochs + 5
    _, feasibility = build_feasibility(walker, n_epochs)
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * DT_S, dt_s=DT_S)
    positions = propagate_keplerian(elements, times)

    n = walker.M
    pp = make_policy_params(T=T_orb_epochs)
    wl = WindowedLaplacian(n, window_length=T_orb_epochs)
    rng = make_rng(seed=42)

    policy = EquilibriumMatching(
        n=n,
        feasibility=feasibility,
        positions=positions,
        dt_s=DT_S,
        params=pp,
        rng=rng,
        windowed_laplacian=wl,
        record_potential_trace=True,
    )

    # Drive the policy. We don't care about the union here, just the per-epoch
    # potential traces.
    from orbitmatch.graph.laplacian import actions_to_edges  # noqa: PLC0415

    for t in range(n_epochs):
        actions = policy.decide(t)
        matching = actions_to_edges(actions)
        wl.push(matching)
        policy.step(t, actions)

    traces = policy.diagnostics().get("br_potential_trace")
    if traces is None:
        return False, {"reason": "br_potential_trace was not recorded"}

    # Check monotonicity for every trace.
    TOL = 1e-9
    violations: list[tuple[int, int, float, float]] = []
    n_traces = len(traces)
    non_trivial = 0
    for t, trace in enumerate(traces):
        if len(trace) < 2:
            continue
        non_trivial += 1
        for k in range(1, len(trace)):
            if trace[k] < trace[k - 1] - TOL:
                violations.append((t, k, float(trace[k - 1]), float(trace[k])))

    passed = len(violations) == 0
    return passed, {
        "n_epochs_checked": n_traces,
        "n_traces_with_iterations": non_trivial,
        "max_trace_length": max(len(tr) for tr in traces),
        "violations": violations[:5],  # cap printed list
        "tolerance": TOL,
    }


# ---------------------------------------------------------------------------
# S5: single-orbit visualization
# ---------------------------------------------------------------------------


def s5_orbit_visualization(label: str, out_dir: Path) -> tuple[bool, dict]:
    walker = make_walker(label)
    T_orb_epochs = int(math.ceil(walker.orbital_period_s / DT_S))
    n_epochs = T_orb_epochs + 30

    res = run_simulation(
        walker=walker,
        feasibility_params=make_feas_params(),
        policy_name="predictive",
        policy_params=make_policy_params(T=T_orb_epochs),
        n_epochs=n_epochs,
        dt_s=DT_S,
        seed=42,
        config_label=label,
    )

    # Rebuild positions for the orbit plot (the result doesn't store them).
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * DT_S, dt_s=DT_S)
    positions = propagate_keplerian(elements, times)

    # Snapshot at the middle of the first full window.
    epoch = T_orb_epochs // 2
    fig, ax = plot_orbit_projection(
        positions,
        epoch=epoch,
        matchings=list(res.matchings),
        walker=walker,
        show_trails=True,
        trail_epochs=T_orb_epochs // 6,
    )
    out = out_dir / f"orbit_{label}.pdf"
    fig.savefig(out)
    plt.close(fig)

    passed = out.exists() and out.stat().st_size > 0
    return passed, {
        "label": label,
        "epoch_shown": epoch,
        "n_realized_edges_at_epoch": int(res.matchings[epoch].shape[0]),
        "saved_to": str(out),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")
    apply_theme(context="diagnostic")

    diag_results = results_dir("diagnostics")
    diag_figs = figures_dir("diagnostics")
    print(f"results dir: {diag_results}")
    print(f"figures dir: {diag_figs}")
    print()

    checks = []
    t_start = time.perf_counter()

    # S1 on both configs.
    for cfg in ("small", "medium"):
        print(f"[S1/{cfg}] feasibility-union connected over T_orb")
        t0 = time.perf_counter()
        passed, info = s1_feasibility_union_connected(cfg)
        dur = time.perf_counter() - t0
        checks.append((f"S1/{cfg}", passed, info, dur))
        print(
            f"  {'[OK]' if passed else '[FAIL]'} ({dur:.1f}s): lambda_2_min = {info.get('lambda_2_min', float('nan')):.4f}"
        )

    # S2 on both configs.
    for cfg in ("small", "medium"):
        print(f"\n[S2/{cfg}] alpha_0 computed and saved")
        t0 = time.perf_counter()
        passed, info = s2_compute_alpha_0(cfg, diag_results)
        dur = time.perf_counter() - t0
        checks.append((f"S2/{cfg}", passed, info, dur))
        print(
            f"  {'[OK]' if passed else '[FAIL]'} ({dur:.1f}s): alpha_0 = {info.get('alpha_0', float('nan')):.4f}, saved to {Path(info.get('saved_to', '')).name}"
        )

    # S3 on small.
    print(f"\n[S3] predictive deferral mechanism fires on >= {S3_DEFERRAL_THRESHOLD:.0%} of epochs")
    t0 = time.perf_counter()
    passed, info = s3_deferral_fires()
    dur = time.perf_counter() - t0
    checks.append(("S3", passed, info, dur))
    print(
        f"  {'[OK]' if passed else '[FAIL]'} ({dur:.1f}s): fired_frac = {info.get('fired_frac', 0):.2%}, total = {info.get('total_deferrals', 0)}"
    )

    # S4 on small.
    print(f"\n[S4] equilibrium BR dynamics non-decreasing in W_t")
    t0 = time.perf_counter()
    passed, info = s4_br_monotone()
    dur = time.perf_counter() - t0
    checks.append(("S4", passed, info, dur))
    if passed:
        print(
            f"  [OK] ({dur:.1f}s): {info['n_traces_with_iterations']}/{info['n_epochs_checked']} epochs had >1 round; no violations"
        )
    else:
        print(f"  [FAIL] ({dur:.1f}s): {len(info.get('violations', []))} monotonicity violations")
        for tt, k, prev, cur in info.get("violations", [])[:5]:
            print(f"    epoch {tt}, round {k}: W went {prev:.6f} -> {cur:.6f}")

    # S5: orbit viz on both configs.
    for cfg in ("small", "medium"):
        print(f"\n[S5/{cfg}] single-orbit visualization")
        t0 = time.perf_counter()
        passed, info = s5_orbit_visualization(cfg, diag_figs)
        dur = time.perf_counter() - t0
        checks.append((f"S5/{cfg}", passed, info, dur))
        print(
            f"  {'[OK]' if passed else '[FAIL]'} ({dur:.1f}s): saved {Path(info.get('saved_to', '')).name} ({info.get('n_realized_edges_at_epoch', 0)} edges at epoch {info.get('epoch_shown', 0)})"
        )

    total = time.perf_counter() - t_start
    n_failed = sum(1 for _, p, _, _ in checks if not p)
    print()
    print("=" * 72)
    print(f"Sanity checks complete in {total:.1f}s: {len(checks) - n_failed}/{len(checks)} passed")
    print("=" * 72)
    if n_failed:
        for name, p, _, _ in checks:
            if not p:
                print(f"  FAIL: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
