# orbit-match/scripts/check_diagnostic_plots.py
# Run: python -m scripts.check_diagnostic_plots

"""Smoke test for orbitmatch.plotting.diagnostic_plots.

Renders every diagnostic plot function with synthetic data:

1. plot_orbit_projection -- snapshot at one epoch, with optional trails and matching overlay
2. plot_feasibility_heatmap -- (pair, time) heatmap of the feasibility tensor
3. plot_lambda2_phi_trace -- lambda_2(Phi) over time across policies
4. plot_deferral_histogram -- distribution of per-epoch deferrals
5. plot_edge_count_trace -- per-epoch matching + union edge counts
6. plot_br_rounds_trace -- best-response rounds per epoch for equilibrium

All checks verify that the figure renders, has axis labels, and saves
to a non-empty PDF.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.runner import SimulationResult
from orbitmatch.plotting.diagnostic_plots import (
    plot_br_rounds_trace,
    plot_deferral_histogram,
    plot_edge_count_trace,
    plot_feasibility_heatmap,
    plot_lambda2_phi_trace,
    plot_orbit_projection,
)
from orbitmatch.plotting.theme import apply_theme
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def make_synthetic_positions(n_epochs: int = 100, walker: WalkerDeltaConfig = None) -> np.ndarray:
    """Build a synthetic (n_epochs, n, 3) positions tensor.

    Uses real-ish orbital propagation if walker is given, else random
    points on a sphere of radius ~7000 km.
    """
    if walker is not None:
        from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian  # noqa: PLC0415

        elements = walker.initial_elements()
        times = make_time_grid(duration_s=n_epochs * 10.0, dt_s=10.0)
        return propagate_keplerian(elements, times)
    n = 24
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(n_epochs, n, 3))
    pos *= 7000.0 / np.linalg.norm(pos, axis=-1, keepdims=True)
    return pos


def make_synthetic_matchings(n_epochs: int, n: int, seed: int = 0) -> list[np.ndarray]:
    """A list of (k_t, 2) int matchings for each epoch."""
    rng = np.random.default_rng(seed)
    matchings = []
    for _ in range(n_epochs):
        k = rng.integers(0, n // 2 + 1)
        if k == 0:
            matchings.append(np.empty((0, 2), dtype=np.int64))
            continue
        # Random matching: shuffle, pair adjacent.
        perm = rng.permutation(n)
        pairs = []
        for i in range(0, 2 * k, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            pairs.append((min(a, b), max(a, b)))
        matchings.append(np.array(pairs, dtype=np.int64))
    return matchings


def make_synthetic_result(
    policy_name: str,
    seed: int,
    n_epochs: int = 200,
    T: int = 60,
    n: int = 24,
    with_deferrals: bool = False,
    with_br: bool = False,
) -> SimulationResult:
    """Synthetic SimulationResult with the relevant diagnostics."""
    rng = np.random.default_rng(seed + hash(policy_name) % 100)
    lambda2_union = np.clip(
        (np.arange(n_epochs) - T) / T * 0.7 + rng.normal(0, 0.02, n_epochs),
        0,
        None,
    )
    lambda2_phi = 4 * lambda2_union + rng.normal(0, 0.1, n_epochs)
    lambda2_phi = np.clip(lambda2_phi, 0, None)

    matchings = make_synthetic_matchings(n_epochs, n, seed=seed)
    n_edges = np.array([m.shape[0] for m in matchings], dtype=np.int64)
    n_union = np.minimum(np.cumsum(n_edges), n * (n - 1) // 2).astype(np.int64)

    diag: dict[str, np.ndarray] = {}
    if with_deferrals:
        diag["deferrals"] = rng.integers(0, 8, n_epochs).astype(np.int64)
    if with_br:
        diag["br_rounds"] = rng.integers(1, 5, n_epochs).astype(np.int64)
        diag["br_converged"] = np.ones(n_epochs, dtype=np.int64)
        diag["br_converged"][rng.integers(0, n_epochs, 3)] = 0  # a few non-converged

    return SimulationResult(
        policy_name=policy_name,
        config_label="synthetic",
        n=n,
        n_epochs=n_epochs,
        T=T,
        dt_s=10.0,
        lambda2_phi=lambda2_phi,
        lambda2_union=lambda2_union,
        n_edges_per_epoch=n_edges,
        n_union_edges_per_epoch=n_union,
        actions=np.full((n_epochs, n), -1, dtype=np.int64),
        matchings=tuple(matchings),
        final_phi=np.zeros((n, n)),
        final_union_adjacency=np.zeros((n, n), dtype=bool),
        policy_diagnostics=diag,
        metadata={},
    )


def make_synthetic_feasibility(n_epochs: int = 60, n: int = 24, seed: int = 0) -> np.ndarray:
    """Synthetic feasibility tensor with ~20% density."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n_epochs, n, n)) < 0.2
    feas = raw & np.swapaxes(raw, 1, 2)
    eye = np.eye(n, dtype=bool)
    feas[:, eye] = False
    return feas


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def check_axis_labels(ax: plt.Axes, label: str) -> None:
    if not ax.get_xlabel():
        raise AssertionError(f"{label}: x-axis has no label")
    if not ax.get_ylabel():
        raise AssertionError(f"{label}: y-axis has no label")


def check_pdf_saved(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"{label}: PDF not saved correctly at {path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_orbit_projection(tmp_dir: Path) -> None:
    walker = WalkerDeltaConfig(M=24, P=4, F=1, altitude_km=550.0, inclination_deg=53.0, name="small")
    positions = make_synthetic_positions(n_epochs=100, walker=walker)
    matchings = make_synthetic_matchings(100, walker.M, seed=1)
    print("  rendering plot_orbit_projection (with matching, with trails) ...")
    fig, ax = plot_orbit_projection(
        positions,
        epoch=50,
        matchings=matchings,
        walker=walker,
        show_trails=True,
        trail_epochs=30,
    )
    check_axis_labels(ax, "orbit projection")
    out = tmp_dir / "orbit_projection.pdf"
    fig.savefig(out)
    plt.close(fig)
    check_pdf_saved(out, "orbit projection")
    print(f"  [OK] orbit_projection saved ({out.stat().st_size // 1024} KB)")

    print("  rendering plot_orbit_projection (no matching, no trails) ...")
    fig, ax = plot_orbit_projection(positions, epoch=0, matchings=None, walker=None, show_trails=False)
    out = tmp_dir / "orbit_projection_bare.pdf"
    fig.savefig(out)
    plt.close(fig)
    check_pdf_saved(out, "orbit projection bare")
    print(f"  [OK] orbit_projection_bare saved")


def test_feasibility_heatmap(tmp_dir: Path) -> None:
    feasibility = make_synthetic_feasibility(n_epochs=60, n=24)
    print("  rendering plot_feasibility_heatmap ...")
    fig, ax = plot_feasibility_heatmap(feasibility)
    check_axis_labels(ax, "feasibility heatmap")
    out = tmp_dir / "feasibility_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    check_pdf_saved(out, "feasibility heatmap")
    print(f"  [OK] feasibility_heatmap saved ({out.stat().st_size // 1024} KB)")


def test_lambda2_phi_trace(tmp_dir: Path) -> None:
    results_by_policy = {
        name: [make_synthetic_result(name, seed) for seed in (42, 43, 44)]
        for name in ("predictive", "greedy", "random")
    }
    print("  rendering plot_lambda2_phi_trace ...")
    fig, ax = plot_lambda2_phi_trace(results_by_policy)
    check_axis_labels(ax, "lambda2(Phi) trace")
    if ax.get_legend() is None:
        raise AssertionError("lambda2(Phi) trace: no legend")
    out = tmp_dir / "lambda2_phi_trace.pdf"
    fig.savefig(out)
    plt.close(fig)
    check_pdf_saved(out, "lambda2(Phi) trace")
    print(f"  [OK] lambda2_phi_trace saved ({out.stat().st_size // 1024} KB)")


def test_deferral_histogram(tmp_dir: Path) -> None:
    result = make_synthetic_result("predictive", seed=42, with_deferrals=True)
    print("  rendering plot_deferral_histogram ...")
    fig, ax = plot_deferral_histogram(result)
    check_axis_labels(ax, "deferral histogram")
    out = tmp_dir / "deferral_histogram.pdf"
    fig.savefig(out)
    plt.close(fig)
    check_pdf_saved(out, "deferral histogram")
    print(f"  [OK] deferral_histogram saved")

    # Confirm KeyError when the diagnostic is missing.
    result_no_def = make_synthetic_result("greedy", seed=42, with_deferrals=False)
    try:
        plot_deferral_histogram(result_no_def)
    except KeyError:
        print(f"  [OK] KeyError raised when 'deferrals' diagnostic missing")
    else:
        raise AssertionError("Expected KeyError on a result with no 'deferrals' diagnostic")


def test_edge_count_trace(tmp_dir: Path) -> None:
    results_by_policy = {
        name: [make_synthetic_result(name, seed) for seed in (42, 43)] for name in ("predictive", "greedy")
    }
    print("  rendering plot_edge_count_trace ...")
    fig, ax = plot_edge_count_trace(results_by_policy, show_union=True)
    check_axis_labels(ax, "edge count")
    out = tmp_dir / "edge_count_trace.pdf"
    fig.savefig(out)
    plt.close(fig)
    check_pdf_saved(out, "edge count trace")
    print(f"  [OK] edge_count_trace saved")


def test_br_rounds_trace(tmp_dir: Path) -> None:
    result = make_synthetic_result("equilibrium", seed=42, with_br=True)
    print("  rendering plot_br_rounds_trace ...")
    fig, ax = plot_br_rounds_trace(result)
    check_axis_labels(ax, "br rounds")
    out = tmp_dir / "br_rounds_trace.pdf"
    fig.savefig(out)
    plt.close(fig)
    check_pdf_saved(out, "br rounds trace")
    print(f"  [OK] br_rounds_trace saved")

    # Confirm KeyError on policy without br diagnostic.
    result_no_br = make_synthetic_result("predictive", seed=42, with_br=False)
    try:
        plot_br_rounds_trace(result_no_br)
    except KeyError:
        print(f"  [OK] KeyError raised when 'br_rounds' diagnostic missing")
    else:
        raise AssertionError("Expected KeyError on a result with no 'br_rounds' diagnostic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")
    apply_theme(context="diagnostic")
    print("[setup] apply_theme(context='diagnostic')")

    tmp_dir = Path(tempfile.mkdtemp(prefix="orbitmatch_diag_plots_"))
    print(f"  output dir: {tmp_dir}\n")

    failed = 0
    tests = [
        ("orbit projection", test_orbit_projection),
        ("feasibility heatmap", test_feasibility_heatmap),
        ("lambda_2(Phi) trace", test_lambda2_phi_trace),
        ("deferral histogram", test_deferral_histogram),
        ("edge count trace", test_edge_count_trace),
        ("BR rounds trace", test_br_rounds_trace),
    ]

    try:
        for label, fn in tests:
            print(f"[{label}]")
            try:
                fn(tmp_dir)
            except AssertionError as e:
                print(f"  [FAIL] {e}")
                failed += 1
            except Exception as e:
                import traceback

                traceback.print_exc()
                failed += 1
            print()
    finally:
        shutil.rmtree(tmp_dir)

    if failed:
        print(f"FAILED: {failed} sub-tests")
        return 1
    print("All diagnostic_plots smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
