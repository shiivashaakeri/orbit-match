# orbit-match/orbitmatch/experiments/diagnostics.py
# Run: imported by scripts; not a runnable script.

"""Post-hoc diagnostics from a :class:`SimulationResult`.

Computes the headline numbers reported in the paper:

- ``alpha_0(T_0)``: the persistent-feasibility constant of Assumption 5.
  Computed directly from the feasibility tensor; independent of any policy.
- ``rho_realized(t)``: the empirical efficiency ratio
  lambda_2(L_union_G(t; T)) / alpha_0 at a given t.
- ``rho_cover``: realized union edges / feasibility union edges over the
  same window. The "fraction of available geometry captured."
- ``rho_match``: the matching-constraint efficiency. Reported two ways:
  - **Empirical**: rho_match = rho_realized / rho_cover. Tautological
    if rho_cover > 0; this is just the implicit factor that makes the
    decomposition rho = rho_match * rho_cover hold.
  - **Vizing bound**: an interval [T / (Delta+1), T / Delta] where
    Delta is the max degree of the feasibility-union graph. The exact
    rho_match is bounded above by T / chi', where chi' is the edge
    chromatic number, and Vizing's theorem says chi' in {Delta, Delta+1}.

The :class:`DiagnosticsReport` dataclass bundles every number so
scripts can pass it whole into :func:`orbitmatch.utils.io.save_trace`
as metadata.

This module is pure: it consumes a :class:`SimulationResult` and a
feasibility tensor and returns a report. No disk I/O, no simulations,
no policy code touched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from orbitmatch.experiments.runner import SimulationResult
from orbitmatch.feasibility.compute import feasibility_union
from orbitmatch.graph.laplacian import adjacency_to_laplacian
from orbitmatch.graph.spectral import lambda_2
from orbitmatch.utils.logging_setup import get_logger
from orbitmatch.utils.timing import timed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure functions: feasibility-only
# ---------------------------------------------------------------------------


def compute_alpha_0(
    feasibility: np.ndarray,
    T_0: int,
    *,
    stride: int = 1,
) -> tuple[float, int]:
    """Compute the persistent-feasibility constant alpha_0 from Assumption 5.

    Scans every sliding window of length T_0 starting at t = 0, 1, ...
    (or t = 0, stride, 2*stride, ... if a stride > 1 is given to make
    long-horizon scans cheap), computes the feasibility-union Laplacian
    for that window, and returns its lambda_2. The minimum over all
    such windows is alpha_0.

    Parameters
    ----------
    feasibility
        Boolean ``(n_epochs, n, n)`` tensor.
    T_0
        Window length in epochs.
    stride
        Step between window starts. ``stride=1`` (default) scans every
        position; larger strides are an approximation but much faster.

    Returns
    -------
    alpha_0
        Min over t of lambda_2(L^union_F(t; T_0)).
    argmin_t
        The starting epoch of the worst-case window.
    """
    n_epochs = feasibility.shape[0]
    if n_epochs < T_0:
        raise ValueError(f"T_0={T_0} exceeds simulation length {n_epochs}.")

    starts = range(0, n_epochs - T_0 + 1, stride)
    if len(starts) == 0:
        raise ValueError(f"No valid windows of length {T_0} in {n_epochs} epochs.")

    best_alpha = float("inf")
    best_t = 0
    with timed(f"compute_alpha_0(T_0={T_0}, stride={stride})", level="DEBUG"):
        for t in starts:
            union_adj = feasibility_union(feasibility, window_start=t, window_length=T_0)
            L = adjacency_to_laplacian(union_adj.astype(np.float64))
            lam = lambda_2(L)
            if lam < best_alpha:
                best_alpha = lam
                best_t = t
    log.info("alpha_0(T_0=%d) = %.4f (worst window starts at t=%d)", T_0, best_alpha, best_t)
    return float(best_alpha), int(best_t)


def feasibility_union_edge_count(
    feasibility: np.ndarray,
    t: int,
    T_0: int,
) -> int:
    """Number of edges in the feasibility-union graph over [t, t+T_0)."""
    adj = feasibility_union(feasibility, window_start=t, window_length=T_0)
    return int(adj.sum() // 2)


def feasibility_union_max_degree(
    feasibility: np.ndarray,
    t: int,
    T_0: int,
) -> int:
    """Max degree of the feasibility-union graph over [t, t+T_0)."""
    adj = feasibility_union(feasibility, window_start=t, window_length=T_0)
    return int(adj.sum(axis=1).max())


# ---------------------------------------------------------------------------
# Vizing-bound interval for rho_match
# ---------------------------------------------------------------------------


def rho_match_vizing_interval(T: int, max_degree: int) -> tuple[float, float]:
    """Upper-bound interval on rho_match from Vizing's theorem.

    The matching-constraint efficiency rho_match is bounded above by
    T / chi', where chi' is the edge chromatic number of the
    feasibility-union graph. Vizing's theorem says
    chi' in {Delta, Delta + 1}, so

        T / (Delta + 1) <= upper bound on rho_match <= T / Delta.

    Both endpoints are bounds: rho_match <= T / Delta (tightest case
    where chi' = Delta), and a theoretical-policy that achieves chi'
    can do no better than rho_match = T / chi'.

    Parameters
    ----------
    T
        Certificate window length.
    max_degree
        Delta, the max degree of the feasibility-union graph.

    Returns
    -------
    lower, upper
        The interval [T / (Delta+1), T / Delta]. Both clipped to 1.0
        since rho_match <= 1 by definition.
    """
    if max_degree <= 0:
        return 0.0, 0.0
    upper = min(1.0, T / max_degree)
    lower = min(1.0, T / (max_degree + 1))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Pure functions: result-driven
# ---------------------------------------------------------------------------


def rho_realized_trace(result: SimulationResult, alpha_0: float) -> np.ndarray:
    """Per-epoch rho_realized = lambda_2(L^union_G(t; T)) / alpha_0.

    Returns
    -------
    numpy.ndarray
        Length-``n_epochs`` float64 array. Entries are zero during the
        warmup window t < T (the union hasn't filled), then climb.
    """
    if alpha_0 <= 0:
        raise ValueError(f"alpha_0 must be positive; got {alpha_0}.")
    return result.lambda2_union / alpha_0


def rho_cover_at_t(
    result: SimulationResult,
    feasibility: np.ndarray,
    t: int,
    T_0: Optional[int] = None,
) -> float:
    """rho_cover = (realized union edges at t) / (feasibility union edges over T_0).

    The realized union is taken from the SimulationResult at epoch t
    (so the certificate window is result.T). The feasibility union
    uses the window ``[t - T_0 + 1, t + 1)``, mirroring the realized
    union's slicing convention.

    Parameters
    ----------
    result
        SimulationResult. result.n_union_edges_per_epoch[t] gives the
        realized count.
    feasibility
        Feasibility tensor.
    t
        Epoch at which to evaluate.
    T_0
        Feasibility window. Defaults to result.T (use the same window
        for both, which is the saturation case T = T_0 = T_orb).

    Returns
    -------
    float
        Realized fraction in [0, 1]. Zero if the feasibility window is
        empty.
    """
    if T_0 is None:
        T_0 = result.T

    realized = int(result.n_union_edges_per_epoch[t])
    # Mirror the realized union's [t - T + 1, t] window.
    window_start = max(0, t - T_0 + 1)
    window_length = min(T_0, t + 1)
    feas_adj = feasibility_union(feasibility, window_start, window_length)
    feas_edges = int(feas_adj.sum() // 2)

    if feas_edges == 0:
        return 0.0
    return realized / feas_edges


# ---------------------------------------------------------------------------
# DiagnosticsReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsReport:
    """All headline numbers from one (feasibility, result) pair.

    Two epochs of interest are evaluated: the final epoch of the
    simulation (steady-state numbers reported in tables), and the
    post-warmup epoch t = T (the first epoch at which the certificate
    is meaningful).

    Fields
    ------
    alpha_0
        Min over t of lambda_2(L^union_F(t; T_0)).
    alpha_0_worst_t
        Window start of the worst-case feasibility window.
    T_0
        Feasibility window used for alpha_0.
    rho_realized_final
        lambda_2(L^union_G(final; T)) / alpha_0.
    rho_realized_max
        Maximum of the post-warmup rho_realized trace.
    rho_realized_mean
        Mean of the post-warmup rho_realized trace.
    rho_cover_final
        (realized union edges at final t) / (feasibility union edges).
    rho_match_empirical_final
        rho_realized_final / rho_cover_final. NaN if rho_cover is 0.
    rho_match_vizing_lower
        T / (Delta + 1), where Delta is the max degree of the
        feasibility union at the final epoch's window.
    rho_match_vizing_upper
        T / Delta.
    feasibility_union_edges_final
        Edge count of the feasibility union at the final epoch.
    realized_union_edges_final
        Edge count of the realized union at the final epoch.
    max_degree_feasibility_final
        Delta at the final epoch.
    """

    # Geometric (feasibility-only) numbers.
    alpha_0: float
    alpha_0_worst_t: int
    T_0: int

    # Policy-side numbers at the final epoch.
    rho_realized_final: float
    rho_realized_max: float
    rho_realized_mean: float

    # Cover and match decompositions.
    rho_cover_final: float
    rho_match_empirical_final: float
    rho_match_vizing_lower: float
    rho_match_vizing_upper: float

    # Edge / degree counts for the final window.
    feasibility_union_edges_final: int
    realized_union_edges_final: int
    max_degree_feasibility_final: int

    # Provenance.
    policy_name: str
    config_label: str
    n: int
    T: int

    def to_metadata(self) -> dict:
        """Return a plain dict for save_trace metadata."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def build_report(
    result: SimulationResult,
    feasibility: np.ndarray,
    *,
    T_0: Optional[int] = None,
    alpha_0_stride: int = 1,
    precomputed_alpha_0: Optional[tuple[float, int]] = None,
) -> DiagnosticsReport:
    """Build a DiagnosticsReport from a result and its feasibility tensor.

    Parameters
    ----------
    result
        Output of :func:`run_simulation`.
    feasibility
        The feasibility tensor used by the simulation. Shape
        ``(n_epochs, n, n)``, boolean.
    T_0
        Feasibility window for alpha_0. Defaults to ``result.T`` so
        that T = T_0 (the saturation case from EXPERIMENTS_LOG).
    alpha_0_stride
        Stride for the alpha_0 scan. ``1`` is exact; larger is faster.
    precomputed_alpha_0
        If alpha_0 has already been computed (it depends only on the
        feasibility tensor, not on the policy), pass it here as
        ``(alpha_0, argmin_t)`` to skip the recomputation. Useful when
        diagnosing multiple policies on the same constellation.

    Returns
    -------
    DiagnosticsReport
        Frozen dataclass. See class doc.
    """
    n_epochs = result.n_epochs
    T = result.T
    T_0 = T_0 or T

    # Geometric numbers.
    if precomputed_alpha_0 is None:
        alpha_0, alpha_0_t = compute_alpha_0(feasibility, T_0, stride=alpha_0_stride)
    else:
        alpha_0, alpha_0_t = precomputed_alpha_0

    # Policy-side trace.
    rho_trace = rho_realized_trace(result, alpha_0)

    # Post-warmup slice.
    post = rho_trace[T:]
    if len(post) == 0:
        # Simulation didn't run past warmup; report final-epoch values for both.
        rho_max = float(rho_trace[-1])
        rho_mean = float(rho_trace[-1])
    else:
        rho_max = float(post.max())
        rho_mean = float(post.mean())

    final_t = n_epochs - 1
    rho_final = float(rho_trace[final_t])

    # Cover decomposition at the final epoch.
    rho_cover_final = rho_cover_at_t(result, feasibility, final_t, T_0=T_0)
    rho_match_emp = rho_final / rho_cover_final if rho_cover_final > 0 else float("nan")

    # Vizing-bound interval at the final epoch's feasibility window.
    window_start = max(0, final_t - T_0 + 1)
    window_length = min(T_0, final_t + 1)
    feas_adj_final = feasibility_union(feasibility, window_start, window_length)
    feas_edges_final = int(feas_adj_final.sum() // 2)
    max_deg_final = int(feas_adj_final.sum(axis=1).max())
    vizing_lo, vizing_hi = rho_match_vizing_interval(T, max_deg_final)

    realized_edges_final = int(result.n_union_edges_per_epoch[final_t])

    report = DiagnosticsReport(
        alpha_0=alpha_0,
        alpha_0_worst_t=alpha_0_t,
        T_0=T_0,
        rho_realized_final=rho_final,
        rho_realized_max=rho_max,
        rho_realized_mean=rho_mean,
        rho_cover_final=rho_cover_final,
        rho_match_empirical_final=rho_match_emp,
        rho_match_vizing_lower=vizing_lo,
        rho_match_vizing_upper=vizing_hi,
        feasibility_union_edges_final=feas_edges_final,
        realized_union_edges_final=realized_edges_final,
        max_degree_feasibility_final=max_deg_final,
        policy_name=result.policy_name,
        config_label=result.config_label,
        n=result.n,
        T=T,
    )

    log.info(
        "[%s/%s] alpha_0=%.4f, rho_realized(final)=%.4f, rho_cover(final)=%.4f, "
        "rho_match_empirical=%.4f, Vizing bound on rho_match in [%.4f, %.4f]",
        result.config_label,
        result.policy_name,
        report.alpha_0,
        report.rho_realized_final,
        report.rho_cover_final,
        report.rho_match_empirical_final,
        report.rho_match_vizing_lower,
        report.rho_match_vizing_upper,
    )
    return report
