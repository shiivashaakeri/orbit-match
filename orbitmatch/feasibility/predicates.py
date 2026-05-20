# orbit-match/orbitmatch/feasibility/predicates.py
# Run: imported by other modules; not a runnable script.

"""Feasibility predicates for inter-satellite links.

Three predicates determine whether a pair (i, j) is feasible at epoch t:

1. **Line of sight** — the chord from r_i to r_j must clear the Earth
   plus an atmospheric buffer.
2. **Range** — the chord length must be below the laser-link range limit.
3. **Pointing rate** — the angular rate of the line-of-sight bearing
   must be below the gimbal slew limit.

Each function below operates on full ``(n_epochs, n, 3)`` position
tensors and returns a boolean tensor of shape ``(n_epochs, n, n)``.
The diagonal is set to ``False`` (a satellite is never feasible with
itself), and the upper and lower triangles are symmetric (feasibility
is undirected).

These predicates are combined in :mod:`orbitmatch.feasibility.compute`,
which produces the final feasibility tensor used by policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitmatch.constellation.walker_delta import R_EARTH
from orbitmatch.utils.logging_setup import get_logger
from orbitmatch.utils.timing import timed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeasibilityParams:
    """Tunable parameters for the three feasibility predicates.

    Defaults are the values agreed in the project plan.

    Parameters
    ----------
    atm_buffer_km
        Atmospheric clearance above the Earth's surface. LOS chords must
        pass at least ``R_EARTH + atm_buffer_km`` from Earth's center.
    range_max_km
        Maximum allowed inter-satellite distance.
    rate_max_rad_per_s
        Maximum allowed angular rate of the line-of-sight bearing
        (the gimbal slew limit).
    """

    atm_buffer_km: float = 80.0
    range_max_km: float = 8000.0
    rate_max_rad_per_s: float = np.deg2rad(1.0)  # 1 deg / s


# ---------------------------------------------------------------------------
# Pairwise primitives
# ---------------------------------------------------------------------------


def _pairwise_relative_positions(positions: np.ndarray) -> np.ndarray:
    """Compute r_j - r_i for all pairs (i, j) and all epochs.

    Parameters
    ----------
    positions
        Shape ``(n_epochs, n, 3)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_epochs, n, n, 3)``. Entry ``[t, i, j]`` is
        ``positions[t, j] - positions[t, i]``.
    """
    # Broadcasting: positions[..., None, :, :] is (T, 1, n, 3),
    # positions[..., :, None, :] is (T, n, 1, 3); difference is (T, n, n, 3).
    return positions[:, None, :, :] - positions[:, :, None, :]


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Euclidean distance between every pair of satellites at every epoch.

    Parameters
    ----------
    positions
        Shape ``(n_epochs, n, 3)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_epochs, n, n)``. The diagonal is exactly zero.
    """
    delta = _pairwise_relative_positions(positions)  # (T, n, n, 3)
    return np.linalg.norm(delta, axis=-1)  # (T, n, n)


# ---------------------------------------------------------------------------
# Predicate 1: line of sight
# ---------------------------------------------------------------------------


def line_of_sight(
    positions: np.ndarray,
    params: FeasibilityParams = FeasibilityParams(),
) -> np.ndarray:
    """Boolean LOS tensor: does the chord from r_i to r_j clear the Earth?

    The minimum-distance point of the chord from Earth's center is computed
    in closed form. For a chord ``r(lambda) = r_i + lambda (r_j - r_i)``
    with ``lambda in [0, 1]``, the unconstrained minimum is at
    ``lambda* = -r_i . (r_j - r_i) / |r_j - r_i|^2``. If ``lambda*``
    lies in ``[0, 1]``, the minimum distance is at the perpendicular foot
    from the origin; otherwise the minimum is at one of the endpoints.
    For same-altitude orbits the closest approach is always at the chord
    midpoint, so the interior branch dominates. The endpoint branch is
    kept for generality and for any future extension to non-circular or
    multi-shell constellations.

    Parameters
    ----------
    positions
        Shape ``(n_epochs, n, 3)``.
    params
        Feasibility parameters; uses ``atm_buffer_km``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n_epochs, n, n)``, with ``True`` where
        the chord clears the Earth + atmosphere. Diagonal is ``False``.
    """
    T, n, _ = positions.shape
    r_min_threshold = R_EARTH + params.atm_buffer_km  # km

    with timed("line_of_sight", level="DEBUG"):
        # r_i at every (t, i, j) is positions[t, i]; r_j is positions[t, j].
        # Broadcast positions to (T, n, 1, 3) and (T, 1, n, 3).
        r_i = positions[:, :, None, :]  # (T, n, 1, 3)
        r_j = positions[:, None, :, :]  # (T, 1, n, 3)
        delta = r_j - r_i  # (T, n, n, 3)
        delta_sq = np.einsum("...k,...k->...", delta, delta)  # (T, n, n)

        # Avoid division by zero on the diagonal (where delta == 0).
        eye = np.eye(n, dtype=bool)
        delta_sq_safe = np.where(eye, 1.0, delta_sq)

        # lambda* in [0, 1] interior; otherwise minimum is at endpoint.
        r_i_dot_delta = np.einsum("...k,...k->...", r_i, delta)  # (T, n, n)
        lam_star = -r_i_dot_delta / delta_sq_safe

        # Distance from origin to chord, for interior case:
        # |r_i + lam_star * delta|^2 = |r_i|^2 - (r_i . delta)^2 / |delta|^2
        r_i_sq = np.einsum("...k,...k->...", r_i, r_i)  # (T, n, n)
        min_dist_sq_interior = r_i_sq - (r_i_dot_delta**2) / delta_sq_safe
        min_dist_sq_interior = np.maximum(min_dist_sq_interior, 0.0)  # guard tiny negatives

        # For two satellites at similar radii (always the case for our
        # circular constellation), the closest approach to Earth's center
        # lies near the chord midpoint (lam* ≈ 0.5), so `interior` is
        # almost always True. The endpoint branch is kept for generality:
        # if lam* falls outside [0, 1], the chord's closest approach is
        # at an endpoint, both of which are satellites above the
        # atmosphere by construction.
        interior = (lam_star > 0.0) & (lam_star < 1.0)

        # LOS fails only when the interior minimum is below threshold.
        min_dist_sq = np.where(interior, min_dist_sq_interior, np.inf)
        los = min_dist_sq >= r_min_threshold**2

        # Zero the diagonal: a satellite is never feasible with itself.
        los[:, eye] = False

    return los


# ---------------------------------------------------------------------------
# Predicate 2: range
# ---------------------------------------------------------------------------


def range_ok(
    positions: np.ndarray,
    params: FeasibilityParams = FeasibilityParams(),
) -> np.ndarray:
    """Boolean range tensor: is ``|r_j - r_i| <= range_max_km``?

    Returns
    -------
    numpy.ndarray
        Shape ``(n_epochs, n, n)``, boolean. Diagonal is ``False``.
    """
    with timed("range_ok", level="DEBUG"):
        dist = _pairwise_distances(positions)
        ok = dist <= params.range_max_km

        n = positions.shape[1]
        eye = np.eye(n, dtype=bool)
        ok[:, eye] = False

    return ok


# ---------------------------------------------------------------------------
# Predicate 3: pointing rate
# ---------------------------------------------------------------------------


def pointing_rate_ok(
    positions: np.ndarray,
    dt_s: float,
    params: FeasibilityParams = FeasibilityParams(),
) -> np.ndarray:
    """Boolean pointing-rate tensor: is the bearing slew below threshold?

    The unit bearing from i to j at epoch t is
    ``u_ij(t) = (r_j(t) - r_i(t)) / |r_j(t) - r_i(t)|``. Its rate
    of change is approximated by centered finite differences in time.

    For the endpoints of the time grid (no neighbor on one side),
    one-sided differences are used.

    Parameters
    ----------
    positions
        Shape ``(n_epochs, n, 3)``.
    dt_s
        Epoch length in seconds. Used to scale the finite difference.
    params
        Feasibility parameters; uses ``rate_max_rad_per_s``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_epochs, n, n)``, boolean. Diagonal is ``False``.
    """
    T, n, _ = positions.shape

    with timed("pointing_rate_ok", level="DEBUG"):
        # Bearings at every epoch.
        delta = _pairwise_relative_positions(positions)  # (T, n, n, 3)
        dist = np.linalg.norm(delta, axis=-1)  # (T, n, n)

        eye = np.eye(n, dtype=bool)
        dist_safe = np.where(eye, 1.0, dist)  # avoid /0 on diagonal
        u = delta / dist_safe[..., None]  # (T, n, n, 3)

        # Centered differences interior; one-sided at endpoints.
        # u_dot[t] approximates du/dt at t.
        u_dot = np.empty_like(u)
        u_dot[1:-1] = (u[2:] - u[:-2]) / (2.0 * dt_s)
        u_dot[0] = (u[1] - u[0]) / dt_s
        u_dot[-1] = (u[-1] - u[-2]) / dt_s

        rate = np.linalg.norm(u_dot, axis=-1)  # (T, n, n)
        ok = rate <= params.rate_max_rad_per_s
        ok[:, eye] = False

    return ok
