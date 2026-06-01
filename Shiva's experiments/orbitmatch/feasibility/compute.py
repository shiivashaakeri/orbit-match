# orbit-match/orbitmatch/feasibility/compute.py
# Run: imported by other modules; not a runnable script.

"""Combined feasibility tensor and disk caching.

The function :func:`compute_feasibility` AND's the three predicates from
:mod:`orbitmatch.feasibility.predicates` into a single boolean tensor
``F`` of shape ``(n_epochs, n, n)``. The function :func:`load_or_compute_feasibility`
adds content-hash-based caching: if the same ``(positions, dt_s, params)``
have been computed before, the tensor is loaded from
``data/processed/`` instead of recomputed.

Caching is appropriate here because:

- The feasibility tensor is deterministic in its inputs.
- It is expensive to compute (~150 MB intermediate tensor for the
  medium config; tens of seconds wall time).
- It is reused across every policy and every paper experiment.

Cache key
---------
Filename ``feas_{config_label}_dt{dt_s}_atm{atm}_rng{rng}_rate{rate}_{hash8}.npz``
where ``hash8`` is a short hash of the positions tensor itself, so a
change in the constellation invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from orbitmatch.feasibility.predicates import (
    FeasibilityParams,
    line_of_sight,
    pointing_rate_ok,
    range_ok,
)
from orbitmatch.utils.io import DATA_PROCESSED, load_trace, params_hash, save_trace
from orbitmatch.utils.logging_setup import get_logger
from orbitmatch.utils.timing import timed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_feasibility(
    positions: np.ndarray,
    dt_s: float,
    params: FeasibilityParams = FeasibilityParams(),
) -> np.ndarray:
    """Compute the combined feasibility tensor for a constellation trajectory.

    A pair ``(i, j)`` at epoch ``t`` is feasible iff all three predicates
    hold simultaneously:

    - Line of sight (chord clears Earth + atmosphere)
    - Range (distance below the laser-link limit)
    - Pointing rate (bearing angular rate below gimbal slew limit)

    Parameters
    ----------
    positions
        Shape ``(n_epochs, n, 3)``, ECI position tensor in km.
    dt_s
        Epoch length in seconds, used by the pointing-rate predicate.
    params
        Thresholds for the three predicates.

    Returns
    -------
    numpy.ndarray
        Boolean tensor of shape ``(n_epochs, n, n)``. Symmetric in the
        last two axes; diagonal is ``False``.
    """
    with timed("compute_feasibility"):
        los = line_of_sight(positions, params)
        rng = range_ok(positions, params)
        rate = pointing_rate_ok(positions, dt_s, params)
        feasibility = los & rng & rate

    n_offdiag = positions.shape[1] * (positions.shape[1] - 1)
    mean_frac = feasibility.sum(axis=(1, 2)).mean() / n_offdiag
    mean_neighbors = feasibility.sum(axis=(1, 2)).mean() / positions.shape[1]
    log.info(
        "Feasibility tensor: shape %s, mean fraction %.3f, mean neighbors/sat %.2f",
        feasibility.shape,
        mean_frac,
        mean_neighbors,
    )
    return feasibility


def load_or_compute_feasibility(
    positions: np.ndarray,
    dt_s: float,
    params: FeasibilityParams = FeasibilityParams(),
    *,
    config_label: str,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> tuple[np.ndarray, Path]:
    """Load the feasibility tensor from cache, or compute and cache it.

    The cache filename encodes the constellation label, the epoch length,
    and a hash of the positions tensor itself, so cache hits require the
    full input to match.

    Parameters
    ----------
    positions
        Shape ``(n_epochs, n, 3)``, ECI position tensor in km.
    dt_s
        Epoch length in seconds.
    params
        Feasibility parameters.
    config_label
        Short human-readable identifier for the constellation, used in
        the cache filename (e.g. ``"walker_medium_60_6_2_alt550_inc53"``).
    cache_dir
        Optional override for the cache directory. Defaults to
        ``data/processed/``.
    force_recompute
        If True, skip the cache lookup and recompute unconditionally.
        The result is still written back to the cache.

    Returns
    -------
    feasibility
        Boolean tensor of shape ``(n_epochs, n, n)``.
    path
        Path on disk where the cached tensor lives (whether loaded or
        just written).
    """
    cache_dir = cache_dir or DATA_PROCESSED
    cache_dir.mkdir(parents=True, exist_ok=True)

    positions_hash = _hash_array(positions, length=8)
    params_dict = asdict(params) | {"dt_s": float(dt_s)}
    param_h = params_hash(params_dict, length=8)
    filename = f"feas_{config_label}_dt{int(dt_s)}_p{param_h}_x{positions_hash}.npz"
    path = cache_dir / filename

    if path.exists() and not force_recompute:
        arrays, manifest = load_trace(path)
        log.info("Loaded feasibility tensor from cache: %s", path.name)
        return arrays["feasibility"], path

    log.info("Cache miss; computing feasibility tensor (will save to %s)", path.name)
    feasibility = compute_feasibility(positions, dt_s, params)

    metadata = {
        "kind": "feasibility_tensor",
        "config_label": config_label,
        "dt_s": float(dt_s),
        "params": asdict(params),
        "shape": list(feasibility.shape),
        "mean_fraction": float(feasibility.sum(axis=(1, 2)).mean() / (positions.shape[1] * (positions.shape[1] - 1))),
        "positions_hash": positions_hash,
    }
    save_trace(path, arrays={"feasibility": feasibility}, metadata=metadata)
    return feasibility, path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def feasibility_union(
    feasibility: np.ndarray,
    window_start: int,
    window_length: int,
) -> np.ndarray:
    """Build the feasibility-union graph over a sliding window.

    Computes ``F^u(t; T_0)`` from §IV.D of the paper: the graph whose
    edges are the pairs feasible at *any* epoch in ``[t, t + T_0)``.

    Parameters
    ----------
    feasibility
        Shape ``(n_epochs, n, n)``, the per-epoch feasibility tensor.
    window_start
        Starting epoch index ``t``.
    window_length
        Window length ``T_0`` in epochs.

    Returns
    -------
    numpy.ndarray
        Boolean ``(n, n)`` adjacency matrix of the union graph.
    """
    n_epochs = feasibility.shape[0]
    end = min(window_start + window_length, n_epochs)
    if window_start < 0 or window_start >= n_epochs:
        raise ValueError(f"window_start {window_start} out of range [0, {n_epochs}).")
    return feasibility[window_start:end].any(axis=0)


def feasible_neighbors(
    feasibility: np.ndarray,
    epoch: int,
    sat: int,
) -> np.ndarray:
    """Return the indices of satellites feasible with ``sat`` at ``epoch``.

    Convenience for policy code that needs to enumerate ``F_i(t)``.

    Parameters
    ----------
    feasibility
        Shape ``(n_epochs, n, n)``.
    epoch
        Epoch index ``t``.
    sat
        Satellite index ``i``.

    Returns
    -------
    numpy.ndarray
        1-D integer array of satellite indices ``j`` with
        ``feasibility[epoch, sat, j] == True``.
    """
    return np.flatnonzero(feasibility[epoch, sat])


def _hash_array(arr: np.ndarray, length: int = 8) -> str:
    """Short deterministic hash of an array's contents."""
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    return h[:length]
