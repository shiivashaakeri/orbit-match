# orbit-match/orbitmatch/graph/windowed.py
# Run: imported by other modules; not a runnable script.

"""Windowed Laplacian and certificate-related operations.

The windowed Laplacian is the central object of §II:

    Phi(t) = sum_{s = t - T + 1}^{t} L(s)

where L(s) is the Laplacian of the realized matching graph at epoch s
and T is the certificate window length. The connectivity certificate

    lambda_2(Phi(t)) >= alpha   for all t >= T                 (*)

is the property the policy is designed to maintain.

This module provides two primitives:

- :class:`WindowedLaplacian` — an incremental updater that holds Phi(t) as
  a dense (n, n) array and updates it in O(n) time when the matching
  rolls forward by one epoch (one Laplacian enters the window, one
  leaves). This is what the simulation loop calls at every epoch.

- :func:`compute_lambda2_trace` — a convenience that, given the full
  sequence of realized matchings, returns the lambda_2 trace over time.
  Used by the experiment runner after a simulation completes.

The incremental update is correct because the Laplacian is additive:

    Phi(t+1) - Phi(t) = L(t+1) - L(t - T + 1)

so we maintain the running sum by adding the new epoch's Laplacian and
subtracting the one that just fell out of the window.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np

from orbitmatch.graph.laplacian import (
    EMPTY_MATCHING,
    adjacency_to_laplacian,
    matching_to_laplacian,
)
from orbitmatch.graph.spectral import lambda_2
from orbitmatch.utils.logging_setup import get_logger
from orbitmatch.utils.timing import timed

log = get_logger(__name__)


class WindowedLaplacian:
    """Running sum of the last ``T`` Laplacians.

    Holds the windowed Laplacian Phi(t) = sum_{s = t - T + 1}^{t} L(s) as
    a dense (n, n) array and supports O(n) incremental rolling: each
    :meth:`push` adds a new matching and (once the window is full) evicts
    the oldest.

    Before T matchings have been pushed, the running sum is *the partial
    window* from epoch 0 to the current epoch — i.e. an incomplete
    initialization regime. Section IV of the paper analyzes the regime
    t >= T, so callers should ignore the lambda_2 trace before that
    point in experiments.

    Parameters
    ----------
    n
        Number of satellites (vertices).
    window_length
        T, the certificate window length, in epochs.

    Notes
    -----
    The matchings are stored as a deque of ``(k, 2)`` int arrays, not as
    pre-built Laplacians. This is a memory/recompute trade: storing each
    Laplacian dense is ``8 * n^2 * T`` bytes (which at n=60, T=360 is
    ~10 MB — tolerable but wasteful when matchings have <= n/2 edges and
    take 8*(n/2)*2 = 8*n bytes each).
    """

    def __init__(self, n: int, window_length: int) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}.")
        if window_length <= 0:
            raise ValueError(f"window_length must be positive; got {window_length}.")

        self.n = n
        self.T = window_length
        # Phi as a running dense matrix.
        self._phi = np.zeros((n, n), dtype=np.float64)
        # Recent matchings; left = oldest.
        self._matchings: deque[np.ndarray] = deque(maxlen=window_length)

    # ---- Mutating operations -------------------------------------------------

    def push(self, matching: np.ndarray) -> None:
        """Add a new epoch's matching to the window.

        Adds ``L(matching)`` to ``Phi`` and (if the window was full)
        subtracts the Laplacian of the matching that just fell out.

        Parameters
        ----------
        matching
            ``(k, 2)`` int array of edges.
        """
        # Evict the oldest if at capacity. deque.append handles maxlen but
        # we need to subtract its Laplacian *before* it disappears.
        if len(self._matchings) == self.T:
            old = self._matchings[0]
            self._phi -= matching_to_laplacian(old, self.n)

        # Add the new one.
        new_matching = np.asarray(matching, dtype=np.int64).reshape(-1, 2)
        self._phi += matching_to_laplacian(new_matching, self.n)
        self._matchings.append(new_matching)

    def reset(self) -> None:
        """Empty the window."""
        self._phi.fill(0.0)
        self._matchings.clear()

    # ---- Accessors -----------------------------------------------------------

    @property
    def phi(self) -> np.ndarray:
        """Current windowed Laplacian (n, n)."""
        return self._phi

    @property
    def window_filled(self) -> bool:
        """True once at least ``T`` matchings have been pushed."""
        return len(self._matchings) == self.T

    @property
    def epochs_in_window(self) -> int:
        """Number of matchings currently in the window (<= T)."""
        return len(self._matchings)

    def lambda_2(self) -> float:
        """Algebraic connectivity of the current windowed Laplacian."""
        return lambda_2(self._phi)


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------


def compute_lambda2_trace(
    matchings: Sequence[np.ndarray],
    n: int,
    window_length: int,
) -> np.ndarray:
    """Compute the lambda_2 trace over a sequence of matchings.

    Convenience for post-processing a saved simulation: given the matching
    at each epoch, returns ``lambda_2(Phi(t))`` for all ``t``.

    Before the window is full, the returned trace uses the partial sum;
    callers analyzing the certificate should slice to ``t >= T``.

    Parameters
    ----------
    matchings
        Iterable of ``(k_t, 2)`` int arrays, one per epoch.
    n
        Number of satellites.
    window_length
        ``T``.

    Returns
    -------
    numpy.ndarray
        Float64 array of length ``len(matchings)``.
    """
    wl = WindowedLaplacian(n, window_length)
    trace = np.zeros(len(matchings), dtype=np.float64)

    with timed("compute_lambda2_trace"):
        for t, m in enumerate(matchings):
            wl.push(m if m is not None else EMPTY_MATCHING)
            trace[t] = wl.lambda_2()

    log.info(
        "Computed lambda_2 trace over %d epochs (T=%d): min=%.4f, max=%.4f, mean=%.4f",
        len(matchings),
        window_length,
        float(trace[window_length - 1 :].min()) if len(trace) >= window_length else float("nan"),
        float(trace[window_length - 1 :].max()) if len(trace) >= window_length else float("nan"),
        float(trace[window_length - 1 :].mean()) if len(trace) >= window_length else float("nan"),
    )
    return trace


def matchings_to_laplacian_sequence(
    matchings: Iterable[np.ndarray],
    n: int,
) -> list[np.ndarray]:
    """Convert a sequence of matchings to a list of Laplacian matrices.

    Convenience used by diagnostic code that wants to inspect individual
    L(t) rather than the windowed sum. Not used in the hot path.
    """
    return [matching_to_laplacian(np.asarray(m, dtype=np.int64).reshape(-1, 2), n) for m in matchings]


# ---------------------------------------------------------------------------
# Windowed union (realized union graph G_union(t; T))
# ---------------------------------------------------------------------------


class WindowedUnion:
    """Running union of recent matchings.

    Maintains the realized union graph

        G_union(t; T) = (N, union of edges of G(s) for s in [t-T+1, t])

    as a dense boolean (n, n) adjacency matrix, plus a per-edge count
    deque so we know when an edge falls out of the window. The
    certificate (Theorem 6) is on lambda_2 of this graph's Laplacian.

    Parameters
    ----------
    n
        Number of satellites.
    window_length
        T, the certificate window length, in epochs.

    Notes
    -----
    Implementation uses an integer count matrix tracking how many
    epochs each edge has appeared in the window. When the count for an
    edge drops to zero, the edge is removed from the adjacency. This
    is O(k) per push where k is the number of edges in the pushed and
    evicted matchings (at most n/2 each).
    """

    def __init__(self, n: int, window_length: int) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}.")
        if window_length <= 0:
            raise ValueError(f"window_length must be positive; got {window_length}.")

        self.n = n
        self.T = window_length
        # Per-edge appearance count over the window.
        self._counts = np.zeros((n, n), dtype=np.int32)
        # Recent matchings; left = oldest.
        self._matchings: deque[np.ndarray] = deque(maxlen=window_length)

    # ---- Mutating operations -------------------------------------------------

    def push(self, matching: np.ndarray) -> None:
        """Add a matching to the window, evicting the oldest if at capacity."""
        # Evict the oldest if at capacity.
        if len(self._matchings) == self.T:
            old = self._matchings[0]
            if old.size > 0:
                self._counts[old[:, 0], old[:, 1]] -= 1
                self._counts[old[:, 1], old[:, 0]] -= 1

        # Add the new one.
        new_matching = np.asarray(matching, dtype=np.int64).reshape(-1, 2)
        if new_matching.size > 0:
            self._counts[new_matching[:, 0], new_matching[:, 1]] += 1
            self._counts[new_matching[:, 1], new_matching[:, 0]] += 1
        self._matchings.append(new_matching)

    def reset(self) -> None:
        """Empty the window."""
        self._counts.fill(0)
        self._matchings.clear()

    # ---- Accessors -----------------------------------------------------------

    @property
    def adjacency(self) -> np.ndarray:
        """Current union-graph adjacency as a boolean (n, n) matrix."""
        return self._counts > 0

    @property
    def laplacian(self) -> np.ndarray:
        """Laplacian L_union_G(t; T) of the current union graph."""
        return adjacency_to_laplacian(self.adjacency.astype(np.float64))

    @property
    def n_edges(self) -> int:
        """Number of distinct edges currently in the union graph."""
        return int(self.adjacency.sum() // 2)

    @property
    def window_filled(self) -> bool:
        """True once at least T matchings have been pushed."""
        return len(self._matchings) == self.T

    @property
    def epochs_in_window(self) -> int:
        """Number of matchings currently in the window (<= T)."""
        return len(self._matchings)

    def lambda_2(self) -> float:
        """Algebraic connectivity of the current union graph."""
        return lambda_2(self.laplacian)


# ---------------------------------------------------------------------------
# Batch utility for union-graph lambda_2
# ---------------------------------------------------------------------------


def compute_lambda2_union_trace(
    matchings: Sequence[np.ndarray],
    n: int,
    window_length: int,
) -> np.ndarray:
    """Compute the realized-union lambda_2 trace over a matching sequence.

    Mirror of :func:`compute_lambda2_trace` but on the union graph
    rather than the windowed sum. The union form is the object the
    certificate Theorem 6 guarantees.

    Parameters
    ----------
    matchings
        Iterable of ``(k_t, 2)`` int arrays, one per epoch.
    n
        Number of satellites.
    window_length
        ``T``.

    Returns
    -------
    numpy.ndarray
        Float64 array of length ``len(matchings)``.
    """
    wu = WindowedUnion(n, window_length)
    trace = np.zeros(len(matchings), dtype=np.float64)

    with timed("compute_lambda2_union_trace"):
        for t, m in enumerate(matchings):
            wu.push(m if m is not None else EMPTY_MATCHING)
            trace[t] = wu.lambda_2()

    if len(trace) >= window_length:
        log.info(
            "Computed union-graph lambda_2 trace over %d epochs (T=%d): min=%.4f, max=%.4f, mean=%.4f",
            len(matchings),
            window_length,
            float(trace[window_length - 1 :].min()),
            float(trace[window_length - 1 :].max()),
            float(trace[window_length - 1 :].mean()),
        )
    return trace
