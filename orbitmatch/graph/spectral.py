# orbit-match/orbitmatch/graph/spectral.py
# Run: imported by other modules; not a runnable script.

"""Spectral computations on graph Laplacians.

The two operations we need from a Laplacian are the algebraic connectivity
``lambda_2`` and (occasionally) the full eigendecomposition for diagnostics.
Both are expensive enough to be worth caching, since policy code evaluates
the same matching many times during best-response dynamics.

Implementation
--------------
- ``lambda_2(L)``: for small graphs (n <= 64) we use dense ``np.linalg.eigvalsh``;
  for larger we use sparse ``scipy.sparse.linalg.eigsh`` requesting only the
  two smallest eigenvalues. For our constellations (n <= 96) the dense path
  is almost always faster, so it's the default.

- ``lambda_2_of_matching(matching, n)``: a thin wrapper that hashes the
  matching as a cache key. This is the form policy code calls.

Caching
-------
We use ``functools.lru_cache`` keyed on a frozen bytes representation of
the matching. Maxsize defaults to 65536; raise it via ``set_cache_size``
if profiling shows misses.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from orbitmatch.graph.laplacian import matching_to_laplacian
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Direct (uncached) lambda_2
# ---------------------------------------------------------------------------


# Threshold below which we always use the dense eigvalsh path. For n in our
# range (<= 96) dense is faster than sparse, but we keep the toggle so that
# future scaling experiments at n=200+ can flip to sparse.
DENSE_THRESHOLD = 128


def lambda_2(L: np.ndarray, *, prefer: str = "auto") -> float:
    """Algebraic connectivity of a graph Laplacian.

    Computes the second-smallest eigenvalue of ``L``. Disconnected graphs
    return a value at or near zero (subject to floating-point noise).

    Parameters
    ----------
    L
        Symmetric positive-semidefinite Laplacian of shape ``(n, n)``.
    prefer
        ``"dense"``, ``"sparse"``, or ``"auto"`` (default). Auto routes
        to dense for ``n < DENSE_THRESHOLD`` and sparse otherwise.

    Returns
    -------
    float
        :math:`\\lambda_2(L)`, clipped to be non-negative (tiny negative
        eigenvalues from floating-point noise are reported as 0).
    """
    n = L.shape[0]
    if prefer == "auto":
        prefer = "dense" if n < DENSE_THRESHOLD else "sparse"

    if prefer == "dense":
        eigs = np.linalg.eigvalsh(L)
        return max(float(eigs[1]) if n > 1 else 0.0, 0.0)

    if prefer == "sparse":
        # Sparse path: pull the two smallest eigenvalues.
        from scipy.sparse import csr_matrix  # noqa: PLC0415
        from scipy.sparse.linalg import eigsh  # noqa: PLC0415

        L_sparse = csr_matrix(L)
        # `sigma=0` with `which='LM'` and `mode='normal'` would target
        # eigenvalues near 0 via shift-invert; simpler is `which='SM'`,
        # which is fine for our problem size.
        eigs = eigsh(L_sparse, k=2, which="SM", return_eigenvectors=False)
        eigs = np.sort(eigs)
        return max(float(eigs[1]), 0.0)

    raise ValueError(f"prefer must be 'dense', 'sparse', or 'auto'; got {prefer!r}.")


def eigenvalues(L: np.ndarray) -> np.ndarray:
    """Full ascending-sorted spectrum of a Laplacian.

    Used in diagnostics only; not in policy hot paths.

    Parameters
    ----------
    L
        Symmetric Laplacian.

    Returns
    -------
    numpy.ndarray
        1-D array of eigenvalues in ascending order. Negative entries
        from floating-point noise are clipped to 0.
    """
    eigs = np.linalg.eigvalsh(L)
    return np.maximum(eigs, 0.0)


# ---------------------------------------------------------------------------
# Cached lambda_2 of a matching
# ---------------------------------------------------------------------------


_CACHE_MAXSIZE = 65536


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _cached_lambda_2(matching_bytes: bytes, n: int) -> float:
    """Internal: dispatch to ``lambda_2`` given a matching as raw bytes.

    The bytes-based signature is necessary because ``lru_cache`` needs
    hashable arguments and ``np.ndarray`` is not hashable.
    """
    if len(matching_bytes) == 0:
        # Empty matching has L = 0, so lambda_2 = 0.
        return 0.0
    matching = np.frombuffer(matching_bytes, dtype=np.int64).reshape(-1, 2)
    L = matching_to_laplacian(matching, n)
    return lambda_2(L)


def lambda_2_of_matching(matching: np.ndarray, n: int) -> float:
    """Cached algebraic connectivity of a matching graph.

    The matching is normalized to canonical form (rows sorted with
    ``i < j``, then sorted lexicographically by row) so that two equivalent
    matchings always hit the same cache entry.

    Parameters
    ----------
    matching
        Integer ``(k, 2)`` matching array.
    n
        Number of vertices.

    Returns
    -------
    float
        :math:`\\lambda_2` of the matching's Laplacian.
    """
    canonical = canonicalize_matching(matching)
    return _cached_lambda_2(canonical.tobytes(), n)


def canonicalize_matching(matching: np.ndarray) -> np.ndarray:
    """Sort a matching into canonical form.

    Each row is reordered so the smaller index comes first, then rows are
    sorted lexicographically. The result is unique up to set equality.

    Empty matchings are returned unchanged (as a ``(0, 2)`` int64 array).
    """
    if matching.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    m = np.asarray(matching, dtype=np.int64)
    # Sort each row in ascending order.
    m = np.sort(m, axis=1)
    # Sort rows lexicographically.
    order = np.lexsort((m[:, 1], m[:, 0]))
    return m[order]


# ---------------------------------------------------------------------------
# Cache controls
# ---------------------------------------------------------------------------


def cache_info():
    """Return the underlying ``lru_cache`` info object for inspection."""
    return _cached_lambda_2.cache_info()


def cache_clear() -> None:
    """Empty the matching ``lambda_2`` cache. Use sparingly; mostly for tests."""
    _cached_lambda_2.cache_clear()
    log.debug("Cleared lambda_2 matching cache.")


def set_cache_size(maxsize: int) -> None:
    """Resize the cache. Resets contents.

    Use only at start-of-script; calling mid-run discards work in progress.
    """
    global _cached_lambda_2, _CACHE_MAXSIZE  # noqa: PLW0603
    _CACHE_MAXSIZE = maxsize
    _cached_lambda_2 = lru_cache(maxsize=maxsize)(_cached_lambda_2.__wrapped__)
    log.debug("Resized lambda_2 matching cache to %d.", maxsize)
