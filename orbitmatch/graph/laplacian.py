# orbit-match/orbitmatch/graph/laplacian.py
# Run: imported by other modules; not a runnable script.

"""Graph Laplacian construction from matchings.

A *matching* is a set of edges in which no two edges share a vertex. In our
problem, the matching is the set of realized inter-satellite links at one
epoch. This module converts a matching, represented as a list of vertex
pairs, into the graph Laplacian matrix.

For a graph G = (V, E) with V = {0, ..., n-1} and edge set E, the Laplacian
is the n x n matrix L = D - A, where D is the diagonal degree matrix and A
is the symmetric adjacency matrix. For an unweighted graph:

    L[i, i] =  deg(i)
    L[i, j] = -1   if (i, j) in E
    L[i, j] =  0   otherwise

The Laplacian is symmetric positive semidefinite, with smallest eigenvalue
zero (eigenvector = 1) and second-smallest eigenvalue lambda_2 equal to
the algebraic connectivity.

Matching representation
-----------------------
A matching is represented as a 2-D integer array of shape ``(k, 2)``, where
``k`` is the number of edges in the matching. Each row ``(i, j)`` is an
edge between vertices ``i`` and ``j``; we require ``i < j`` to avoid
double-counting.

This shape is convenient for vectorized construction and for storage
(matchings can be saved directly as small int arrays in .npz files).
"""

from __future__ import annotations

import numpy as np

from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# A matching is an (k, 2) int array of edges. We do not enforce this with
# a NewType; conventions live in docstrings.

EMPTY_MATCHING: np.ndarray = np.empty((0, 2), dtype=np.int64)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def matching_to_laplacian(matching: np.ndarray, n: int) -> np.ndarray:
    """Build the Laplacian of an unweighted matching graph.

    Parameters
    ----------
    matching
        Integer array of shape ``(k, 2)``. Each row ``(i, j)`` is an edge.
        We assume but do not enforce ``i < j`` and that the rows form a
        valid matching (no vertex appears twice). Empty matchings (``k == 0``)
        produce an all-zero Laplacian.
    n
        Total number of vertices in the graph (i.e. number of satellites).

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n, n)`` float64 Laplacian.

    Raises
    ------
    ValueError
        If ``matching`` has an unexpected shape or contains out-of-range
        vertex indices.
    """
    if matching.size == 0:
        return np.zeros((n, n), dtype=np.float64)

    if matching.ndim != 2 or matching.shape[1] != 2:
        raise ValueError(f"matching must have shape (k, 2); got {matching.shape}.")

    i_idx = matching[:, 0]
    j_idx = matching[:, 1]

    if (i_idx < 0).any() or (j_idx < 0).any() or (i_idx >= n).any() or (j_idx >= n).any():
        raise ValueError(
            f"matching contains vertex indices outside [0, {n}); "
            f"min={int(min(i_idx.min(), j_idx.min()))}, "
            f"max={int(max(i_idx.max(), j_idx.max()))}."
        )

    L = np.zeros((n, n), dtype=np.float64)
    # Off-diagonal: -1 for each edge in both directions.
    L[i_idx, j_idx] = -1.0
    L[j_idx, i_idx] = -1.0
    # Diagonal: degree. In a matching, every vertex appearing in `matching`
    # has degree exactly 1.
    L[i_idx, i_idx] += 1.0
    L[j_idx, j_idx] += 1.0

    return L


def adjacency_to_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Build the Laplacian from a dense adjacency matrix.

    More flexible than :func:`matching_to_laplacian`, since it allows the
    graph to be non-matching (e.g. for the feasibility-union graph
    :math:`\\mathcal{F}^\\cup`, whose Laplacian appears in Assumption 5).

    Parameters
    ----------
    adjacency
        Symmetric ``(n, n)`` array. May be boolean (treated as unweighted)
        or float (treated as weighted, with weights = entries).

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n, n)`` float64 Laplacian ``L = D - A``.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square; got {adjacency.shape}.")

    A = adjacency.astype(np.float64, copy=False)
    if not np.allclose(A, A.T):
        raise ValueError("adjacency must be symmetric.")

    # Zero the diagonal: self-loops do not contribute to a Laplacian.
    if not np.allclose(np.diag(A), 0.0):
        A = A.copy()
        np.fill_diagonal(A, 0.0)

    degrees = A.sum(axis=1)
    L = -A
    np.fill_diagonal(L, degrees)
    return L


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------


def edges_to_actions(matching: np.ndarray, n: int) -> np.ndarray:
    """Convert a matching to an action vector ``a[i]`` of length ``n``.

    For each vertex ``i``, ``a[i] = j`` if edge ``(i, j)`` is in the
    matching, else ``a[i] = -1`` (representing :math:`\\varnothing`).

    This is the dual representation used by policy code: each satellite
    holds a single integer indicating its requested partner.

    Parameters
    ----------
    matching
        Integer ``(k, 2)`` matching array.
    n
        Number of satellites.

    Returns
    -------
    numpy.ndarray
        Int64 array of length ``n``. Entry ``-1`` means "no link."
    """
    actions = -np.ones(n, dtype=np.int64)
    if matching.size == 0:
        return actions
    actions[matching[:, 0]] = matching[:, 1]
    actions[matching[:, 1]] = matching[:, 0]
    return actions


def actions_to_edges(actions: np.ndarray) -> np.ndarray:
    """Convert an action vector back to a matching edge list.

    Only mutual choices form edges: if ``actions[i] == j`` but
    ``actions[j] != i``, no edge is recorded. This is the mutual-choice
    rule from Section II of the paper.

    Parameters
    ----------
    actions
        Int array of length ``n``. ``actions[i] == -1`` means no request.

    Returns
    -------
    numpy.ndarray
        Int64 array of shape ``(k, 2)``, with rows sorted so each row
        ``(i, j)`` has ``i < j``. Duplicates eliminated.
    """
    actions = np.asarray(actions, dtype=np.int64)
    n = len(actions)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        j = actions[i]
        if j < 0 or j == i:
            continue
        if 0 <= j < n and actions[j] == i and i < j:
            edges.append((i, j))
    if not edges:
        return EMPTY_MATCHING
    return np.array(edges, dtype=np.int64)
