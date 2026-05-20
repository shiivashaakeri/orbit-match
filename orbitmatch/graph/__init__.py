# orbit-match/orbitmatch/graph/__init__.py
"""Graph Laplacian, windowed Laplacian, and spectral utilities."""

from orbitmatch.graph.laplacian import (
    EMPTY_MATCHING,
    actions_to_edges,
    adjacency_to_laplacian,
    edges_to_actions,
    matching_to_laplacian,
)
from orbitmatch.graph.spectral import (
    cache_clear,
    cache_info,
    canonicalize_matching,
    eigenvalues,
    lambda_2,
    lambda_2_of_matching,
    set_cache_size,
)
from orbitmatch.graph.windowed import (
    WindowedLaplacian,
    compute_lambda2_trace,
    matchings_to_laplacian_sequence,
)

__all__ = [
    "EMPTY_MATCHING",
    "WindowedLaplacian",
    "actions_to_edges",
    "adjacency_to_laplacian",
    "cache_clear",
    "cache_info",
    "canonicalize_matching",
    "compute_lambda2_trace",
    "edges_to_actions",
    "eigenvalues",
    "lambda_2",
    "lambda_2_of_matching",
    "matching_to_laplacian",
    "matchings_to_laplacian_sequence",
    "set_cache_size",
]
