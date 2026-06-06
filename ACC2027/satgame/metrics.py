"""Connectivity metrics over a series of realized graphs.

Reports three connectivity measures: algebraic connectivity (lambda_2),
effective graph resistance, and ``log tau`` -- the log number of spanning trees.
``log tau`` is the quantity the game policy actually optimizes: adding edge
(i, j) raises it by exactly ``log(1 + R_ij)`` (the single-edge marginal scored
in ``game.py``), so it is the apples-to-apples objective for evaluation.

These should be evaluated on the **windowed-union** graph G^cup(t; T) -- the
graph the game targets -- not on a single sparse epoch snapshot (which is
disconnected, giving lambda_2 = 0 and resistance = inf). ``series_metrics``
evaluates a list of graphs into parallel lists ready for plotting / CSV export.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def logtau(G):
    """Log number of spanning trees (Kirchhoff matrix-tree theorem).

    For a connected graph, ``log tau(G) = sum_{k>=2} log lambda_k(L) - log n``,
    where ``lambda_2..lambda_n`` are the nonzero Laplacian eigenvalues. This is
    the exact objective behind the game's per-edge score: adding edge (i, j)
    increases ``log tau`` by ``log(1 + R_ij)``.

    A disconnected graph has zero spanning trees (``log tau = -inf``). To keep
    the metric finite and informative on graphs that are not yet fully connected,
    we return the spanning-forest analog -- the log product of the nonzero
    Laplacian eigenvalues minus ``log n`` -- which coincides with ``log tau`` when
    the graph is connected and degrades gracefully otherwise.
    """
    n = G.number_of_nodes()
    if n < 2 or G.number_of_edges() == 0:
        return float("-inf")
    spectrum = nx.laplacian_spectrum(G)  # ascending eigenvalues, length n
    tol = 1e-8 * max(float(spectrum[-1]), 1.0)
    nonzero = spectrum[spectrum > tol]
    if nonzero.size == 0:
        return float("-inf")
    return float(np.sum(np.log(nonzero)) - np.log(n))


def graph_metrics(G):
    """Per-graph summary: lambda_2, effective resistance, log tau, edges, max deg."""
    max_degree = max((d for _, d in G.degree()), default=0)
    return {
        "lambda2": nx.algebraic_connectivity(G),
        "resistance": nx.effective_graph_resistance(G),
        "logtau": logtau(G),
        "edges": G.number_of_edges(),
        "max_degree": max_degree,
    }


def series_metrics(graphs):
    """Evaluate a list of graphs (typically the windowed-union series).

    Returns
    -------
    dict of lists, each of length ``len(graphs)``:
        ``lambda2``, ``resistance``, ``logtau``, ``edges``, ``max_degree``.
    """
    out = {"lambda2": [], "resistance": [], "logtau": [], "edges": [],
           "max_degree": []}
    for G in graphs:
        m = graph_metrics(G)
        for k in out:
            out[k].append(m[k])
    return out
