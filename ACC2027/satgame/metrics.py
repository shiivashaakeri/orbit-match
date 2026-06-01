"""Connectivity metrics over a series of realized graphs.

Wraps the two metrics the notebook reports (cell 5): algebraic connectivity
(lambda_2) and effective graph resistance. ``series_metrics`` evaluates a list
of per-epoch graphs into parallel lists ready for plotting / CSV export.
"""

from __future__ import annotations

import networkx as nx


def graph_metrics(G):
    """Per-graph summary: lambda_2, effective resistance, edges, max degree."""
    max_degree = max((d for _, d in G.degree()), default=0)
    return {
        "lambda2": nx.algebraic_connectivity(G),
        "resistance": nx.effective_graph_resistance(G),
        "edges": G.number_of_edges(),
        "max_degree": max_degree,
    }


def series_metrics(graphs):
    """Evaluate a list of per-epoch graphs.

    Returns
    -------
    dict of lists, each of length ``len(graphs)``:
        ``lambda2``, ``resistance``, ``edges``, ``max_degree``.
    """
    out = {"lambda2": [], "resistance": [], "edges": [], "max_degree": []}
    for G in graphs:
        m = graph_metrics(G)
        for k in out:
            out[k].append(m[k])
    return out
