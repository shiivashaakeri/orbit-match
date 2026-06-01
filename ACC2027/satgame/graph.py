"""Graph construction helpers.

``generate_network`` builds the empty node-only graph the policies populate
(from the notebook's cell 1). ``windowed_union`` builds the windowed-union
graph G^cup(t; T) the game best-responds against -- the connectivity context
introduced when fixing the snapshot-vs-union bug.
"""

from __future__ import annotations

import networkx as nx


def generate_network(positions):
    """Empty graph with one node per satellite (indices 0..N-1), no edges."""
    num_satellites = len(positions)
    G = nx.Graph()
    for i in range(num_satellites):
        G.add_node(i)
    return G


def windowed_union(num_satellites, history, window):
    """Union of the last ``window`` realized graphs in ``history``.

    Parameters
    ----------
    num_satellites : int
        Node count; all satellites are present even if isolated.
    history : list[nx.Graph]
        Past realized graphs, most recent last.
    window : int
        Horizon T: only the last ``window`` graphs contribute edges.

    Returns
    -------
    nx.Graph
        G^cup(t; T) -- nodes 0..N-1 with the union of windowed edges.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_satellites))
    for past in history[-window:]:
        G.add_edges_from(past.edges())
    return G
