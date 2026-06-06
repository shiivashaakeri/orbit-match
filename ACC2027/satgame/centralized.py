"""Centralized greedy ISL topology formation -- the optimization ceiling.

This is the *centralized* counterpart to the decentralized matching game in
``game.py``. A single planner with full global state chooses one epoch's edge
set to maximize the SAME objective the game targets,

    Phi = log tau(G^cup(t; T)) - alpha * sum_i c_i(s_i),

subject to the SAME constraints: feasibility (mutual visibility), the k=4
directional terminal budget, and mutual choice (automatic when one planner
assigns both endpoints).

It differs from the game in exactly two ways, which is the whole point:

* **Exact global marginals** -- the single-edge gain of ``log tau`` is the exact
  effective resistance ``log(1 + R_ij)`` on the FULL current Laplacian (via its
  pseudoinverse ``L_dagger`` with rank-one Sherman-Morrison updates), not an
  h-hop local estimate.
* **Global coordination, re-evaluated greedily** -- one global priority order
  with marginals recomputed after every accepted edge (true submodular
  diminishing returns), rather than a linearized fixed-base score realized by a
  bilateral matching.

So ``game`` vs ``centralized`` measures the price of decentralization plus local
information -- the paper's quantity of interest.

Implementation: lazy greedy (CELF) over feasible candidate edges. Because
``log tau`` is monotone and submodular and the slewing term is a per-edge
constant, accepted marginals are non-increasing, so CELF's "re-score only the
heap top" rule is valid and keeps the full-shell run tractable. A short bridging
phase connects disconnected components first (cross-component effective
resistance is infinite / undefined on the pseudoinverse); within-component
refinement then proceeds by exact greedy.
"""

from __future__ import annotations

import heapq

import networkx as nx
import numpy as np

from .game import _DIRECTIONS, _OPPOSITE, _slew_fraction


class _UnionFind:
    """Minimal union-find for component tracking during the bridging phase."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _angle_prev(G_prev, relative_positions, num_satellites):
    """Bearing of last epoch's neighbor in each (satellite, direction) slot."""
    angle_prev = {}
    if G_prev is None:
        return angle_prev
    for i in range(num_satellites):
        if not G_prev.has_node(i):
            continue
        for nbr in G_prev.neighbors(i):
            d = G_prev[i][nbr].get("dir", {}).get(i)
            if d is not None:
                angle_prev[(i, d)] = relative_positions[i, nbr, 1]
    return angle_prev


def _laplacian_pinv(W, num_satellites):
    """Moore-Penrose pseudoinverse of the Laplacian of ``W`` (nodes 0..N-1)."""
    L = nx.laplacian_matrix(W, nodelist=range(num_satellites)).toarray().astype(float)
    return np.linalg.pinv(L)


def centralized_greedy_formation(
    G,
    positions,
    relative_positions,
    fov_matrix_front,
    fov_matrix_behind,
    fov_matrix_left,
    fov_matrix_right,
    G_prev=None,
    G_union=None,
    alpha=0.5,
):
    """Form one epoch's ISL topology by centralized greedy log-tau maximization.

    Parameters mirror :func:`satgame.game.game_theory_formation`. ``G_union`` is
    the fixed past windowed union (connectivity context); ``G_prev`` supplies the
    previous pointing for the slewing cost. Returns the populated graph ``G`` of
    this epoch's realized edges (with per-endpoint ``dir`` labels).
    """
    num_satellites = len(positions)
    fov_map = {
        "front": fov_matrix_front,
        "behind": fov_matrix_behind,
        "left": fov_matrix_left,
        "right": fov_matrix_right,
    }

    if G_union is None:
        G_union = nx.Graph()
        G_union.add_nodes_from(range(num_satellites))

    angle_prev = _angle_prev(G_prev, relative_positions, num_satellites)

    # --- Feasible candidate edges (each unordered pair once, with per-endpoint
    #     directions) that are NOT already in the past windowed union (those
    #     already contribute to the union objective, so re-selecting them is a
    #     no-op gain -- the same myopic per-epoch view the game uses). ---
    edge_dir = {}   # (a, b) with a < b -> {a: dir_a, b: dir_b}
    edge_cost = {}  # (a, b) -> alpha-free slewing cost (slew_a + slew_b)
    for i in range(num_satellites):
        for direction in _DIRECTIONS:
            fov = fov_map[direction]
            fov_opp = fov_map[_OPPOSITE[direction]]
            opp = _OPPOSITE[direction]
            row = fov[i]
            opp_col = fov_opp[:, i]
            for j in range(num_satellites):
                if i == j or row[j] != 1 or opp_col[j] != 1:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in edge_dir or G_union.has_edge(a, b):
                    continue
                da = direction if a == i else opp
                db = opp if a == i else direction
                edge_dir[(a, b)] = {a: da, b: db}
                slew_a = _slew_fraction(angle_prev.get((a, da)),
                                        relative_positions[a, b, 1])
                slew_b = _slew_fraction(angle_prev.get((b, db)),
                                        relative_positions[b, a, 1])
                edge_cost[(a, b)] = slew_a + slew_b

    used = set()  # (node, direction) terminal slots consumed this epoch

    def feasible(key):
        da_db = edge_dir[key]
        a, b = key
        return (a, da_db[a]) not in used and (b, da_db[b]) not in used

    def consume(key):
        da_db = edge_dir[key]
        a, b = key
        used.add((a, da_db[a]))
        used.add((b, da_db[b]))

    def realize(key):
        a, b = key
        G.add_edge(a, b, dir=dict(edge_dir[key]))
        consume(key)

    # --- Component tracking, seeded from the past union. ---
    uf = _UnionFind(num_satellites)
    for u, v in G_union.edges():
        uf.union(u, v)

    # --- Phase 1: bridge disconnected components (cheapest slew first). ---
    # Cross-component effective resistance is infinite, so these gains dominate
    # any within-component marginal; we connect first, breaking ties by slewing
    # cost (mirrors the game's bridge_bonus, exact here on components).
    bridged = nx.Graph()
    bridged.add_nodes_from(range(num_satellites))
    bridged.add_edges_from(G_union.edges())
    for key in sorted(edge_cost, key=edge_cost.get):
        a, b = key
        if uf.find(a) != uf.find(b) and feasible(key):
            realize(key)
            bridged.add_edge(a, b)
            uf.union(a, b)

    # --- Phase 2: exact greedy refinement within components (CELF). ---
    lpinv = _laplacian_pinv(bridged, num_satellites)

    def resistance(a, b):
        return lpinv[a, a] + lpinv[b, b] - 2.0 * lpinv[a, b]

    def net_gain(key):
        a, b = key
        return float(np.log1p(resistance(a, b))) - alpha * edge_cost[key]

    # Lazy-greedy heap of (-marginal, freshness_stamp, key). A within-component,
    # still-feasible candidate is only re-scored when it bubbles to the top.
    added = 0
    heap = []
    for key in edge_dir:
        a, b = key
        if uf.find(a) == uf.find(b) and feasible(key):
            heapq.heappush(heap, (-net_gain(key), 0, key))

    placed = set()
    while heap:
        neg_mg, stamp, key = heapq.heappop(heap)
        if key in placed or not feasible(key):
            continue
        if stamp == added:
            if -neg_mg <= 0.0:
                break  # best remaining marginal is non-positive: stop
            # Accept: realize the edge and rank-one update L_dagger.
            a, b = key
            b_vec = lpinv[:, a] - lpinv[:, b]   # L_dagger @ (e_a - e_b), O(n)
            r = b_vec[a] - b_vec[b]             # = R_ab
            lpinv = lpinv - np.outer(b_vec, b_vec) / (1.0 + r)
            realize(key)
            placed.add(key)
            added += 1
        else:
            heapq.heappush(heap, (-net_gain(key), added, key))

    return G
