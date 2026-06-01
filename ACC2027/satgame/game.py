"""Predictive ISL link formation as a matching game.

This implements the policy described in the project writeups: at each epoch every
satellite scores its feasible neighbors by the single-edge marginal of
``log tau`` minus a slewing cost, and links are realized by a **stable matching**
(deferred acceptance) rather than a one-shot optimistic pick. Matching directly
realizes the mutual-choice rule and avoids the wasted-terminal reciprocation
failure of a single synchronous round.

Per terminal, satellite ``i`` ranks candidate ``j`` by

    score_i(j) = log(1 + R_ij) - alpha * angle(theta_i^prev, theta_ij),

where ``R_ij`` is the effective resistance between ``i`` and ``j`` estimated
locally on the union of ``i``'s and ``j``'s h-hop neighborhoods in the windowed
union graph ``G_union`` (measured BEFORE the edge is added -- the exact marginal
of log #spanning-trees). Terminals pair up by direction (front<->behind,
left<->right); each is a stable matching, so realized degree never exceeds 4.
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np

_DIRECTIONS = ["front", "behind", "left", "right"]
_OPPOSITE = {"front": "behind", "behind": "front", "left": "right", "right": "left"}
# The two independent matchings: (proposer direction, receiver direction).
_DIRECTION_PAIRS = [("front", "behind"), ("left", "right")]


def _slew_fraction(angle_prev, cand_angle):
    """Normalized angular repointing distance in [0, 1] (0 if no prior pointing)."""
    if angle_prev is None:
        return 0.0
    diff = abs(cand_angle - angle_prev)
    diff = min(diff, 360.0 - diff)
    return diff / 180.0


def _gale_shapley(proposer_pref, receiver_score):
    """Deferred-acceptance stable matching (capacity 1 on both sides).

    Parameters
    ----------
    proposer_pref : dict[int, list[int]]
        proposer -> receivers in descending preference order.
    receiver_score : callable(receiver, proposer) -> comparable
        Higher is better; used by each receiver to compare its suitors.

    Returns
    -------
    dict[int, int]
        receiver -> matched proposer.
    """
    held = {}        # receiver -> proposer
    held_score = {}  # receiver -> score of held proposer
    nxt = {p: 0 for p in proposer_pref}
    free = deque(p for p in proposer_pref if proposer_pref[p])

    while free:
        p = free.popleft()
        prefs = proposer_pref[p]
        while nxt[p] < len(prefs):
            r = prefs[nxt[p]]
            nxt[p] += 1
            sc = receiver_score(r, p)
            if r not in held:
                held[r] = p
                held_score[r] = sc
                break
            if sc > held_score[r]:
                bumped = held[r]
                held[r] = p
                held_score[r] = sc
                if nxt[bumped] < len(proposer_pref[bumped]):
                    free.append(bumped)
                break
            # else: rejected, try next preference
    return held


def game_theory_formation(
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
    bridge_bonus=1e3,
    est_radius=3,
):
    """Form one epoch's ISL topology as a stable matching game.

    Parameters
    ----------
    G : nx.Graph
        Empty graph on the N satellite nodes; populated and returned.
    G_prev : nx.Graph or None
        Previous epoch's realized graph (for slewing cost). None at t=0.
    G_union : nx.Graph or None
        Windowed union graph G^cup(t; T); connectivity context for the marginal.
        None/empty at t=0, so every candidate bridges two components.
    alpha : float
        Slewing-cost weight.
    bridge_bonus : float
        Finite score for a pair the local estimate cannot connect (different
        components of the local subgraph) -- large enough to dominate any
        in-component marginal, finite so slewing/proximity break ties.
    est_radius : int
        Hop radius of each satellite's local neighborhood used to estimate R_ij.
    """
    num_satellites = len(positions)

    fov_map = {
        "front": fov_matrix_front,
        "behind": fov_matrix_behind,
        "left": fov_matrix_left,
        "right": fov_matrix_right,
    }

    # --- Feasible, mutually visible candidates per (satellite, direction) ---
    candidates = {i: {d: [] for d in _DIRECTIONS} for i in range(num_satellites)}
    for i in range(num_satellites):
        for direction in _DIRECTIONS:
            fov = fov_map[direction]
            fov_opposite = fov_map[_OPPOSITE[direction]]
            for j in range(num_satellites):
                if (i != j) and (fov[i, j] == 1) and (fov_opposite[j, i] == 1):
                    candidates[i][direction].append(j)

    if G_union is None:
        G_union = nx.Graph()
        G_union.add_nodes_from(range(num_satellites))

    # --- Current terminal pointing per (satellite, direction), from G_prev ---
    # angle_prev[(i, direction)] = bearing of last epoch's neighbor in that
    # direction (None if none). The per-endpoint "dir" dict disambiguates the
    # undirected edge.
    angle_prev = {}
    if G_prev is not None:
        for i in range(num_satellites):
            if not G_prev.has_node(i):
                continue
            for nbr in G_prev.neighbors(i):
                edge_dir = G_prev[i][nbr].get("dir", {})
                d = edge_dir.get(i)
                if d is not None:
                    angle_prev[(i, d)] = relative_positions[i, nbr, 1]

    # --- Local effective-resistance estimate (cached per unordered pair) ---
    ego_cache = {}
    marginal_cache = {}

    def ego(i):
        s = ego_cache.get(i)
        if s is None:
            s = set(nx.ego_graph(G_union, i, radius=est_radius, undirected=True))
            ego_cache[i] = s
        return s

    def marginal(i, j):
        """log(1 + R_ij) on the union of i's and j's neighborhoods; bridge bonus
        if the local view cannot connect them."""
        key = (i, j) if i < j else (j, i)
        val = marginal_cache.get(key)
        if val is not None:
            return val
        H = G_union.subgraph(ego(i) | ego(j))
        comp = nx.node_connected_component(H, i)
        if j in comp:
            Hc = H.subgraph(comp).copy()
            val = float(np.log1p(nx.resistance_distance(Hc, i, j)))
        else:
            val = bridge_bonus  # pair bridges two (locally visible) components
        marginal_cache[key] = val
        return val

    # --- Two stable matchings: front<->behind and left<->right ---
    for pdir, rdir in _DIRECTION_PAIRS:
        # Proposer preference lists (best first); tie-break by proximity.
        proposer_pref = {}
        for i in range(num_satellites):
            cand = candidates[i][pdir]
            if not cand:
                continue
            a_prev = angle_prev.get((i, pdir))

            def p_key(j, i=i, a_prev=a_prev):
                score = marginal(i, j) - alpha * _slew_fraction(
                    a_prev, relative_positions[i, j, 1]
                )
                return (score, -relative_positions[i, j, 0])

            proposer_pref[i] = sorted(cand, key=p_key, reverse=True)

        # Receiver scoring: receiver r ranks suitor p by its own utility.
        def receiver_score(r, p):
            a_prev = angle_prev.get((r, rdir))
            score = marginal(p, r) - alpha * _slew_fraction(
                a_prev, relative_positions[r, p, 1]
            )
            return (score, -relative_positions[r, p, 0])

        matched = _gale_shapley(proposer_pref, receiver_score)
        for r, p in matched.items():
            G.add_edge(p, r, dir={p: pdir, r: rdir})

    return G
