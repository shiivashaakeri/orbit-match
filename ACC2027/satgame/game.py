"""The potential-game best-response link-formation policy.

This is the corrected ``game_theory_formation`` (see the project history for the
four bugs fixed: dead slewing cost, post-edge resistance, missing mutual-choice
/ degree cap, and snapshot-vs-windowed-union objective). Kept identical to the
notebook's cell 2 so the module and notebook never diverge.
"""

import networkx as nx
import numpy as np


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
):
    """
    Distributed ISL topology formation as one synchronous round of
    best-response in the potential game.

    For each of its four terminals (front/behind/left/right) satellite ``i``
    independently picks the feasible neighbor ``j`` maximizing the utility

        u_i(j) = log(1 + R_ij) - alpha * angle(theta_prev, theta_ij),

    where ``R_ij`` is the effective resistance between ``i`` and ``j`` measured
    in the windowed-union graph ``G_union`` BEFORE the link is added -- i.e. the
    exact marginal of log(number of spanning trees) -- and the angular term is
    the slewing cost of repointing the terminal from its previous target.

    A link ``(i, j)`` is realized only on MUTUAL choice: ``i`` points a terminal
    at ``j`` and ``j`` points the opposite terminal back at ``i``. Each satellite
    commits at most one target per terminal, so realized degree never exceeds 4,
    matching the hardware budget and the baselines.

    Parameters
    ----------
    G : nx.Graph
        Empty graph on the N satellite nodes; populated and returned.
    G_prev : nx.Graph or None
        Previous epoch's realized graph, used for the slewing cost. None at t=0.
    G_union : nx.Graph or None
        Windowed union graph G^cup(t; T) of past realized epochs, used as the
        connectivity context for the marginal R_ij. None/empty at t=0, in which
        case every candidate bridges two components.
    alpha : float
        Slewing-cost weight.
    bridge_bonus : float
        Finite utility for a candidate that bridges two components (i and j not
        connected within radius 3 of G_union). Large enough to dominate any
        in-component marginal, yet finite so the slewing cost still breaks ties
        among bridging candidates.
    """
    num_satellites = len(positions)

    fov_map = {
        "front": fov_matrix_front,
        "behind": fov_matrix_behind,
        "left": fov_matrix_left,
        "right": fov_matrix_right,
    }
    opposite_map = {
        "front": "behind",
        "behind": "front",
        "left": "right",
        "right": "left",
    }
    directions = ["front", "behind", "left", "right"]

    # --- Feasible, mutually visible candidates per (satellite, direction) ---
    candidates = {i: {d: [] for d in directions} for i in range(num_satellites)}
    for i in range(num_satellites):
        for direction in directions:
            fov = fov_map[direction]
            fov_opposite = fov_map[opposite_map[direction]]
            for j in range(num_satellites):
                if (i != j) and (fov[i, j] == 1) and (fov_opposite[j, i] == 1):
                    candidates[i][direction].append(j)

    # Connectivity context for the marginal: windowed union of past epochs.
    if G_union is None:
        G_union = nx.Graph()
        G_union.add_nodes_from(range(num_satellites))

    # --- Phase 1: simultaneous best response against the FIXED G_union ---
    # desired[i][direction] = preferred target j*, or None.
    desired = {i: {d: None for d in directions} for i in range(num_satellites)}

    for i in range(num_satellites):
        if not any(candidates[i][d] for d in directions):
            continue

        # Radius-3 ego view of i in the union graph. It depends on i only, so
        # compute it once and reuse it across all four terminals.
        G_sub = nx.ego_graph(G_union, i, radius=3, undirected=True)

        for direction in directions:
            target_list = candidates[i][direction]
            if not target_list:
                continue

            # Current pointing of this terminal = bearing of the neighbor it
            # tracked last epoch in this direction (None if none / t=0). The
            # per-endpoint "dir" dict avoids the undirected-edge ambiguity.
            current_angle = None
            if G_prev is not None and G_prev.has_node(i):
                for nbr in G_prev.neighbors(i):
                    edge_dir = G_prev[i][nbr].get("dir", {})
                    if edge_dir.get(i) == direction:
                        current_angle = relative_positions[i, nbr, 1]
                        break

            best_score, best_target, best_dist = -np.inf, None, np.inf
            for j in target_list:
                # Slewing cost: shortest angular repointing distance.
                slew = 0.0
                if current_angle is not None:
                    cand_angle = relative_positions[i, j, 1]
                    diff = abs(cand_angle - current_angle)
                    diff = min(diff, 360.0 - diff)
                    slew = alpha * (diff / 180.0)

                # Marginal of log(tau): log(1 + R_ij) with R_ij measured in the
                # union graph BEFORE adding (i, j). j in G_sub implies i and j
                # are already connected within radius 3.
                if j in G_sub:
                    R_ij = nx.resistance_distance(G_sub, i, j)
                    marginal = np.log1p(R_ij)
                else:
                    marginal = bridge_bonus  # (i, j) bridges two components

                utility = marginal - slew
                # The locked utility is silent on ties (e.g. at t=0 every
                # candidate bridges with no slew, so all utilities are equal).
                # Resolve them deterministically by physical proximity -- the
                # closer satellite is cheaper to acquire/track -- rather than by
                # the meaningless node-index order of a plain `>` comparison.
                dist_ij = relative_positions[i, j, 0]
                if (utility > best_score + 1e-12) or (
                    abs(utility - best_score) <= 1e-12 and dist_ij < best_dist
                ):
                    best_score, best_target, best_dist = utility, j, dist_ij

            desired[i][direction] = best_target

    # --- Phase 2: realize links only on mutual choice (degree <= 4 per side) ---
    for i in range(num_satellites):
        for direction in directions:
            j = desired[i][direction]
            if j is None:
                continue
            opp = opposite_map[direction]
            if desired[j][opp] == i:
                # Record each endpoint's terminal so next epoch's slewing
                # lookup is unambiguous on the undirected edge.
                G.add_edge(i, j, dir={i: direction, j: opp})

    return G
