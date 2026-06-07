"""Best-response link formation (the noncooperative game realization).

This is the *best-response* counterpart to the matching mechanism in ``game.py``.
Same game (players = satellites, action = who to point at per terminal, payoff =
connectivity marginal minus slew, mutual-choice link rule), but instead of
clearing a stable matching, satellites repeatedly **best-respond** to each
other's current pointings until the profile settles. The solution concept is a
(linearized) Nash equilibrium of the per-epoch game.

Decision rule (per satellite i, per direction d, against the current profile):

    g_i^d(j) = ( Delta_i(j)      if j currently points back at i
              ( rho * Delta_i(j) otherwise            ) - alpha * slew_i^d(j)

    point at argmax_j g_i^d(j)  (or nobody, value 0)

where Delta_i(j) = log(1 + R_ij) is the local h-hop log-tau marginal on the fixed
windowed union, and ``rho in [0, 1]`` is the **optimism** knob for non-reciprocated
candidates:

    rho = 0  (strict)     -> never point at a non-reciprocator. From a COLD start
                             nobody points at anybody and the best response is to
                             point at nobody: the empty graph is a Nash
                             equilibrium -- the *coordination-failure trap*.
    rho = 1  (optimistic) -> ignore reciprocation, point at your favorite and hope
                             it reciprocates -> escapes the trap, but causes
                             reciprocation failures (wasted terminals).
    0<rho<1               -> prefer sure links, chase a sufficiently better
                             unreciprocated one -> genuine give-and-take dynamics.

Updates are **simultaneous** (all satellites at once), so the dynamics can cycle;
we detect convergence and cycles and cap the round count.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .game import _DIRECTIONS, _OPPOSITE, _slew_fraction
from .graph import generate_network, windowed_union


# --------------------------------------------------------------------------- #
# Local marginal estimate (same estimator the matching policy uses)
# --------------------------------------------------------------------------- #
def _make_marginal(G_union, est_radius, bridge_bonus):
    """Return a cached ``marginal(i, j)`` = log(1 + R_ij) on the h-hop view."""
    ego_cache, marg_cache = {}, {}

    def ego(i):
        s = ego_cache.get(i)
        if s is None:
            s = set(nx.ego_graph(G_union, i, radius=est_radius, undirected=True))
            ego_cache[i] = s
        return s

    def marginal(i, j):
        key = (i, j) if i < j else (j, i)
        val = marg_cache.get(key)
        if val is not None:
            return val
        H = G_union.subgraph(ego(i) | ego(j))
        comp = nx.node_connected_component(H, i)
        if j in comp:
            Hc = H.subgraph(comp).copy()
            val = float(np.log1p(nx.resistance_distance(Hc, i, j)))
        else:
            val = bridge_bonus
        marg_cache[key] = val
        return val

    return marginal


def _candidates(num_satellites, fov_map):
    """Mutually-feasible candidates per (satellite, direction)."""
    cands = {i: {d: [] for d in _DIRECTIONS} for i in range(num_satellites)}
    for i in range(num_satellites):
        for d in _DIRECTIONS:
            fov, fov_opp = fov_map[d], fov_map[_OPPOSITE[d]]
            for j in range(num_satellites):
                if i != j and fov[i, j] == 1 and fov_opp[j, i] == 1:
                    cands[i][d].append(j)
    return cands


def _angle_prev(G_prev, relative_positions, num_satellites):
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


def best_response_epoch(
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
    rho=1.0,
    init="warm",
    max_rounds=50,
    bridge_bonus=1e3,
    est_radius=3,
):
    """Form one epoch's topology by simultaneous best-response dynamics.

    Returns ``(G, diag)`` where ``diag`` carries convergence info, the
    reciprocation-failure rate, and the final profile/candidates (for the
    epsilon-Nash measurement).
    """
    n = len(positions)
    fov_map = {
        "front": fov_matrix_front, "behind": fov_matrix_behind,
        "left": fov_matrix_left, "right": fov_matrix_right,
    }
    if G_union is None:
        G_union = nx.Graph()
        G_union.add_nodes_from(range(n))

    cands = _candidates(n, fov_map)
    angle_prev = _angle_prev(G_prev, relative_positions, n)
    marginal = _make_marginal(G_union, est_radius, bridge_bonus)

    def slew(i, d, j):
        return alpha * _slew_fraction(angle_prev.get((i, d)), relative_positions[i, j, 1])

    # --- Initial profile: warm = keep last epoch's pointings; cold = empty. ---
    profile = {}  # (i, direction) -> target j
    if init == "warm" and G_prev is not None:
        for i in range(n):
            if not G_prev.has_node(i):
                continue
            for nbr in G_prev.neighbors(i):
                d = G_prev[i][nbr].get("dir", {}).get(i)
                if d is not None:
                    profile[(i, d)] = nbr

    def best_response(prof):
        """One simultaneous round: every satellite best-responds to ``prof``."""
        new = {}
        for i in range(n):
            for d in _DIRECTIONS:
                best_j, best_g = None, 0.0  # 0 = point at nobody
                opp = _OPPOSITE[d]
                for j in cands[i][d]:
                    delta = marginal(i, j)
                    reciprocates = prof.get((j, opp)) == i
                    g = (delta if reciprocates else rho * delta) - slew(i, d, j)
                    if g > best_g:
                        best_g, best_j = g, j
                if best_j is not None:
                    new[(i, d)] = best_j
        return new

    converged, cycled, rounds = False, False, 0
    seen = set()
    while rounds < max_rounds:
        new = best_response(profile)
        rounds += 1
        if new == profile:
            converged = True
            profile = new
            break
        nkey = frozenset(new.items())
        if nkey in seen:
            cycled = True
            profile = new
            break
        seen.add(frozenset(profile.items()))
        profile = new

    # --- Realize mutual links. ---
    for (i, d), j in profile.items():
        if profile.get((j, _OPPOSITE[d])) == i and i < j:
            G.add_edge(i, j, dir={i: d, j: _OPPOSITE[d]})

    # --- Reciprocation-failure rate. ---
    pointings = [(k, v) for k, v in profile.items() if v is not None]
    failures = sum(
        1 for (i, d), j in pointings if profile.get((j, _OPPOSITE[d])) != i
    )
    recip_failure = failures / len(pointings) if pointings else 0.0

    diag = {
        "rounds": rounds,
        "converged": converged,
        "cycled": cycled,
        "recip_failure": recip_failure,
        "profile": profile,
        "candidates": cands,
        "angle_prev": angle_prev,
    }
    return G, diag


def epsilon_nash(U, profile, candidates, angle_prev, relative_positions, alpha):
    """Empirical epsilon-Nash gap at the final profile.

    For each satellite, the largest it could improve its marginal-contribution
    utility by a unilateral repointing, using EXACT single-edge log-tau marginals
    log(1 + R_ij) on the realized windowed union ``U`` (vs. the local estimates
    the policy acted on). A unilateral deviation can only create a link with a
    candidate that already points back, so deviations range over reciprocators
    (per direction, additively). epsilon = max over satellites.
    """
    n = U.number_of_nodes()
    if U.number_of_edges() == 0:
        return 0.0
    L = nx.laplacian_matrix(U, nodelist=range(n)).toarray().astype(float)
    Lp = np.linalg.pinv(L)

    def R(a, b):
        return Lp[a, a] + Lp[b, b] - 2.0 * Lp[a, b]

    eps = 0.0
    for i in range(n):
        gain_i = 0.0
        for d in _DIRECTIONS:
            opp = _OPPOSITE[d]
            j0 = profile.get((i, d))
            cur = 0.0
            if j0 is not None:
                formed = U.has_edge(i, j0) and profile.get((j0, opp)) == i
                s0 = alpha * _slew_fraction(angle_prev.get((i, d)),
                                            relative_positions[i, j0, 1])
                cur = (float(np.log1p(R(i, j0))) if formed else 0.0) - s0
            best = 0.0  # dropping the terminal is always available (value 0)
            for j in candidates[i][d]:
                if profile.get((j, opp)) == i:  # only reciprocators can form links
                    s = alpha * _slew_fraction(angle_prev.get((i, d)),
                                               relative_positions[i, j, 1])
                    val = float(np.log1p(R(i, j))) - s
                    if val > best:
                        best = val
            if cur > best:
                best = cur
            gain_i += best - cur
        if gain_i > eps:
            eps = gain_i
    return eps


def run_best_response(cfg, rho, init, trajectory, measure_eps=True):
    """Multi-epoch best-response run for one (rho, init) setting.

    Returns ``{"graphs": [...], "diags": [...]}`` -- realized per-epoch graphs and
    per-epoch diagnostics (rounds, converged, cycled, recip_failure, epsilon).
    """
    from .simulate import _fov  # local import avoids any import cycle

    graphs, diags = [], []
    prev, history = None, []
    for snap in trajectory:
        n = len(snap["positions"])
        G_union = windowed_union(n, history, cfg.window)
        G, diag = best_response_epoch(
            generate_network(snap["positions"]),
            snap["positions"],
            snap["relative_positions"],
            *_fov(snap),
            G_prev=prev,
            G_union=G_union,
            alpha=cfg.alpha,
            rho=rho,
            init=init,
            bridge_bonus=cfg.bridge_bonus,
        )
        if measure_eps:
            U = G_union.copy()
            U.add_edges_from(G.edges())
            diag["epsilon"] = epsilon_nash(
                U, diag["profile"], diag["candidates"], diag["angle_prev"],
                snap["relative_positions"], cfg.alpha,
            )
        # Drop bulky fields we no longer need before storing.
        for k in ("profile", "candidates", "angle_prev"):
            diag.pop(k, None)
        prev = G
        history.append(G)
        graphs.append(G)
        diags.append(diag)
    return {"graphs": graphs, "diags": diags}
