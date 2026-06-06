#!/usr/bin/env python3
"""Exact-optimum validation on a tiny constellation.

The centralized policy (``satgame.centralized``) is a *greedy* maximizer of
``log tau``. On a small enough instance we can compute the TRUE optimum by brute
force -- exhaustively enumerating every feasible directional b-matching (each
satellite uses at most one terminal per front/behind/left/right slot) and taking
the one with the largest ``log tau``. This script does that and prints:

    OPT (brute force)   centralized greedy   game   furthest

so you can confirm (a) the greedy ceiling is essentially tight (centralized
approx OPT) and (b) report an honest price of anarchy PoA = log tau(game) /
log tau(OPT).

We evaluate on top of a **connected base graph** (a ring over the satellites,
standing in for the past windowed union). This is deliberate: the centralized
policy's exact, provably near-optimal behavior is its within-component log-tau
refinement, where the single-edge marginal log(1 + R_ij) is exact. Starting from
a connected base puts every candidate in that regime, so the test isolates the
ceiling we actually rely on. (From a disconnected empty union, greedy
degree-constrained *bridging* can dead-end on pathologically sparse tiny
instances -- a greedy-matroid artifact that does not arise on the dense real
constellation, so it is not what we validate here.)

Notes
-----
* ``log tau`` is nonlinear, so this is exhaustive search, not an ILP. Keep the
  instance tiny (control via --planes / --sats-per-plane / --isl-range-km); the
  script refuses to run if the candidate-edge count exceeds --max-edges.
* Slewing cost is zero at this single epoch (no previous pointing), so the
  objective reduces to pure ``log tau`` maximization -- the hard, nonlinear,
  submodular core. The slewing term is a linear per-edge cost greedy handles
  exactly, so it needs no separate validation.
"""

from __future__ import annotations

import argparse

import networkx as nx
import numpy as np

from config import SimConfig
from satgame.baselines import greedy_search_furthest
from satgame.centralized import centralized_greedy_formation
from satgame.constellation import (
    build_constellation,
    calculate_cartesian_coordinates,
)
from satgame.game import _DIRECTIONS, _OPPOSITE, game_theory_formation
from satgame.geometry import (
    calculate_relative_positions_all,
    create_field_of_view_matrices,
)
from satgame.graph import generate_network
from satgame.metrics import logtau


def epoch_snapshot(cfg):
    """Propagate the constellation at t=0 and build geometry + FOV matrices."""
    constellation = build_constellation(cfg)
    positions, velocities = calculate_cartesian_coordinates(constellation, 0.0)
    headings = np.degrees(np.arctan2(velocities[:, 1], velocities[:, 0]))
    headings = (headings + 360) % 360
    rel = calculate_relative_positions_all(positions, headings)
    ff, fb, fl, fr = create_field_of_view_matrices(
        rel, cfg.fov_angle_deg, int(cfg.isl_range_m)
    )
    for m in (ff, fb, fl, fr):
        np.fill_diagonal(m, 0)
    return positions, rel, (ff, fb, fl, fr)


def feasible_edges(num_satellites, fov):
    """Unordered feasible pairs with per-endpoint direction labels."""
    fov_map = dict(zip(_DIRECTIONS, fov))
    edge_dir = {}
    for i in range(num_satellites):
        for d in _DIRECTIONS:
            opp = _OPPOSITE[d]
            row, opp_col = fov_map[d][i], fov_map[opp][:, i]
            for j in range(num_satellites):
                if i == j or row[j] != 1 or opp_col[j] != 1:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in edge_dir:
                    continue
                da = d if a == i else opp
                edge_dir[(a, b)] = {a: da, b: _OPPOSITE[da]}
    return edge_dir


def ring_base(num_satellites):
    """Connected base graph (a ring over the satellites) = stand-in past union."""
    G = nx.Graph()
    G.add_nodes_from(range(num_satellites))
    for k in range(num_satellites):
        G.add_edge(k, (k + 1) % num_satellites)
    return G


def brute_force_opt(num_satellites, edge_dir, base):
    """Exhaustive maximum-``log tau`` feasible directional b-matching over base."""
    edges = list(edge_dir)
    base_edges = list(base.edges())
    best = {"logtau": float("-inf"), "edges": []}

    def evaluate(chosen):
        G = generate_network([None] * num_satellites)
        G.add_edges_from(base_edges)
        G.add_edges_from(chosen)
        val = logtau(G)
        if val > best["logtau"]:
            best["logtau"] = val
            best["edges"] = list(chosen)

    def rec(idx, chosen, used):
        if idx == len(edges):
            evaluate(chosen)
            return
        key = edges[idx]
        rec(idx + 1, chosen, used)  # skip this edge
        a, b = key
        da, db = edge_dir[key][a], edge_dir[key][b]
        if (a, da) not in used and (b, db) not in used:  # include if slots free
            chosen.append(key)
            rec(idx + 1, chosen, used | {(a, da), (b, db)})
            chosen.pop()

    rec(0, [], set())
    return best


def run_policies(positions, rel, fov, alpha, base):
    """Run game / centralized / furthest for one epoch over ``base``.

    Returns each method's log tau evaluated on base + its realized current edges.
    """
    def lt(G_current):
        U = nx.Graph()
        U.add_nodes_from(range(len(positions)))
        U.add_edges_from(base.edges())
        U.add_edges_from(G_current.edges())
        return logtau(U)

    out = {}
    out["game"] = lt(game_theory_formation(
        generate_network(positions), positions, rel, *fov,
        G_prev=None, G_union=base.copy(), alpha=alpha,
    ))
    out["centralized"] = lt(centralized_greedy_formation(
        generate_network(positions), positions, rel, *fov,
        G_prev=None, G_union=base.copy(), alpha=alpha,
    ))
    G_far, _ = greedy_search_furthest(generate_network(positions), positions, *fov)
    out["furthest"] = lt(G_far)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Tiny synthetic instance: 2x5 = 10 sats; range and FOV widened so each
    # terminal cone holds several candidates (~14 feasible edges) -- this makes
    # the k=4 budget BIND, so the policies must choose and the test is
    # discriminating, while staying small enough to brute force in well under a
    # second. (Wide cones are a stress-test artifact, not the real 60-deg config.)
    p.add_argument("--planes", type=int, default=2)
    p.add_argument("--sats-per-plane", type=int, default=5)
    p.add_argument("--isl-range-km", type=float, default=12000.0)
    p.add_argument("--fov-angle", type=float, default=150.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-edges", type=int, default=20,
                   help="refuse brute force above this many feasible edges")
    a = p.parse_args(argv)

    cfg = SimConfig(planes=a.planes, sats_per_plane=a.sats_per_plane,
                    isl_range_km=a.isl_range_km, fov_angle_deg=a.fov_angle,
                    alpha=a.alpha)
    n = cfg.n_satellites
    positions, rel, fov = epoch_snapshot(cfg)
    base = ring_base(n)
    base_pairs = {(min(u, v), max(u, v)) for u, v in base.edges()}
    # Current-epoch candidates = feasible edges not already in the (base) union.
    edge_dir = {k: v for k, v in feasible_edges(n, fov).items() if k not in base_pairs}

    print(f"Instance: {n} satellites, ISL range {a.isl_range_km:g} km, connected "
          f"ring base + {len(edge_dir)} feasible candidate edges")
    if len(edge_dir) > a.max_edges:
        raise SystemExit(
            f"Too many candidate edges ({len(edge_dir)} > {a.max_edges}) for brute "
            f"force. Shrink the instance (fewer sats / smaller --isl-range-km) or "
            f"raise --max-edges if you're sure."
        )

    opt = brute_force_opt(n, edge_dir, base)
    pol = run_policies(positions, rel, fov, a.alpha, base)

    opt_lt = opt["logtau"]
    print(f"\n{'method':>14} {'log tau':>12} {'ratio vs OPT':>14}")
    print(f"{'OPT (brute)':>14} {opt_lt:>12.4f} {1.0:>14.3f}")
    for name in ("centralized", "game", "furthest"):
        ratio = pol[name] / opt_lt if opt_lt not in (0.0, float('-inf')) else float('nan')
        print(f"{name:>14} {pol[name]:>12.4f} {ratio:>14.3f}")

    print(f"\nOPT uses {len(opt['edges'])} edges. "
          f"Price of anarchy (game / OPT) = "
          f"{pol['game'] / opt_lt:.3f}" if opt_lt > 0 else "")


if __name__ == "__main__":
    main()
