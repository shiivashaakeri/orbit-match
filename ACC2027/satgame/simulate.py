"""Simulation driver: precompute trajectories, then form topologies per epoch.

Mirrors the notebook's cell 4, with the corrected game loop: at each epoch the
game best-responds against the FIXED windowed-union graph G^cup(t; T) (so the
round is simultaneous and order-independent), and links are realized by mutual
choice with the degree cap inside ``game_theory_formation``.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .baselines import greedy_search, greedy_search_furthest, mdmd
from .constellation import build_constellation, calculate_cartesian_coordinates
from .game import game_theory_formation
from .geometry import calculate_relative_positions_all, create_field_of_view_matrices
from .graph import generate_network, windowed_union

# Baseline name -> function (each returns (graph, distances)).
_BASELINES = {
    "furthest": greedy_search_furthest,
    "greedy": greedy_search,
    "mdmd": mdmd,
}


def _fov(snap):
    """Unpack the four FOV matrices from a trajectory snapshot, in order."""
    return (snap["fov_front"], snap["fov_behind"], snap["fov_left"], snap["fov_right"])


def precompute_trajectories(cfg, progress: bool = True):
    """Propagate the constellation and build per-epoch geometry + FOV matrices."""
    constellation = build_constellation(cfg)
    trajectory = []

    times = cfg.times
    iterator = tqdm(times, desc="Propagating trajectories") if progress else times
    for t in iterator:
        positions, velocities = calculate_cartesian_coordinates(constellation, t)
        headings = np.degrees(np.arctan2(velocities[:, 1], velocities[:, 0]))
        headings = (headings + 360) % 360
        relative_positions = calculate_relative_positions_all(positions, headings)

        ff, fb, fl, fr = create_field_of_view_matrices(
            relative_positions, cfg.fov_angle_deg, int(cfg.isl_range_m)
        )
        for m in (ff, fb, fl, fr):
            np.fill_diagonal(m, 0)

        trajectory.append({
            "positions": positions,
            "relative_positions": relative_positions,
            "fov_front": ff,
            "fov_behind": fb,
            "fov_left": fl,
            "fov_right": fr,
        })
    return trajectory


def run_simulation(cfg, trajectory=None, progress: bool = True):
    """Run the configured methods over every epoch.

    Returns
    -------
    dict[str, list[nx.Graph]]
        Method name -> list of realized graphs, one per epoch.
    """
    if trajectory is None:
        trajectory = precompute_trajectories(cfg, progress=progress)

    methods = list(cfg.methods)
    graphs = {m: [] for m in methods}

    # Game state carried across epochs.
    G_prev = None
    game_history = []

    iterator = tqdm(trajectory, desc="Simulating network") if progress else trajectory
    for snap in iterator:
        positions = snap["positions"]
        n = len(positions)

        if "game" in methods:
            # Fixed connectivity context = windowed union of past realized graphs.
            G_union = windowed_union(n, game_history, cfg.window)
            G_game = game_theory_formation(
                generate_network(positions),
                positions,
                snap["relative_positions"],
                *_fov(snap),
                G_prev=G_prev,
                G_union=G_union,
                alpha=cfg.alpha,
                bridge_bonus=cfg.bridge_bonus,
            )
            G_prev = G_game
            game_history.append(G_game)
            graphs["game"].append(G_game)

        for name, fn in _BASELINES.items():
            if name in methods:
                G_b, _ = fn(generate_network(positions), positions, *_fov(snap))
                graphs[name].append(G_b)

    return graphs
