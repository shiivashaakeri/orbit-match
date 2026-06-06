"""Decentralized link-selection baseline.

``greedy_search_furthest`` is the decentralized reference policy: each satellite
independently points each terminal at its FURTHEST reachable in-cone neighbor,
which shortens hop count across the constellation. A link forms only if both
endpoints have the paired slot free (i's front <-> target's behind, i's left <->
target's right), so realized degree stays <= 4 -- the same k=4 hardware budget
the game and centralized policies respect.

The earlier greedy-closest and minimum-degree/maximum-distance (MDMD) heuristics
were removed: the experiment now contrasts the decentralized game against this
single decentralized heuristic and the centralized greedy ceiling
(``satgame.centralized``).
"""

from __future__ import annotations

import numpy as np


def greedy_search_furthest(
    G, positions, fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right
):
    """Connect each satellite to its FURTHEST reachable neighbor per direction.

    Connecting to the furthest in-cone satellite reduces hop count across the
    constellation. A link forms only if both endpoints have the paired slot free
    (i's front <-> target's behind, i's left <-> target's right), so degree <= 4.
    """
    num_satellites = len(positions)
    distance_average = []
    fov_map = {
        "front": fov_matrix_front,
        "behind": fov_matrix_behind,
        "left": fov_matrix_left,
        "right": fov_matrix_right,
    }
    existing_connections = {
        i: {"front": None, "behind": None, "left": None, "right": None}
        for i in range(num_satellites)
    }

    for i in range(num_satellites):
        for direction, fov_matrix, opposite in zip(
            ["front", "behind", "left", "right"],
            [fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right],
            ["behind", "front", "right", "left"],
        ):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[fov_matrix[i] == 0] = -1

            while np.max(distances) != -1:
                furthest_idx = np.argmax(distances)
                # Mutual-visibility (bilateral) check: the link only forms if i is
                # also inside furthest_idx's opposite cone -- the same mutual-choice
                # constraint the game and centralized policies obey. Without it the
                # baseline forms geometrically one-sided links and is not comparable.
                if fov_map[opposite][furthest_idx, i] != 1:
                    distances[furthest_idx] = -1
                    continue
                if (
                    existing_connections[i][direction] is None
                    and existing_connections[furthest_idx][opposite] is None
                ):
                    existing_connections[i][direction] = furthest_idx
                    existing_connections[furthest_idx][opposite] = i
                    distance_average.append(distances[furthest_idx])
                    break
                else:
                    distances[furthest_idx] = -1

    for satellite, connections in existing_connections.items():
        for direction, target in connections.items():
            if target is not None and not G.has_edge(satellite, target):
                G.add_edge(satellite, target)

    return G, distance_average
