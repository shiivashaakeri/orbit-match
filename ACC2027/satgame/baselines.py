"""Link-selection baselines.

Extracted verbatim from the notebook's cell 1. All three baselines reserve the
opposite directional slot on BOTH endpoints, so realized degree is capped at 4
per satellite -- the same hardware budget the game policy now respects.
"""

from __future__ import annotations

import numpy as np


def greedy_search(
    G, positions, fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right
):
    """Connect each satellite to its CLOSEST reachable neighbor per direction."""
    num_satellites = len(positions)
    max_distance = 999999999999  # sentinel for unreachable
    distance_average = []
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
            distances[fov_matrix[i] == 0] = max_distance

            while np.min(distances) != max_distance:
                closest_idx = np.argmin(distances)
                if (
                    existing_connections[i][direction] is None
                    and existing_connections[closest_idx][opposite] is None
                ):
                    existing_connections[i][direction] = closest_idx
                    existing_connections[closest_idx][opposite] = i
                    distance_average.append(distances[closest_idx])
                    break
                else:
                    distances[closest_idx] = max_distance

    for satellite, connections in existing_connections.items():
        for direction, target in connections.items():
            if target is not None and not G.has_edge(satellite, target):
                G.add_edge(satellite, target)
    return G, distance_average


def find_min_degree_node(G, excluded_nodes):
    min_degree = min(G.degree(v) for v in G.nodes())
    min_degree_nodes = [
        v for v in G.nodes() if G.degree(v) == min_degree and v not in excluded_nodes
    ]
    if not min_degree_nodes:
        return None
    return min(min_degree_nodes, key=lambda v: extendibility_centrality(G, v))


def find_max_distance_node(positions, start_node):
    distances = np.linalg.norm(positions - positions[start_node], axis=1)
    distances[start_node] = -np.inf
    return np.argmax(distances)


def extendibility_centrality(G, node):
    neighbors = list(G.neighbors(node))
    return sum(G.degree(v) for v in neighbors)


def mdmd(
    G, positions, fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right
):
    """Minimum-Degree / Maximum-Distance heuristic."""
    num_satellites = len(positions)
    distance_average = []
    existing_connections = {
        i: {"front": None, "behind": None, "left": None, "right": None}
        for i in range(num_satellites)
    }
    excluded_nodes = set()

    for i in range(num_satellites):
        min_degree_node = find_min_degree_node(G, excluded_nodes)
        if min_degree_node is None:
            break
        connection_made = False

        for direction, fov_matrix, opposite in zip(
            ["front", "behind", "left", "right"],
            [fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right],
            ["behind", "front", "right", "left"],
        ):
            distances = np.linalg.norm(positions - positions[min_degree_node], axis=1)
            distances[fov_matrix[min_degree_node] == 0] = -1
            while np.max(distances) != -1:
                furthest_idx = np.argmax(distances)
                if (
                    existing_connections[min_degree_node][direction] is None
                    and existing_connections[furthest_idx][opposite] is None
                ):
                    existing_connections[min_degree_node][direction] = furthest_idx
                    existing_connections[furthest_idx][opposite] = min_degree_node
                    G.add_edge(furthest_idx, min_degree_node)
                    distance_average.append(distances[furthest_idx])
                    connection_made = True
                    break
                else:
                    distances[furthest_idx] = -1
        if not connection_made:
            excluded_nodes.add(min_degree_node)
    return G, distance_average


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
