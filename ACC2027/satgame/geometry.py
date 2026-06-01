"""Relative geometry and field-of-view feasibility.

Extracted verbatim from the notebook's cell 1. Each satellite carries four
terminals (front/behind/left/right); a directed pair (i, j) is feasible if j
falls inside i's directional cone and within the ISL range.
"""

from __future__ import annotations

import numpy as np


def calculate_relative_positions_all(positions, headings):
    """Distance and local bearing from every satellite to every other.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        ECI positions (meters).
    headings : np.ndarray, shape (N,)
        Heading of each satellite (degrees, 0-360), from its velocity vector.

    Returns
    -------
    relative_positions_all : np.ndarray, shape (N, N, 2)
        ``[i, j, 0]`` = distance i -> j (meters);
        ``[i, j, 1]`` = bearing of j in i's local frame (degrees, 0-360).
    """
    num_satellites = len(positions)
    relative_positions_all = np.zeros((num_satellites, num_satellites, 2))

    for i in range(num_satellites):
        diffs = positions - positions[i]
        distances = np.linalg.norm(diffs, axis=1)
        angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
        relative_angles = angles - headings[i]
        relative_angles = (relative_angles + 360) % 360
        relative_positions_all[i, :, 0] = distances
        relative_positions_all[i, :, 1] = relative_angles
    return relative_positions_all


def create_field_of_view_matrices(relative_positions_all, fov_angle, max_distance):
    """Four binary visibility matrices (front, behind, left, right).

    ``matrix[i, j] = 1`` iff satellite j lies in satellite i's directional cone
    and within ``max_distance``. Cones are centered at front=0 deg, right=90,
    behind=180, left=270, each spanning +/- ``fov_angle / 2``.
    """
    num_satellites = len(relative_positions_all)
    fov_half_angle = fov_angle / 2
    fov_matrix_front = np.zeros((num_satellites, num_satellites), dtype=int)
    fov_matrix_behind = np.zeros((num_satellites, num_satellites), dtype=int)
    fov_matrix_left = np.zeros((num_satellites, num_satellites), dtype=int)
    fov_matrix_right = np.zeros((num_satellites, num_satellites), dtype=int)

    angles = relative_positions_all[:, :, 1]
    distances = relative_positions_all[:, :, 0]
    front_mask = (360 - fov_half_angle <= angles) | (angles < fov_half_angle)
    right_mask = (90 - fov_half_angle <= angles) & (angles < 90 + fov_half_angle)
    behind_mask = (180 - fov_half_angle <= angles) & (angles < 180 + fov_half_angle)
    left_mask = (270 - fov_half_angle <= angles) & (angles < 270 + fov_half_angle)

    fov_matrix_front[(front_mask) & (distances <= max_distance)] = 1
    fov_matrix_behind[(behind_mask) & (distances <= max_distance)] = 1
    fov_matrix_left[(left_mask) & (distances <= max_distance)] = 1
    fov_matrix_right[(right_mask) & (distances <= max_distance)] = 1

    return fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right
