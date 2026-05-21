# orbit-match/orbitmatch/policy/lever.py
# Run: imported by other modules; not a runnable script.

"""LeverPredictive: predictive matching with two information-side knobs.

Two extensions to the paper's policy, both default off:

1. Scarcity-weighted value (Lever 1b):
   The value of edge (i, j) is multiplied by (1 + beta * s_ij) where
   s_ij is the scarcity score in [0, 1) measuring how rare the edge is
   over the full orbital horizon. Edges feasible in few epochs (rare)
   get boosted; edges feasible everywhere stay at base value.

2. History-aware reciprocation (Lever 2 option a):
   The reciprocation prediction p_ij is multiplied by (1 + gamma * h_ij)
   where h_ij is the count of (i, j) realized links in the last
   history_window epochs, divided by history_window. History can only
   boost an already-positive prediction (the eq.17 indicator must be 1
   for the history term to matter), so this should not create false
   positives or trap the policy in repeated matchings.

Both knobs default to zero. With beta = 0 and gamma = 0 the class
reduces to PredictiveMatching exactly.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from orbitmatch.feasibility.compute import feasible_neighbors
from orbitmatch.graph.spectral import kirchhoff_index
from orbitmatch.policy.base import NO_LINK
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


class LeverPredictive(PredictiveMatching):
    """PredictiveMatching with scarcity weighting and history-aware p_ij.

    Parameters
    ----------
    scarcity_beta
        Weight for the scarcity boost on the value function. Default 0.0
        (no boost; falls back to PredictiveMatching).
    history_gamma
        Weight for the history boost on the reciprocation prediction.
        Default 0.0 (no boost; falls back to PredictiveMatching).
    history_window
        Window length for the history term, in epochs. If None, defaults
        to params.T (one certificate window's worth of history).
    scarcity_horizon
        Lookahead horizon for the scarcity statistic, in epochs. If None,
        defaults to params.T (one orbital period for Walker constellations
        with T = T_orb).
    """

    name: str = "lever"

    def __init__(
        self,
        *args,
        scarcity_beta: float = 0.0,
        history_gamma: float = 0.0,
        history_window: int | None = None,
        scarcity_horizon: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if scarcity_beta < 0:
            raise ValueError(f"scarcity_beta must be non-negative; got {scarcity_beta}.")
        if history_gamma < 0:
            raise ValueError(f"history_gamma must be non-negative; got {history_gamma}.")
        self.scarcity_beta = float(scarcity_beta)
        self.history_gamma = float(history_gamma)
        self.history_window = history_window if history_window is not None else self.params.T
        self.scarcity_horizon = scarcity_horizon if scarcity_horizon is not None else self.params.T

        # Precompute the scarcity score s_ij for the whole simulation, once.
        # The score depends only on geometry, which is common knowledge.
        # Cost: O(T_max) per pair; precomputable from the feasibility tensor.
        # Cache: indexed by start epoch, but for a periodic Walker
        # geometry the scarcity is essentially constant in t; we cache a
        # single matrix using a horizon starting at t=0 and reuse it.
        # (More accurate per-epoch recomputation is possible but unnecessary
        # for the regime we test.)
        T_max = self.feasibility.shape[0]
        horizon = min(self.scarcity_horizon, T_max)
        feasible_count = self.feasibility[:horizon].sum(axis=0).astype(np.float64)
        feasible_rate = feasible_count / float(horizon)
        # Scarcity is 1 - rate, but clamp negatives to 0 in case of edge cases.
        self._scarcity = np.maximum(0.0, 1.0 - feasible_rate)
        # Zero out the diagonal so self-edges don't get scarcity weight.
        np.fill_diagonal(self._scarcity, 0.0)

        # History buffer: deque of past realized adjacency matrices, used
        # by _reciprocation_prob. Only populated when history_gamma > 0.
        self._history_buffer: deque[np.ndarray] = deque(maxlen=self.history_window)
        # Running sum of the buffer contents; updated incrementally to
        # avoid O(W * n^2) recomputation on every _reciprocation_prob call.
        self._history_sum = np.zeros((self.n, self.n), dtype=np.float64)

    def _compute_value_matrix(self, t: int) -> np.ndarray:
        """Compute V[i, j] with optional scarcity weighting.

        If scarcity_beta = 0 this is identical to the parent's value
        matrix. Otherwise V_ij is multiplied by (1 + beta * s_ij)
        elementwise.
        """
        V = super()._compute_value_matrix(t)
        if self.scarcity_beta > 0.0:
            V = V * (1.0 + self.scarcity_beta * self._scarcity)
        return V

    def _reciprocation_prob(self, i: int, j: int, t: int) -> float:
        """Eq.17 indicator, optionally multiplied by a history boost.

        With history_gamma = 0 returns the parent's 0/1 indicator. With
        history_gamma > 0 returns p_eq17 * (1 + gamma * h_ij) where h_ij
        is the running fraction of past windows in which (i, j) linked.

        Because the multiplier is applied to the eq.17 indicator
        (which is 0 or 1), history can only boost existing
        predictions, never introduce new ones. Edges that eq.17 says
        will not reciprocate keep p = 0 regardless of history.
        """
        base = super()._reciprocation_prob(i, j, t)
        if self.history_gamma == 0.0 or base == 0.0:
            return base
        # h_ij is in [0, 1].
        if len(self._history_buffer) == 0:
            h_ij = 0.0
        else:
            h_ij = float(self._history_sum[i, j]) / float(len(self._history_buffer))
        return base * (1.0 + self.history_gamma * h_ij)

    def step(self, t: int, actions: np.ndarray) -> None:
        """Run parent step, then update the history buffer with the
        realized adjacency at epoch t.
        """
        super().step(t, actions)
        if self.history_gamma > 0.0:
            adj = self._build_realized_adjacency(actions)
            # If the buffer is full, the oldest matrix is about to drop;
            # subtract it from the running sum.
            if len(self._history_buffer) == self.history_window:
                self._history_sum -= self._history_buffer[0]
            self._history_buffer.append(adj)
            self._history_sum += adj

    def _build_realized_adjacency(self, actions: np.ndarray) -> np.ndarray:
        """Symmetric adjacency matrix of mutually-formed links at this epoch."""
        adj = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            j = int(actions[i])
            if j == NO_LINK or not (0 <= j < self.n):
                continue
            if int(actions[j]) == i:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj
