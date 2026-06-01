# orbit-match/orbitmatch/policy/adaptive.py
# Run: imported by other modules; not a runnable script.

"""Adaptive-depth predictive matching.

Idea: most satellites at most epochs face an unambiguous choice -- their
top candidate scores much higher than their second-best, so the level-1
prediction is essentially as good as level-infinity. Only the satellites
with a *close* top-two gap benefit from deeper strategic reasoning.

This module implements that observation:

1. Every satellite computes its level-1 (predictive) scores.
2. Each satellite measures the gap between its top two non-zero scores.
3. Satellites with gap >= delta_threshold commit to the level-1 choice.
4. The rest (the "ambiguous" satellites) run k_max - 1 additional rounds
   of synchronous BR. Their level-k decisions are committed.

The non-ambiguous satellites keep their level-1 decisions throughout
the recursion -- they don't update. Only the ambiguous set iterates.

Computational saving
--------------------
If only fraction p << 1 of satellites are ambiguous per epoch, the cost
is roughly (1 + p * (k_max - 1)) times level-1 cost, vs. k_max times
for the uniform k-step policy. Empirically (see scripts/run_k_ablation
results) p is small for Walker constellations, so the savings are large.

Diagnostics recorded per epoch
------------------------------
- ``n_ambiguous``: how many satellites had gap < delta_threshold.
- ``ambiguous_frac``: n_ambiguous / n.
- ``k_max``: the depth used when escalating (constant).
- ``delta_threshold``: the threshold used (constant).
"""

from __future__ import annotations

import numpy as np

from orbitmatch.feasibility.compute import feasible_neighbors
from orbitmatch.policy.base import NO_LINK, Policy
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


DEFAULT_DELTA: float = 0.10
DEFAULT_K_MAX: int = 3


class AdaptivePredictive(PredictiveMatching):
    """Adaptive-depth predictive matching.

    Parameters
    ----------
    delta_threshold
        Gap below which a satellite is considered ambiguous and
        escalated to k_max. Measured on the normalized score scale,
        so 0.1 means "top score within 10% of second-best."
    k_max
        Recursion depth used when a satellite is ambiguous. k_max = 1
        means no escalation (equivalent to PredictiveMatching).
    """

    name: str = "adaptive"

    def __init__(
        self,
        *args,
        delta_threshold: float = DEFAULT_DELTA,
        k_max: int = DEFAULT_K_MAX,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if delta_threshold < 0:
            raise ValueError(f"delta_threshold must be >= 0; got {delta_threshold}.")
        if k_max < 1:
            raise ValueError(f"k_max must be >= 1; got {k_max}.")
        self.delta_threshold = delta_threshold
        self.k_max = k_max

    def decide(self, t: int) -> np.ndarray:
        """Adaptive decision: level-1 by default, escalate to k_max if ambiguous."""
        # Populate caches.
        if self._cached_value_epoch != t:
            self._cached_value_matrix = self._compute_value_matrix(t)
            self._cached_value_epoch = t
        if self._cached_top_partner_epoch != t:
            self._cached_top_partners = self._compute_top_value_partners(t)
            self._cached_top_partner_epoch = t

        # Step 1: every satellite computes its level-1 action and its top-two gap.
        a_level1, gaps = self._level1_actions_and_gaps(t)
        ambiguous = gaps < self.delta_threshold
        n_ambiguous = int(ambiguous.sum())

        # Diagnostics.
        self._record("n_ambiguous", n_ambiguous)
        self._record("ambiguous_frac", float(n_ambiguous) / self.n)
        self._record("k_max", self.k_max)
        self._record("delta_threshold", self.delta_threshold)

        # Step 2: if no ambiguous satellites or k_max == 1, return level-1.
        if n_ambiguous == 0 or self.k_max == 1:
            return a_level1

        # Step 3: escalate. Run synchronous BR (k_max - 1) more rounds,
        # but only update the actions of ambiguous satellites. Non-ambiguous
        # satellites' actions are frozen at their level-1 commitments and
        # treated as constants by the BR sweeps -- they are part of the
        # "environment" the ambiguous ones best-respond against.
        a = a_level1.copy()
        ambiguous_idx = np.flatnonzero(ambiguous)
        for _ in range(self.k_max - 1):
            a_new = a.copy()
            for i in ambiguous_idx:
                a_new[i] = self._best_response_against(int(i), t, a)
            a = a_new

        return a

    def step(self, t: int, actions: np.ndarray) -> None:
        """Update pointing and invalidate caches; no deferral diagnostic."""
        Policy.step(self, t, actions)
        self._cached_baseline_epoch = -1
        self._cached_baseline_matrix = None
        self._cached_baseline_omega = None
        self._cached_value_epoch = -1
        self._cached_value_matrix = None
        self._cached_top_partner_epoch = -1
        self._cached_top_partners = None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _level1_actions_and_gaps(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute every satellite's level-1 action and its top-two score gap.

        The level-1 score for satellite i and candidate j is the same as
        in the predictive policy:
            s_ij = p_ij^(1-step) * (V_tilde_i(j) - c * C_tilde_ij)
        where p_ij^(1-step) reads from the top-partner cache (j's top-V
        choice).

        The "gap" is the difference between the top two non-zero scores
        among i's candidates. If only one non-zero candidate exists, the
        gap is infinity (unambiguous). If none, the gap is also
        infinity (unambiguous: i plays varnothing).
        """
        c = self.params.switching_cost_scale
        actions = np.full(self.n, NO_LINK, dtype=np.int64)
        gaps = np.full(self.n, np.inf, dtype=np.float64)

        top_partners = self._cached_top_partners  # j's top-V partner

        for i in range(self.n):
            candidates = feasible_neighbors(self.feasibility, t, i)
            if len(candidates) == 0:
                continue

            n_cands = len(candidates)
            v_raw = np.zeros(n_cands, dtype=np.float64)
            for k, j in enumerate(candidates):
                v_raw[k] = self._value(i, int(j), t)
            v_max = v_raw.max()
            if v_max <= 0.0:
                continue

            scores = np.zeros(n_cands + 1, dtype=np.float64)
            for k, j in enumerate(candidates):
                v_tilde = v_raw[k] / v_max
                c_raw = self._switching_cost(i, int(j), t)
                c_tilde = c_raw / np.pi
                p = 1.0 if int(top_partners[int(j)]) == i else 0.0
                scores[k] = p * (v_tilde - c * c_tilde)
            # scores[-1] = 0 (varnothing).

            actions[i] = self._argmax_with_varnothing_preference(scores, candidates)

            # Gap: top two non-zero scores. If varnothing is the choice (no
            # non-zero positive scores), the choice is unambiguous.
            positive = scores[scores > 0]
            if positive.size < 2:
                gaps[i] = np.inf  # zero or one positive: nothing to be ambiguous about
            else:
                top_two = np.partition(positive, -2)[-2:]
                gaps[i] = float(top_two[1] - top_two[0])

        return actions, gaps

    def _best_response_against(self, i: int, t: int, a: np.ndarray) -> int:
        """Best response of satellite i to the iterate a (Jacobi-style)."""
        candidates = feasible_neighbors(self.feasibility, t, i)
        if len(candidates) == 0:
            return NO_LINK

        c = self.params.switching_cost_scale
        n_candidates = len(candidates)

        v_raw = np.zeros(n_candidates, dtype=np.float64)
        for kk, j in enumerate(candidates):
            v_raw[kk] = self._value(i, int(j), t)
        v_max = v_raw.max()
        if v_max <= 0.0:
            return NO_LINK

        scores = np.zeros(n_candidates + 1, dtype=np.float64)
        for kk, j in enumerate(candidates):
            v_tilde = v_raw[kk] / v_max
            c_raw = self._switching_cost(i, int(j), t)
            c_tilde = c_raw / np.pi
            p = 1.0 if int(a[int(j)]) == i else 0.0
            scores[kk] = p * (v_tilde - c * c_tilde)

        return self._argmax_with_varnothing_preference(scores, candidates)
