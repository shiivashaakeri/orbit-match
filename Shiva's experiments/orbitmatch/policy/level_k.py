# orbit-match/orbitmatch/policy/level_k.py
# Run: imported by other modules; not a runnable script.

"""Level-$k$ predictive matching: cognitive hierarchy on the matching game.

The paper of Section IV.D Remark notes that the predictive policy with
the one-step reciprocation indicator is one round of best-response on
the potential game Gamma_t, and that "richer reciprocation predictors
correspond to deeper truncations of best-response dynamics." This
module makes "deeper truncations" concrete by interpreting depth in
the level-$k$ thinking sense, not as best-response iteration.

The construction
----------------
At level $k$, satellite $i$ models every other satellite $j$ as a
level-$(k-1)$ agent. The reciprocation prediction becomes:
    p^{(k)}_{ij}(t) = 1 iff a^{(k-1)}_j(t) = i,
where a^{(k-1)}_j(t) is satellite $j$'s best response under level-(k-1)
reasoning. The base case is level 0: every satellite is modeled as
greedy (picks top-V partner, ignoring reciprocation).

This is different from best-response iteration. In BR iteration,
every satellite updates synchronously against the previous round's
joint action; level-$k$ is a single computation in which each
satellite reasons about $k$-deep nested counterfactuals about others.

The two are related: at the fixed point of best-response dynamics,
both yield the same joint action. But for finite $k <$ fixed-point
depth, they differ.

Boundary cases (verified in scripts/check_level_k.py)
-----------------------------------------------------
  level = 0   identical to GreedyMatching (every satellite picks top-V).
  level = 1   identical to PredictiveMatching (the paper's policy).
  level >= 2  new content; converges to a NE in finite levels.

Computational cost
------------------
Computing satellite $i$'s level-$k$ best response requires recursive
computation of every other satellite's level-$(k-1)$ best response,
each of which requires every satellite's level-$(k-2)$ best response,
etc. Without memoization the cost is exponential in $k$. With proper
caching at the (satellite, level) granularity, the cost is roughly
$O(n \\cdot k \\cdot \\Delta)$ per epoch, where Delta is the average
feasibility degree.
"""

from __future__ import annotations

import numpy as np

from orbitmatch.feasibility.compute import feasible_neighbors
from orbitmatch.policy.base import NO_LINK, Policy
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


class LevelKPredictive(PredictiveMatching):
    """Predictive matching with cognitive depth $k$.

    Parameters
    ----------
    level
        Cognitive depth (>= 0). At level $k$, satellites model each
        other as level-$(k-1)$ agents. level = 1 reduces to the paper's
        predictive policy.

    Diagnostics recorded per epoch
    ------------------------------
    - ``level``: the fixed level used (constant; for sweep bookkeeping).
    """

    name: str = "level_k"

    def __init__(self, *args, level: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if level < 0:
            raise ValueError(f"level must be non-negative; got {level}.")
        self.level = level
        # Per-epoch memo: (satellite_id, depth) -> action under that depth.
        # Cleared at the start of each decide() call.
        self._level_action_memo: dict[tuple[int, int], int] = {}

    def decide(self, t: int) -> np.ndarray:
        """Compute every satellite's level-self.level best response at epoch t."""
        # Populate value-matrix and top-partner caches once.
        if self._cached_value_epoch != t:
            self._cached_value_matrix = self._compute_value_matrix(t)
            self._cached_value_epoch = t
        if self._cached_top_partner_epoch != t:
            self._cached_top_partners = self._compute_top_value_partners(t)
            self._cached_top_partner_epoch = t

        # Fresh memo for this epoch. Cached recursive results are valid
        # only within an epoch because the value matrix changes with t.
        self._level_action_memo.clear()

        actions = np.full(self.n, NO_LINK, dtype=np.int64)
        for i in range(self.n):
            actions[i] = self._level_k_action(i, t, self.level)

        self._record("level", self.level)
        return actions

    def _level_k_action(self, i: int, t: int, depth: int) -> int:
        """Satellite $i$'s best response at cognitive depth $depth$, at epoch $t$.

        depth = 0: greedy (top-V partner, ignoring reciprocation).
        depth = 1: paper's predictive (assume others are greedy).
        depth >= 2: assume others are at depth-1.

        Memoized on (i, depth) so the recursion is polynomial, not
        exponential.
        """
        if depth < 0:
            raise ValueError(f"depth must be non-negative; got {depth}.")

        key = (i, depth)
        cached = self._level_action_memo.get(key)
        if cached is not None:
            return cached

        if depth == 0:
            # Greedy: pick top-V partner regardless of reciprocation.
            # The top-partner cache already encodes this.
            assert self._cached_top_partners is not None
            action = int(self._cached_top_partners[i])
            self._level_action_memo[key] = action
            return action

        # depth >= 1: compute the score for every candidate and pick the max.
        candidates = feasible_neighbors(self.feasibility, t, i)
        if len(candidates) == 0:
            self._level_action_memo[key] = NO_LINK
            return NO_LINK

        c = self.params.switching_cost_scale
        n_candidates = len(candidates)

        v_raw = np.zeros(n_candidates, dtype=np.float64)
        for kk, j in enumerate(candidates):
            v_raw[kk] = self._value(i, int(j), t)
        v_max = v_raw.max()
        if v_max <= 0.0:
            self._level_action_memo[key] = NO_LINK
            return NO_LINK

        scores = np.zeros(n_candidates + 1, dtype=np.float64)
        for kk, j in enumerate(candidates):
            v_tilde = v_raw[kk] / v_max
            c_raw = self._switching_cost(i, int(j), t)
            c_tilde = c_raw / np.pi
            # Reciprocation prediction at depth $depth$: 1 iff j's level-(depth-1)
            # action is i.
            j_action_at_lower = self._level_k_action(int(j), t, depth - 1)
            p = 1.0 if j_action_at_lower == i else 0.0
            scores[kk] = p * (v_tilde - c * c_tilde)
        # scores[-1] = 0 (varnothing).

        action = self._argmax_with_varnothing_preference(scores, candidates)
        self._level_action_memo[key] = action
        return action

    def step(self, t: int, actions: np.ndarray) -> None:
        """Update pointing and invalidate caches."""
        Policy.step(self, t, actions)
        self._cached_baseline_epoch = -1
        self._cached_baseline_matrix = None
        self._cached_baseline_omega = None
        self._cached_value_epoch = -1
        self._cached_value_matrix = None
        self._cached_top_partner_epoch = -1
        self._cached_top_partners = None
        self._level_action_memo.clear()
