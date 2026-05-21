# orbit-match/orbitmatch/policy/k_step.py
# Run: imported by other modules; not a runnable script.

"""k-step predictive matching: an explicit recursion-depth family.

The paper of Section IV.D notes that the predictive policy with the
one-step reciprocation predictor is one round of best-response (BR)
dynamics on the potential game Gamma_t, and that "richer reciprocation
predictors correspond to deeper truncations of best-response dynamics."
This module implements that family explicitly: KStepPredictive runs
exactly k rounds of synchronous BR, starting from a level-0
initialization (every satellite picks its top-V partner, ignoring
reciprocation).

Boundary cases (verified numerically in scripts/check_k_step.py):

  k = 0 -> identical to GreedyMatching (every satellite picks
           top-V partner, no reciprocation logic).
  k = 1 -> identical to PredictiveMatching (each satellite does
           one BR against the level-0 profile, which is exactly
           the one-step indicator p_ij = 1{BR_j(a_-j) = i}).
  k -> infty -> converges to a fixed point of BR, i.e., a NE of
           Gamma_t. Equivalent to EquilibriumMatching minus the
           early-exit on convergence.

The synchronous (Jacobi) update is chosen rather than Gauss-Seidel
because the paper's level-k thinking is symmetric across satellites:
when i models j at level k-1, j models i at level k-1 also; there is
no canonical ordering. Synchronous BR can in principle cycle on a
potential game, but at small k (1, 2, 3) on our geometries this is
rare and benign -- we are not iterating to convergence here.
"""

from __future__ import annotations

import numpy as np

from orbitmatch.feasibility.compute import feasible_neighbors
from orbitmatch.policy.base import NO_LINK, Policy
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


class KStepPredictive(PredictiveMatching):
    """Predictive matching with exactly k rounds of synchronous BR.

    At decision time for epoch t:
    1. Initialize a^(0) by every satellite picking its top-V partner
       (the level-0 profile, identical to GreedyMatching's choice).
    2. For r = 1..k: simultaneously update every satellite's action
       to its best response against a^(r-1).
    3. Return a^(k).

    The k = 1 case reduces to the standard predictive policy because
    best-responding to the level-0 profile gives exactly the one-step
    indicator p_ij = 1{a^(0)_j = i} = 1{j's top-V partner is i}.

    Diagnostics recorded per epoch
    ------------------------------
    - ``k_step_value``: the fixed k used (constant; useful when
      comparing across k in sweeps).
    """

    name: str = "k_step"

    def __init__(self, *args, k: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if k < 0:
            raise ValueError(f"k must be non-negative; got {k}.")
        self.k = k

    def decide(self, t: int) -> np.ndarray:
        """Run exactly self.k rounds of synchronous BR at epoch t."""
        # Populate value-matrix and top-partner caches once.
        if self._cached_value_epoch != t:
            self._cached_value_matrix = self._compute_value_matrix(t)
            self._cached_value_epoch = t
        if self._cached_top_partner_epoch != t:
            self._cached_top_partners = self._compute_top_value_partners(t)
            self._cached_top_partner_epoch = t

        # Level-0 profile: every satellite picks its top-V partner.
        # _compute_top_value_partners gives exactly that.
        a = self._cached_top_partners.copy()

        # Synchronous BR: at each round, compute every i's best response
        # against the current (frozen) joint action, then commit all
        # updates simultaneously.
        for _ in range(self.k):
            a_new = np.full(self.n, NO_LINK, dtype=np.int64)
            for i in range(self.n):
                a_new[i] = self._best_response_against(i, t, a)
            a = a_new

        self._record("k_step_value", self.k)
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

    def _best_response_against(self, i: int, t: int, a: np.ndarray) -> int:
        """Best response of satellite i to the joint action a.

        Same score form as the predictive policy, but with the
        reciprocation factor read from the iterate a rather than from
        the level-0 cache:
            p_ij(a) = 1 iff a[j] == i, else 0.
        Tie-breaks: varnothing wins ties.
        """
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
        # scores[-1] = 0 (varnothing).

        return self._argmax_with_varnothing_preference(scores, candidates)
