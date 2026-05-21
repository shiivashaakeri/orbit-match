# orbit-match/orbitmatch/policy/equilibrium.py
# Run: imported by other modules; not a runnable script.

"""Equilibrium baseline: iterated best-response to convergence on Gamma_t.

This is the centralized-ceiling reference for the paper. Where the
predictive policy of Sec III is one round of BR dynamics on the
per-epoch matching game (Prop 8), this policy iterates BR until a
fixed point is reached -- a pure-strategy Nash equilibrium of
Gamma_t. By Corollary 3 (Gamma_t admits an exact potential
W_t, hence finite improvement property), this terminates in finitely
many rounds from any initial profile.

Design choices
--------------
- **Initialization**: each epoch starts from the predictive policy's
  one-step BR profile. Two reasons: it is the warm-start (fewest
  additional rounds to convergence in practice), and it makes the
  semantics clean -- equilibrium = "what the predictive policy would
  produce if it iterated to convergence."
- **Update order**: asynchronous (Gauss-Seidel), satellites updated
  in index order within each round. Each satellite best-responds to
  the most recent actions of all others. This is the standard setting
  in which exact potential games have monotone improvement and finite
  convergence; synchronous BR can cycle even on potential games.
- **Convergence**: stop when one full pass through all satellites
  produces no change, or when ``max_rounds`` is reached. A diagnostic
  records the number of rounds used per epoch.

Best-response evaluation
------------------------
Each satellite's best response uses the same value-and-switching-cost
score as the predictive policy (Sec III), with the reciprocation
predictor replaced by the current iterate's actions: satellite ``i``
treats ``j`` as ``i``'s reciprocator iff the current iterate has
``a_j = i``. This is the multi-round generalization of the one-step
predictor: ``i`` now sees what ``j`` would *actually* do given the
current joint action, not what ``j`` would do as a first-mover.
"""

from __future__ import annotations

import numpy as np

from orbitmatch.feasibility.compute import feasible_neighbors
from orbitmatch.policy.base import NO_LINK, Policy
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# Cap on best-response rounds per epoch. Corollary 3 guarantees finite
# termination on the exact potential game; the cap is insurance against
# numerical floating-point ties that might prevent the strict-improvement
# argument from terminating in practice. Empirically the small constellation
# converges in 3-8 rounds.
DEFAULT_MAX_ROUNDS: int = 20


class EquilibriumMatching(PredictiveMatching):
    """Iterated best-response to convergence on Gamma_t.

    Inherits the full Sec III machinery from :class:`PredictiveMatching`
    (Kirchhoff value, geometric prior, per-satellite normalization,
    switching cost, varnothing-preferring tie-break, per-epoch value
    cache). The only thing that changes is :meth:`decide`: instead of
    a single BR pass, it iterates until the joint action stops moving
    or ``max_rounds`` is reached.

    Diagnostics recorded per epoch
    ------------------------------
    - ``br_rounds``: number of full Gauss-Seidel passes used.
    - ``br_converged``: 1 if a fixed point was reached, 0 if the
      max_rounds cap fired.
    """

    name: str = "equilibrium"

    def __init__(self, *args, max_rounds: int = DEFAULT_MAX_ROUNDS, **kwargs):
        super().__init__(*args, **kwargs)
        if max_rounds <= 0:
            raise ValueError(f"max_rounds must be positive; got {max_rounds}.")
        self.max_rounds = max_rounds

    # -----------------------------------------------------------------------
    # Overridden public interface
    # -----------------------------------------------------------------------

    def decide(self, t: int) -> np.ndarray:
        """Joint action at epoch t: fixed point of best-response dynamics.

        Procedure:
        1. Populate the per-epoch Kirchhoff value cache (same as in the
           predictive policy).
        2. Initialize the iterate ``a`` from the predictive policy's
           one-step BR (so this is the cheapest possible warm start).
        3. Repeat: for each satellite ``i`` in index order, replace
           ``a[i]`` with its best response to the current ``a``. After
           a full pass, check whether any entry changed. If not, the
           iterate is a fixed point; stop. Otherwise continue, up to
           ``max_rounds`` passes total.
        """
        # Populate the value cache once. _best_response_for(i, t, a) reads
        # from this cache via _value() during every BR evaluation.
        if self._cached_value_epoch != t:
            self._cached_value_matrix = self._compute_value_matrix(t)
            self._cached_value_epoch = t

        # Step 2: warm-start from the predictive policy's one-step BR.
        # Calling super().decide(t) reuses the cache populated above
        # (it checks _cached_value_epoch == t) and gives us a's initial
        # value cheaply.
        a = super().decide(t).copy()

        # Step 3: Gauss-Seidel BR sweeps.
        converged = False
        for round_idx in range(self.max_rounds):
            changed = False
            for i in range(self.n):
                br_i = self._best_response_for(i, t, a)
                if br_i != int(a[i]):
                    a[i] = br_i
                    changed = True
            if not changed:
                converged = True
                break

        rounds_used = round_idx + 1
        self._record("br_rounds", rounds_used)
        self._record("br_converged", int(converged))

        if not converged:
            log.debug(
                "EquilibriumMatching: max_rounds=%d reached at epoch %d without convergence",
                self.max_rounds,
                t,
            )

        return a

    def step(self, t: int, actions: np.ndarray) -> None:
        """Update pointing state and invalidate caches.

        Skips the predictive policy's deferral diagnostic (meaningless
        for the equilibrium policy: at a fixed point the reciprocation
        predictor by construction agrees with the realized action).
        """
        Policy.step(self, t, actions)

        # Invalidate per-epoch caches.
        self._cached_baseline_epoch = -1
        self._cached_baseline_matrix = None
        self._cached_baseline_omega = None
        self._cached_value_epoch = -1
        self._cached_value_matrix = None

    # -----------------------------------------------------------------------
    # Best-response inner loop
    # -----------------------------------------------------------------------

    def _best_response_for(self, i: int, t: int, a: np.ndarray) -> int:
        """Best response of satellite i to the joint action a.

        Uses the same value/switching-cost score as the predictive
        policy, but with the reciprocation predictor replaced by the
        current iterate's actions:

            p_ij(a) = 1 if a[j] == i else 0.

        That is, i treats j as reciprocator iff j is currently
        requesting i in the iterate ``a``.

        Tie-breaking: same as the predictive policy (varnothing wins
        all ties; otherwise lowest index among non-varnothing
        candidates, with optional random tie-break per params).
        """
        candidates = feasible_neighbors(self.feasibility, t, i)
        if len(candidates) == 0:
            return NO_LINK

        c = self.params.switching_cost_scale
        n_candidates = len(candidates)

        # First pass: per-satellite normalization constant.
        v_raw = np.zeros(n_candidates, dtype=np.float64)
        for k, j in enumerate(candidates):
            v_raw[k] = self._value(i, int(j), t)
        v_max = v_raw.max()
        if v_max <= 0.0:
            return NO_LINK

        # Second pass: scored against the current iterate.
        scores = np.zeros(n_candidates + 1, dtype=np.float64)
        for k, j in enumerate(candidates):
            v_tilde = v_raw[k] / v_max
            c_raw = self._switching_cost(i, int(j), t)
            c_tilde = c_raw / np.pi
            # Reciprocation under the current iterate.
            p = 1.0 if int(a[int(j)]) == i else 0.0
            scores[k] = p * (v_tilde - c * c_tilde)
        # scores[-1] = 0 (varnothing).

        return self._argmax_with_varnothing_preference(scores, candidates)
