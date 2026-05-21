# orbit-match/orbitmatch/policy/baselines.py
# Run: imported by other modules; not a runnable script.

"""Baseline policies for the paper's experiments.

Two baselines are implemented here:

- :class:`GreedyMatching` -- the predictive policy with the
  reciprocation factor disabled. Tests whether the deferral mechanism
  (multiplying value by reciprocation probability) is doing real work,
  by comparing against a policy that does everything else the same but
  always requests its highest-value candidate.

- :class:`RandomMatching` -- uniform over feasible neighbors. Lower
  bound reference. With the mutual-choice rule, the realized matching
  is approximately a random matching of the feasibility graph; this is
  the "no strategy at all" baseline.

The :class:`EquilibriumMatching` baseline (full best-response dynamics
to convergence) lives in :mod:`orbitmatch.policy.equilibrium` because
it has substantially more structure.

Design notes
------------
Greedy inherits from :class:`PredictiveMatching` and overrides only
``_reciprocation_prob`` to return 1.0. This way the value-matrix cache,
the per-satellite normalization, the switching-cost machinery, and the
varnothing-preferring tie-break are all shared verbatim. The only
difference is that the score becomes

    score_j = V_tilde_i(j; t) - c * C_tilde_switch_ij(t),

with no reciprocation multiplier. The varnothing slot still wins ties
(its score is 0), so greedy refuses to link only when no candidate
offers positive normalized value -- matching the predictive policy's
refusal convention.

Random subclasses :class:`Policy` directly and uses ``self.rng`` for
its draws. It picks varnothing only when the feasibility set is empty
at the current epoch, so it always tries to form a link when it can
(matching the design discussion in the chat: removes "deferral" as a
confounding variable, isolating the picking question).
"""

from __future__ import annotations

from orbitmatch.feasibility.compute import feasible_neighbors
from orbitmatch.policy.base import NO_LINK, Policy
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------


class GreedyMatching(PredictiveMatching):
    """Predictive matching with the reciprocation factor disabled.

    Each satellite requests its highest-scoring feasible neighbor under

        score_j = V_tilde_i(j; t) - c * C_tilde_switch_ij(t),

    ignoring whether the candidate would reciprocate. The mutual-choice
    rule still applies at the realization step (so requests that don't
    match still don't form edges), but the policy does not anticipate
    that.

    All other machinery is inherited from :class:`PredictiveMatching`:
    the Kirchhoff value function with the geometric prior, the
    per-satellite normalization, the switching cost, the per-epoch
    value-matrix cache, and the varnothing-preferring tie-break.
    """

    name: str = "greedy"

    def _reciprocation_prob(self, i: int, j: int, t: int) -> float:  # noqa: ARG002
        """Always 1: greedy does not predict reciprocation."""
        return 1.0


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


class RandomMatching(Policy):
    """Uniform-random matching baseline.

    Each satellite picks one of its feasible neighbors uniformly at
    random. If the feasibility set is empty at the current epoch, it
    plays varnothing (no link). Otherwise it always picks a partner,
    so the realized matching depends only on which random picks happen
    to be mutual under the mutual-choice rule.

    This is the "no strategy" lower-bound reference. The empirical gap
    between Random and Predictive isolates the contribution of the
    value-plus-reciprocation decision rule; the gap between Greedy and
    Random isolates the contribution of the value function alone.
    """

    name: str = "random"

    def decide_for(self, i: int, t: int) -> int:
        candidates = feasible_neighbors(self.feasibility, t, i)
        if len(candidates) == 0:
            return NO_LINK
        return int(self.rng.choice(candidates))
