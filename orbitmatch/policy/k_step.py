# orbit-match/orbitmatch/policy/k_step.py
# Run: imported by other modules; not a runnable script.

"""k-step predictive matching: an explicit recursion-depth family.

The paper of Section IV.D notes that the predictive policy with the
one-step reciprocation predictor is one round of best-response (BR)
dynamics on the potential game Gamma_t, and that "richer reciprocation
predictors correspond to deeper truncations of best-response dynamics."
This module implements that family explicitly: KStepPredictive runs
exactly k rounds of BR, starting from a level-0 initialization (every
satellite picks its top-V partner, ignoring reciprocation).

Two update orders are supported via the ``mode`` kwarg:

  mode = "sync"          synchronous (Jacobi). All satellites update
                         simultaneously against the previous round's
                         actions. Cleanest level-k semantics: when i
                         models j at level k-1, j models i at level k-1
                         too, with no canonical ordering.
  mode = "gauss_seidel"  Gauss-Seidel. Satellites update sequentially
                         in index order within each round; each
                         satellite sees the most recent actions of
                         lower-indexed satellites. The standard setting
                         for finite-improvement convergence on
                         potential games.

The two modes converge to *different* fixed points on the same game.
Synchronous BR has a tendency to land on shallow attractors close to
the level-0 profile; Gauss-Seidel walks more aggressively into the
potential's interior and tends to find higher-W_t fixed points.

Boundary cases:

  k = 0     identical to GreedyMatching regardless of mode
            (every satellite picks top-V; no BR).
  k = 1     identical to PredictiveMatching regardless of mode
            (one BR sweep against the level-0 profile: in sync mode
            this is a Jacobi sweep, in gauss_seidel mode the
            sequential updates of round 1 only see the level-0
            profile for unupdated satellites; in both cases the
            output is the predictive policy's action when the
            level-0 cache feeds _reciprocation_prob).
  k -> inf  sync converges to a synchronous fixed point;
            gauss_seidel converges to a NE of Gamma_t.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from orbitmatch.feasibility.compute import feasible_neighbors
from orbitmatch.policy.base import NO_LINK, Policy
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


class KStepPredictive(PredictiveMatching):
    """Predictive matching with exactly k rounds of BR.

    Parameters
    ----------
    k
        Number of BR rounds (>= 0).
    mode
        Either "sync" (synchronous Jacobi) or "gauss_seidel"
        (sequential). See module docstring for the difference.

    Diagnostics recorded per epoch
    ------------------------------
    - ``k_step_value``: the fixed k used (constant; useful when
      comparing across k in sweeps).
    """

    name: str = "k_step"

    def __init__(
        self,
        *args,
        k: int = 1,
        mode: Literal["sync", "gauss_seidel"] = "sync",
        update_order: np.ndarray | None = None,
        order_by: Literal["identity", "feasibility"] = "identity",
        temporal_warmstart: bool = False,
        partial_observation: bool = False,
        update_pointing_on_request: bool = False,
        dynamic_prior: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if k < 0:
            raise ValueError(f"k must be non-negative; got {k}.")
        if mode not in ("sync", "gauss_seidel"):
            raise ValueError(f"mode must be 'sync' or 'gauss_seidel'; got {mode!r}.")
        if order_by not in ("identity", "feasibility"):
            raise ValueError(f"order_by must be 'identity' or 'feasibility'; got {order_by!r}.")
        if update_order is not None and mode != "gauss_seidel":
            raise ValueError(
                "update_order is only meaningful in gauss_seidel mode; "
                "synchronous mode is order-invariant."
            )
        if update_order is not None and order_by != "identity":
            raise ValueError(
                "update_order and order_by are mutually exclusive: "
                "specify either an explicit ordering or a sorting rule, not both."
            )
        if update_order is not None:
            update_order = np.asarray(update_order, dtype=np.int64)
            if update_order.shape != (self.n,):
                raise ValueError(f"update_order must have shape ({self.n},); got {update_order.shape}")
            if set(update_order.tolist()) != set(range(self.n)):
                raise ValueError("update_order must be a permutation of {0, ..., n-1}")
        self.k = k
        self.mode = mode
        self.update_order = update_order
        self.order_by = order_by
        self.temporal_warmstart = temporal_warmstart
        self.partial_observation = partial_observation
        self.update_pointing_on_request = update_pointing_on_request
        self.dynamic_prior = dynamic_prior

        # State across epochs (for temporal warm-start, dynamic prior).
        self._last_realized_actions: np.ndarray | None = None
        self._last_realized_adjacency: np.ndarray | None = None  # for dynamic prior

        # Successor index of each satellite in the current sweep, populated
        # at the top of decide() when GS+partial_observation is active.
        # Maps satellite i to its position in the update order so that
        # _best_response_against can tell predecessors from successors.
        self._current_order_index: np.ndarray | None = None

    def decide(self, t: int) -> np.ndarray:
        """Run exactly self.k rounds of BR at epoch t."""
        # Populate value-matrix and top-partner caches once.
        if self._cached_value_epoch != t:
            self._cached_value_matrix = self._compute_value_matrix(t)
            self._cached_value_epoch = t
        if self._cached_top_partner_epoch != t:
            self._cached_top_partners = self._compute_top_value_partners(t)
            self._cached_top_partner_epoch = t

        # Choose the iteration starting point.
        if self.temporal_warmstart and self._last_realized_actions is not None:
            a = self._build_warmstart_profile(t)
        else:
            # Level-0 profile: every satellite picks its top-V partner.
            a = self._cached_top_partners.copy()

        # Choose the iteration order for Gauss-Seidel. Synchronous is
        # order-invariant; ignore order_by in that case.
        if self.mode == "gauss_seidel":
            if self.update_order is not None:
                order = self.update_order
            elif self.order_by == "feasibility":
                order = self._feasibility_order(t)
            else:
                order = np.arange(self.n)
        else:
            order = np.arange(self.n)

        # For partial-observation BR, we need to know each satellite's
        # position in the update order at evaluation time, so the BR step
        # can distinguish committed predecessors from uncommitted successors.
        # Populate a satellite -> order-position map. In sync mode this is
        # unused.
        self._current_order_index = np.empty(self.n, dtype=np.int64)
        for pos, sat_idx in enumerate(order):
            self._current_order_index[int(sat_idx)] = pos

        if self.mode == "sync":
            for _ in range(self.k):
                a_new = np.full(self.n, NO_LINK, dtype=np.int64)
                for i in range(self.n):
                    a_new[i] = self._best_response_against(i, t, a)
                a = a_new
        else:  # gauss_seidel
            for _ in range(self.k):
                for i in order:
                    a[int(i)] = self._best_response_against(int(i), t, a)

        self._record("k_step_value", self.k)
        return a

    def step(self, t: int, actions: np.ndarray) -> None:
        """Update pointing, invalidate per-epoch caches, save realized actions
        for temporal warm-start."""
        # Pointing update path:
        #   default (Policy.step) updates pointing only on mutual matches.
        #   With update_pointing_on_request=True, update pointing on every
        #   non-NO_LINK request, so the next epoch's switching cost reflects
        #   wherever the laser actually slewed (even on failed requests).
        if self.update_pointing_on_request:
            self._update_pointing_on_request(t, actions)
        else:
            Policy.step(self, t, actions)

        # Build the realized-actions profile: a_i is set iff the link (i, j)
        # mutually formed. Used by temporal_warmstart and dynamic_prior.
        realized = np.full(self.n, NO_LINK, dtype=np.int64)
        for i in range(self.n):
            j = int(actions[i])
            if j == NO_LINK or not (0 <= j < self.n):
                continue
            if int(actions[j]) == i:
                realized[i] = j
        self._last_realized_actions = realized

        # Build the realized adjacency matrix (symmetric) for the dynamic prior.
        if self.dynamic_prior:
            adj = np.zeros((self.n, self.n), dtype=np.float64)
            for i in range(self.n):
                j = int(realized[i])
                if j != NO_LINK:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
            self._last_realized_adjacency = adj

        # Invalidate per-epoch caches.
        self._cached_baseline_epoch = -1
        self._cached_baseline_matrix = None
        self._cached_baseline_omega = None
        self._cached_value_epoch = -1
        self._cached_value_matrix = None
        self._cached_top_partner_epoch = -1
        self._cached_top_partners = None
        self._current_order_index = None

    def _update_pointing_on_request(self, t: int, actions: np.ndarray) -> None:
        """Update each satellite's pointing direction toward its requested
        partner, regardless of whether the link formed.

        This is idea 3: when a satellite slews its laser toward a candidate
        $j$ during the epoch, the slew has physically happened by the time
        the epoch ends, regardless of $j$'s response. The pointing tracker
        should reflect that.

        Satellites that played NO_LINK (varnothing) keep their previous
        pointing direction.
        """
        positions_t = self.positions[t]
        for i in range(self.n):
            j = int(actions[i])
            if j == NO_LINK or not (0 <= j < self.n):
                continue
            delta = positions_t[j] - positions_t[i]
            norm = float(np.linalg.norm(delta))
            if norm > 0:
                self.pointing.directions[i] = delta / norm
                # partners[i] reflects the *intended* partner, not necessarily
                # the realized one. Could be confusing for downstream code;
                # we set it conservatively only if the link formed.
                if int(actions[j]) == i:
                    self.pointing.partners[i] = j

    def _compute_value_matrix(self, t: int) -> np.ndarray:
        """Compute the per-epoch value matrix, optionally with a dynamic prior.

        When dynamic_prior is enabled and we have a previous realized
        adjacency, the geometric prior used in the Kirchhoff baseline is
        re-weighted: edges that were realized in the previous epoch get
        full weight, edges that were not realized get downweighted by a
        small factor. This corresponds to idea 4: the prior reflects the
        structure that the policy actually uses, not the static feasibility
        union.

        Falls back to the parent's value matrix when dynamic_prior is off
        or no previous epoch is available.
        """
        if not self.dynamic_prior or self._last_realized_adjacency is None:
            return super()._compute_value_matrix(t)

        # Build a weighted prior. Edges in the previous realized adjacency
        # get weight 1.0; other feasibility-union edges get weight 0.25.
        # The weights are arbitrary but small enough that the prior remains
        # a soft regularization, not a hard constraint.
        from orbitmatch.feasibility.compute import feasibility_union  # noqa: PLC0415
        from orbitmatch.graph.laplacian import adjacency_to_laplacian  # noqa: PLC0415

        T_max = self.feasibility.shape[0]
        full_adj = feasibility_union(self.feasibility, window_start=0, window_length=T_max).astype(np.float64)
        # Weighted: 1.0 where realized last epoch, 0.25 elsewhere.
        prev_adj = self._last_realized_adjacency
        weighted_adj = np.where(prev_adj > 0, full_adj, 0.25 * full_adj)
        weighted_prior = adjacency_to_laplacian(weighted_adj)

        # Temporarily swap, call parent, swap back. The parent reads
        # self._union_laplacian at line 231 of predictive.py.
        original_prior = self._union_laplacian
        self._union_laplacian = weighted_prior
        try:
            result = super()._compute_value_matrix(t)
        finally:
            self._union_laplacian = original_prior
        return result
        """Return a permutation sorted by ascending feasibility count.

        Ties broken by satellite index for determinism. Satellites with
        fewer feasible neighbors decide first.
        """
        feasibility_counts = self.feasibility[t].sum(axis=1).astype(np.int64)
        # np.argsort is stable, so equal counts preserve index order.
        return np.argsort(feasibility_counts, kind="stable").astype(np.int64)

    def _feasibility_order(self, t: int) -> np.ndarray:
        """Return a permutation sorted by ascending feasibility count.

        Ties broken by satellite index for determinism. Satellites with
        fewer feasible neighbors decide first.
        """
        feasibility_counts = self.feasibility[t].sum(axis=1).astype(np.int64)
        # np.argsort is stable, so equal counts preserve index order.
        return np.argsort(feasibility_counts, kind="stable").astype(np.int64)

    def _build_warmstart_profile(self, t: int) -> np.ndarray:
        """Initialize action profile from the last realized matching.

        Entries that are NO_LINK in the previous matching, or that are now
        infeasible, fall back to the top-V partner from the cached level-0
        profile.
        """
        assert self._last_realized_actions is not None
        assert self._cached_top_partners is not None
        a = self._last_realized_actions.copy()
        for i in range(self.n):
            j = int(a[i])
            # Fall back to top-V if no previous link or the previous partner
            # is no longer feasible.
            if j == NO_LINK or not bool(self.feasibility[t, i, j]):
                a[i] = int(self._cached_top_partners[i])
        return a

    def _best_response_against(self, i: int, t: int, a: np.ndarray) -> int:
        """Best response of satellite i to the joint action a.

        Same score form as the predictive policy. The reciprocation factor
        p_ij is read from the iterate a:
            p_ij(a) = 1 iff a[j] == i, else 0.

        If partial_observation is enabled and we are in Gauss-Seidel mode,
        the reciprocation factor for satellites that have not yet committed
        in this sweep (successors of i in the update order) is replaced by
        the one-step BR predictor (top-V partner cache). Predecessors keep
        their committed action.

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

        # Decide if partial observation is active for this evaluation.
        use_partial = (
            self.partial_observation
            and self.mode == "gauss_seidel"
            and self._current_order_index is not None
            and self._cached_top_partners is not None
        )
        if use_partial:
            i_pos = int(self._current_order_index[i])

        scores = np.zeros(n_candidates + 1, dtype=np.float64)
        for kk, j in enumerate(candidates):
            v_tilde = v_raw[kk] / v_max
            c_raw = self._switching_cost(i, int(j), t)
            c_tilde = c_raw / np.pi

            if use_partial:
                # If j is a predecessor (has committed earlier in this
                # sweep), use the observed action. If j is a successor (or
                # i itself), use the one-step BR predictor.
                j_pos = int(self._current_order_index[int(j)])
                if j_pos < i_pos:
                    # Predecessor: read committed action.
                    p = 1.0 if int(a[int(j)]) == i else 0.0
                else:
                    # Successor: use one-step BR predictor.
                    p = 1.0 if int(self._cached_top_partners[int(j)]) == i else 0.0
            else:
                p = 1.0 if int(a[int(j)]) == i else 0.0

            scores[kk] = p * (v_tilde - c * c_tilde)
        # scores[-1] = 0 (varnothing).

        return self._argmax_with_varnothing_preference(scores, candidates)
