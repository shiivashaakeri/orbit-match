# orbit-match/orbitmatch/policy/predictive.py
# Run: imported by other modules; not a runnable script.

"""The predictive matching policy of Section III.

Implements the policy

    a_i(t) = argmax_{j in F_i(t) cup {varnothing}}
             p_ij(t) * [V_tilde_i(j; t) - c * C_tilde_switch_ij(t)],

where V_tilde and C_tilde are the normalized value and switching cost
from Sec III.B and III.D of the paper. The four components:

- :meth:`PredictiveMatching._value` evaluates the raw V_i(j; t) as the
  marginal decrease in effective resistance (Kirchhoff index) of the
  policy's internal evaluation graph Phi(t) + epsilon * L_union_F.

- :meth:`PredictiveMatching._switching_cost` evaluates the raw angular
  switching cost in [0, pi].

- :meth:`PredictiveMatching._reciprocation_prob` evaluates p_ij in
  {0, 1} using the one-step best-response indicator.

- :meth:`PredictiveMatching.decide_for` combines them via the
  normalized decision rule.

Geometric prior
---------------
The value function is the marginal decrease in effective resistance
of Phi(t) + epsilon * L_union_F, where L_union_F is the Laplacian of
the feasibility union over the simulation horizon and epsilon > 0 is
the policy's geometric-prior weight. This regularization encodes the
common-knowledge orbital geometry (known to all satellites by
assumption) as a soft prior on the evaluation graph. The certificate
(Theorem 6) is on the realized union graph G^cup(t; T), which does
not include any epsilon term.

Per-satellite normalization
---------------------------
The raw value V_i can range widely depending on the state of Phi.
The decision rule normalizes per satellite by the max over its
candidates:

    V_tilde_i(j; t) = V_i(j; t) / max_{k in F_i(t)} V_i(k; t).

This produces V_tilde in [0, 1] regardless of the global graph state.
Per-satellite normalization preserves argmax within each satellite's
candidate set, so the policy's choice (the BR direction) is
unchanged.

Per-epoch caching
-----------------
To amortize the cost of computing Kirchhoff drops across many
candidate pairs, the value matrix V[i, j] = V_i(j; t) is computed
once per epoch (in decide(t)) and read from cache during all
per-satellite _value and _reciprocation_prob lookups. The cache is
invalidated at the end of each step().
"""

from __future__ import annotations

import numpy as np

from orbitmatch.feasibility.compute import feasibility_union, feasible_neighbors
from orbitmatch.graph.laplacian import adjacency_to_laplacian
from orbitmatch.graph.spectral import kirchhoff_index
from orbitmatch.policy.base import NO_LINK, Policy
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


class PredictiveMatching(Policy):
    """The predictive matching policy of Sec III."""

    name: str = "predictive"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pre-compute the feasibility-union Laplacian once. Used as the
        # geometric prior in the value function (per Sec III.B of the
        # paper). Encodes common-knowledge orbital geometry.
        T_max = self.feasibility.shape[0]
        union_adj = feasibility_union(self.feasibility, window_start=0, window_length=T_max)
        self._union_laplacian = adjacency_to_laplacian(union_adj.astype(np.float64))

        # Tolerance for floating-point ties in scores.
        self._tie_tol = 1e-12

        # Per-epoch caches for the Kirchhoff baseline and the value
        # matrix. Both invalidated at the end of step().
        self._cached_baseline_epoch: int = -1
        self._cached_baseline_matrix: np.ndarray | None = None
        self._cached_baseline_omega: float | None = None
        self._cached_value_epoch: int = -1
        self._cached_value_matrix: np.ndarray | None = None

        # Per-epoch cache: for each satellite j, the partner that j
        # would pick if it ignored reciprocation (its top-V choice
        # under the c-discounted score). Populated once per epoch by
        # _compute_top_value_partners and used by _reciprocation_prob
        # and the deferral diagnostic. Invalidated by step().
        self._cached_top_partner_epoch: int = -1
        self._cached_top_partners: np.ndarray | None = None

        log.info(
            "PredictiveMatching: H=%d, T=%d, c=%.3f, epsilon=%.4f, Kirchhoff baseline = %.2f",
            self.params.H,
            self.params.T,
            self.params.switching_cost_scale,
            self.params.epsilon_geometric_prior,
            kirchhoff_index(self.params.epsilon_geometric_prior * self._union_laplacian),
        )

    # -----------------------------------------------------------------------
    # Public decision interface
    # -----------------------------------------------------------------------

    def decide(self, t: int) -> np.ndarray:
        """Joint action at epoch t.

        Overridden to populate the per-epoch value-matrix cache once,
        so all per-satellite _value and _reciprocation_prob lookups
        share the work.
        """
        if self._cached_value_epoch != t:
            self._cached_value_matrix = self._compute_value_matrix(t)
            self._cached_value_epoch = t

        # Precompute every satellite's top-V partner (the choice it
        # would make ignoring reciprocation). _reciprocation_prob and
        # the deferral diagnostic read from this cache; otherwise both
        # would re-derive the same answer per call.
        if self._cached_top_partner_epoch != t:
            self._cached_top_partners = self._compute_top_value_partners(t)
            self._cached_top_partner_epoch = t

        actions = np.full(self.n, NO_LINK, dtype=np.int64)
        for i in range(self.n):
            actions[i] = self.decide_for(i, t)
        return actions

    def decide_for(self, i: int, t: int) -> int:
        """Compute satellite i's action at epoch t.

        Implements the normalized decision rule from Sec III.E:
        score_j = p_ij * [V_tilde_i(j;t) - c * C_tilde_switch_ij(t)]
        with the varnothing-preferring tie-break.
        """
        candidates = feasible_neighbors(self.feasibility, t, i)
        if len(candidates) == 0:
            return NO_LINK

        c = self.params.switching_cost_scale
        n_candidates = len(candidates)

        # First pass: compute raw values to determine the per-satellite
        # normalization constant.
        v_raw = np.zeros(n_candidates, dtype=np.float64)
        for k, j in enumerate(candidates):
            v_raw[k] = self._value(i, int(j), t)
        v_max = v_raw.max()

        # If all candidates have zero or negative value, no candidate
        # offers an improvement; the satellite refuses to link.
        if v_max <= 0.0:
            return NO_LINK

        # Second pass: form the normalized scores.
        scores = np.zeros(n_candidates + 1, dtype=np.float64)
        for k, j in enumerate(candidates):
            v_tilde = v_raw[k] / v_max
            c_raw = self._switching_cost(i, int(j), t)
            c_tilde = c_raw / np.pi
            p = self._reciprocation_prob(i, int(j), t)
            scores[k] = p * (v_tilde - c * c_tilde)
        # scores[-1] = 0 (varnothing) already.

        return self._argmax_with_varnothing_preference(scores, candidates)

    def step(self, t: int, actions: np.ndarray) -> None:
        """Update pointing state and record deferral diagnostics.

        Diagnostic runs BEFORE cache invalidation so it reuses the
        per-epoch value matrix already computed by decide(t).
        """
        # Diagnostic: deferrals are epochs where the realized action
        # differs from the top-V_tilde candidate (the choice the
        # policy would make ignoring the reciprocation factor).
        deferrals = 0
        for i in range(self.n):
            top_v = self._top_value_partner(i, t)
            if top_v == NO_LINK:
                continue
            if int(actions[i]) != top_v:
                deferrals += 1
        self._record("deferrals", deferrals)

        # Run base step to update pointing state.
        super().step(t, actions)

        # Invalidate per-epoch caches for the next epoch.
        self._cached_baseline_epoch = -1
        self._cached_baseline_matrix = None
        self._cached_baseline_omega = None
        self._cached_value_epoch = -1
        self._cached_value_matrix = None
        self._cached_top_partner_epoch = -1
        self._cached_top_partners = None

    # -----------------------------------------------------------------------
    # Components of Sec III
    # -----------------------------------------------------------------------

    def _compute_value_matrix(self, t: int) -> np.ndarray:
        """Compute V[i, j] = V_i(j; t) for all feasible (i, j) at epoch t.

        Returns an (n, n) symmetric matrix; entries for non-feasible
        pairs are 0. Uses the cached per-epoch Kirchhoff baseline so
        the baseline computation runs once per epoch, not once per
        candidate.
        """
        H = self.params.H
        T_max = self.feasibility.shape[0]
        end = min(t + H, T_max)

        # Ensure the baseline cache for epoch t is populated.
        if self._cached_baseline_epoch != t:
            phi = np.zeros((self.n, self.n)) if self.windowed_laplacian is None else self.windowed_laplacian.phi
            self._cached_baseline_matrix = phi + self.params.epsilon_geometric_prior * self._union_laplacian
            self._cached_baseline_omega = kirchhoff_index(self._cached_baseline_matrix)
            self._cached_baseline_epoch = t

        baseline_matrix = self._cached_baseline_matrix
        baseline_omega = self._cached_baseline_omega

        # Lookahead feasibility count for each pair.
        n_feasible = self.feasibility[t:end].sum(axis=0).astype(np.int64)

        V = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if n_feasible[i, j] == 0:
                    continue
                delta_L = np.zeros_like(baseline_matrix)
                delta_L[i, i] += 1.0
                delta_L[j, j] += 1.0
                delta_L[i, j] -= 1.0
                delta_L[j, i] -= 1.0
                aug_omega = kirchhoff_index(baseline_matrix + delta_L)
                v_ij = n_feasible[i, j] * (baseline_omega - aug_omega)
                V[i, j] = v_ij
                V[j, i] = v_ij  # symmetric

        return V

    def _value(self, i: int, j: int, t: int) -> float:
        """V_i(j; t): raw marginal decrease in Kirchhoff index.

        Reads from the per-epoch cache populated by decide(t). If the
        cache hasn't been populated (e.g., direct unit-test access or
        per-iteration calls in the equilibrium baseline), falls back
        to single-pair computation.
        """
        if self._cached_value_epoch == t:
            return float(self._cached_value_matrix[i, j])

        # Fallback: not in the cache. Single-pair computation.
        H = self.params.H
        T_max = self.feasibility.shape[0]
        end = min(t + H, T_max)

        n_feasible = int(self.feasibility[t:end, i, j].sum())
        if n_feasible == 0:
            return 0.0

        if self._cached_baseline_epoch != t:
            phi = np.zeros((self.n, self.n)) if self.windowed_laplacian is None else self.windowed_laplacian.phi
            self._cached_baseline_matrix = phi + self.params.epsilon_geometric_prior * self._union_laplacian
            self._cached_baseline_omega = kirchhoff_index(self._cached_baseline_matrix)
            self._cached_baseline_epoch = t

        baseline_matrix = self._cached_baseline_matrix
        baseline_omega = self._cached_baseline_omega

        delta_L = np.zeros_like(baseline_matrix)
        delta_L[i, i] += 1.0
        delta_L[j, j] += 1.0
        delta_L[i, j] -= 1.0
        delta_L[j, i] -= 1.0
        augmented_omega = kirchhoff_index(baseline_matrix + delta_L)

        return n_feasible * (baseline_omega - augmented_omega)

    def _switching_cost(self, i: int, j: int, t: int) -> float:
        """C_switch_ij(t): raw angular slew in [0, pi].

        Returns 0 if satellite i has no current pointing direction
        (NaN), per the convention that first-link slew is free.

        Normalization by pi is applied in the caller (decide_for /
        _reciprocation_prob) to produce C_tilde in [0, 1].
        """
        current = self.pointing.directions[i]
        if np.any(np.isnan(current)):
            return 0.0

        delta = self.positions[t, j] - self.positions[t, i]
        norm = np.linalg.norm(delta)
        if norm == 0:
            return 0.0
        target = delta / norm

        # Angle via dot product, clamped for numerical safety.
        dot = float(np.clip(current @ target, -1.0, 1.0))
        return float(np.arccos(dot))

    def _reciprocation_prob(self, i: int, j: int, t: int) -> float:
        """p_ij(t): one-step best-response indicator.

        Reads from the per-epoch top-value-partner cache populated by
        decide(t). Returns 1.0 iff j's top-V choice (the choice j
        would make ignoring reciprocation) is i. Falls back to a full
        computation if the cache is not populated (e.g. unit tests
        calling this method directly).
        """
        if self._cached_top_partner_epoch == t and self._cached_top_partners is not None:
            return 1.0 if int(self._cached_top_partners[j]) == i else 0.0

        # Fallback path: no cache; recompute j's top-V choice in full.
        return 1.0 if self._top_value_partner_uncached(j, t) == i else 0.0

    def _top_value_partner(self, i: int, t: int) -> int:
        """Diagnostic helper: i's pick if it ignored p_ij.

        Reads from the per-epoch cache populated by decide(t). Falls
        back to a full computation if the cache is missing.
        """
        if self._cached_top_partner_epoch == t and self._cached_top_partners is not None:
            return int(self._cached_top_partners[i])
        return self._top_value_partner_uncached(i, t)

    def _compute_top_value_partners(self, t: int) -> np.ndarray:
        """Compute every satellite's top-V partner for epoch t in one pass.

        Returns an (n,) int64 array where entry i is satellite i's
        top-V choice (the partner i would request if it ignored p_ij),
        or :data:`NO_LINK` (-1) if i has no positive-value candidate.

        Reads from the per-epoch value-matrix cache, so the only
        per-candidate work is the switching-cost geometry. Same logic
        as :meth:`_top_value_partner_uncached`, but writes results for
        all i in one go and avoids the redundant per-i fallback path.
        """
        partners = np.full(self.n, NO_LINK, dtype=np.int64)
        c = self.params.switching_cost_scale

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
                scores[k] = v_tilde - c * c_tilde

            partners[i] = self._argmax_with_varnothing_preference(scores, candidates)

        return partners

    def _top_value_partner_uncached(self, i: int, t: int) -> int:
        """Compute satellite i's top-V partner from scratch (no cache).

        Used as a fallback when the per-epoch cache has not been
        populated (e.g. unit tests calling _reciprocation_prob or
        _top_value_partner directly).
        """
        candidates = feasible_neighbors(self.feasibility, t, i)
        if len(candidates) == 0:
            return NO_LINK

        c = self.params.switching_cost_scale
        n_cands = len(candidates)

        v_raw = np.zeros(n_cands, dtype=np.float64)
        for k, j in enumerate(candidates):
            v_raw[k] = self._value(i, int(j), t)
        v_max = v_raw.max()
        if v_max <= 0.0:
            return NO_LINK

        scores = np.zeros(n_cands + 1, dtype=np.float64)
        for k, j in enumerate(candidates):
            v_tilde = v_raw[k] / v_max
            c_raw = self._switching_cost(i, int(j), t)
            c_tilde = c_raw / np.pi
            scores[k] = v_tilde - c * c_tilde

        return self._argmax_with_varnothing_preference(scores, candidates)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _argmax_with_varnothing_preference(
        self,
        scores: np.ndarray,
        candidates: np.ndarray,
    ) -> int:
        """Argmax over candidate scores, with varnothing winning ties.

        The varnothing slot is at index ``len(candidates)`` in the
        scores array (last entry). When the varnothing slot's score is
        tied (within tolerance) with the max, varnothing wins. Among
        non-varnothing tied scores, the lowest-index candidate wins.
        """
        n_cands = len(candidates)
        max_val = scores.max()
        tied = np.flatnonzero(scores >= max_val - self._tie_tol)

        # If varnothing is tied for first, take it.
        if n_cands in tied:
            return NO_LINK

        # Otherwise: deterministic or random tie-break per params.
        choice = int(self.rng.choice(tied)) if self.params.tie_break == "random" and len(tied) > 1 else int(tied[0])

        return int(candidates[choice])
