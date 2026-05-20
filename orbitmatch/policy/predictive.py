# orbit-match/orbitmatch/policy/predictive.py
# Run: imported by other modules; not a runnable script.

"""The predictive matching policy of Section III.

Implements the policy

    a_i(t) = argmax_{j in F_i(t) cup {varnothing}}
             p_ij(t) * [V_tilde_i(j; t) - c * C_tilde_switch_ij(t)],

where V_tilde and C_tilde are the normalized value and switching cost
from Sec III.B and III.D of the paper. The four components:

- :meth:`PredictiveMatching._value` evaluates the raw V_i(j; t) in
  [0, 2H] as the marginal contribution of edge (i, j) to lambda_2(Phi)
  scaled by the lookahead horizon.

- :meth:`PredictiveMatching._switching_cost` evaluates the raw angular
  switching cost in [0, pi].

- :meth:`PredictiveMatching._reciprocation_prob` evaluates p_ij in
  {0, 1} using the one-step best-response indicator. The simulated
  best response uses the same normalized form as the real decision.

- :meth:`PredictiveMatching.decide_for` combines them via the
  normalized decision rule.

The predictive matching policy of Section III.

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
the policy's geometric-prior weight. This regularization is *not* a
cold-start hack: it encodes the common-knowledge orbital geometry
(known to all satellites by assumption) as a soft prior on the
evaluation graph. The certificate (Theorem 6) is on the realized
union graph G^cup(t; T), which does not include any epsilon term.

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

The certificate (Sec IV.E) is on the realized union graph
G^cup(t; T), not on Phi, so this cold-start device does not enter the
guarantee.
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
        # paper). Not a cold-start hack; it's the policy's encoding of
        # common-knowledge orbital geometry.
        T_max = self.feasibility.shape[0]
        union_adj = feasibility_union(self.feasibility, window_start=0, window_length=T_max)
        self._union_laplacian = adjacency_to_laplacian(union_adj.astype(np.float64))

        # Tolerance for floating-point ties in scores.
        self._tie_tol = 1e-12

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
            c_raw = self._switching_cost(i, j, t)
            c_tilde = c_raw / np.pi
            p = self._reciprocation_prob(i, j, t)
            scores[k] = p * (v_tilde - c * c_tilde)
        # scores[-1] = 0 (varnothing) already.

        return self._argmax_with_varnothing_preference(scores, candidates)

    # -----------------------------------------------------------------------
    # Components of Sec III
    # -----------------------------------------------------------------------

    def _value(self, i: int, j: int, t: int) -> float:
        """V_i(j; t): raw marginal decrease in Kirchhoff index.

        Computed as

            V_i(j; t) = n_feasible * [Omega(baseline) - Omega(baseline + L_ij)]

        where baseline = Phi(t) + epsilon * L_union_F is the policy's
        internal evaluation graph, and n_feasible is the number of
        epochs in [t, t+H) where (i, j) remains feasible.

        Decrease in Kirchhoff index = improvement in graph connectivity.
        Larger return value means the edge contributes more to
        connectivity over the lookahead.

        The baseline is always connected (because epsilon * L_union_F
        is connected by Assumption 5), so the Kirchhoff index is
        finite and the gradient is well-defined.
        """
        H = self.params.H
        T_max = self.feasibility.shape[0]
        end = min(t + H, T_max)

        n_feasible = int(self.feasibility[t:end, i, j].sum())
        if n_feasible == 0:
            return 0.0

        # Evaluation graph: realized history Phi plus geometric prior.
        phi = np.zeros((self.n, self.n)) if self.windowed_laplacian is None else self.windowed_laplacian.phi
        baseline_matrix = phi + self.params.epsilon_geometric_prior * self._union_laplacian

        baseline_omega = kirchhoff_index(baseline_matrix)

        # Augment with the candidate edge.
        delta_L = np.zeros_like(baseline_matrix)
        delta_L[i, i] += 1.0
        delta_L[j, j] += 1.0
        delta_L[i, j] -= 1.0
        delta_L[j, i] -= 1.0
        augmented_omega = kirchhoff_index(baseline_matrix + delta_L)

        # Drop in Kirchhoff index, scaled by lookahead feasibility.
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

        Simulates j's decision under the same per-satellite normalized
        rule. Returns 1.0 if i is j's choice, else 0.0.
        """
        j_candidates = feasible_neighbors(self.feasibility, t, j)
        if len(j_candidates) == 0:
            return 0.0
        if i not in j_candidates:
            return 0.0

        c = self.params.switching_cost_scale
        n_j_cands = len(j_candidates)

        # First pass: j's raw values.
        v_raw = np.zeros(n_j_cands, dtype=np.float64)
        for k, partner in enumerate(j_candidates):
            v_raw[k] = self._value(j, int(partner), t)
        v_max = v_raw.max()

        if v_max <= 0.0:
            return 0.0  # j has no positive-value candidate, picks varnothing

        # Second pass: j's normalized scores (without recursing on p).
        j_scores = np.zeros(n_j_cands + 1, dtype=np.float64)
        for k, partner in enumerate(j_candidates):
            v_tilde = v_raw[k] / v_max
            c_raw = self._switching_cost(j, partner, t)
            c_tilde = c_raw / np.pi
            j_scores[k] = v_tilde - c * c_tilde

        j_choice = self._argmax_with_varnothing_preference(j_scores, j_candidates)
        return 1.0 if j_choice == i else 0.0

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

    def step(self, t: int, actions: np.ndarray) -> None:
        """Update pointing state and record deferral diagnostics."""
        super().step(t, actions)

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

    def _top_value_partner(self, i: int, t: int) -> int:
        """Diagnostic helper: i's pick if it ignored p_ij."""
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
            c_raw = self._switching_cost(i, j, t)
            c_tilde = c_raw / np.pi
            scores[k] = v_tilde - c * c_tilde

        return self._argmax_with_varnothing_preference(scores, candidates)
