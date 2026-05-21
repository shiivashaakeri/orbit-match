# orbit-match/orbitmatch/policy/base.py
# Run: imported by other modules; not a runnable script.

"""Abstract policy interface and shared scaffolding.

A policy maps each satellite's information state to an action at every
epoch. In this codebase, a Policy object is constructed once per
simulation with the full feasibility tensor and pointing-cost machinery,
and is then driven by the simulation loop:

    policy = SomePolicy(n, feasibility, positions, dt_s, params)
    for t in range(n_epochs):
        actions = policy.decide(t)
        matching = actions_to_edges(actions)
        # ... apply matching, push to windowed laplacian ...
        policy.step(t, actions)

The :class:`Policy` base provides:

- The shared parameter dataclass :class:`PolicyParams`.
- A standard ``__init__`` that consumes
  ``(n, feasibility, positions, dt_s, params, rng, windowed_laplacian)``.
- A pointing-state tracker (:class:`PointingState`).
- A default ``decide(t)`` that loops over satellites calling
  ``decide_for(i, t)``. Subclasses can override ``decide(t)`` to
  vectorize.
- A ``step(t, actions)`` hook that updates pointing state.

Scaling conventions
-------------------
The decision rule in :mod:`orbitmatch.policy.predictive` uses normalized
quantities:

- Value function ``V_i(j; t)`` is computed as the marginal contribution
  of edge (i, j) to the negative effective resistance (Kirchhoff index)
  of the policy's internal evaluation graph
  ``Phi(t) + epsilon * L_union_F``. Returns a non-negative raw value
  whose magnitude depends on the graph state. The decision rule
  normalizes by the per-satellite max over its candidates to produce
  ``V_tilde`` in [0, 1] for combining with the switching cost. The
  per-satellite normalization preserves argmax within each satellite's
  candidate set.

This corresponds to the paper's Sec III.B-III.D scaling. Raw values are
preserved for diagnostics and for the Sec IV potential-game analysis.

Concrete policies live in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from orbitmatch.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from orbitmatch.graph.windowed import WindowedLaplacian

log = get_logger(__name__)


# The action representing varnothing ("no link requested"). Conforms to
# the convention in graph.laplacian.edges_to_actions / actions_to_edges.
NO_LINK: int = -1


@dataclass(frozen=True)
class PolicyParams:
    """Tunable parameters for a policy.

    Parameters
    ----------
    H
        Lookahead horizon in epochs. Used by the predictive policy's
        value function to integrate connectivity contribution.
    T
        Certificate window length in epochs. Carried by the policy for
        bookkeeping; the actual windowed Laplacian lives in the
        simulation runner.
    switching_cost_scale
        The constant ``c`` in the decision rule
        ``p_ij * [V_tilde - c * C_tilde]``. Must lie in [0, 1]; it is
        the maximum fraction of normalized value that a full slew can
        offset. Default 0.2.
    epsilon_geometric_prior
        Weight on the feasibility-union Laplacian in the policy's
        internal evaluation graph: V_i is computed on
        Phi(t) + epsilon * L_union_F. This encodes common-knowledge
        orbital geometry as a soft prior on connectivity, ensuring
        the value function is well-defined even when Phi is
        disconnected (as it is at startup). Default 0.01.
    tie_break
        How to break ties in the argmax over candidate partners.
        ``"lowest_index"`` is deterministic; ``"random"`` uses the
        policy's RNG.
    seed
        Random seed for any policy that needs randomness. Optional.
    """

    H: int = 10
    T: int = 30
    switching_cost_scale: float = 0.2
    epsilon_geometric_prior: float = 0.01
    tie_break: str = "lowest_index"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.H <= 0:
            raise ValueError(f"H must be positive; got {self.H}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive; got {self.T}.")
        if not (0.0 <= self.switching_cost_scale <= 1.0):
            raise ValueError(f"switching_cost_scale must be in [0, 1]; got {self.switching_cost_scale}.")
        if self.epsilon_geometric_prior <= 0.0:
            raise ValueError(f"epsilon_geometric_prior must be positive; got {self.epsilon_geometric_prior}.")
        if self.tie_break not in {"lowest_index", "random"}:
            raise ValueError(f"tie_break must be 'lowest_index' or 'random'; got {self.tie_break!r}.")


# ---------------------------------------------------------------------------
# Pointing-state tracker
# ---------------------------------------------------------------------------


@dataclass
class PointingState:
    """Tracks each satellite's current pointing direction.

    The "current pointing direction" is the unit bearing to the partner
    the satellite is currently pointed at, inherited from the previous
    epoch. If a satellite has no active link, its pointing direction is
    :data:`np.nan`, and the switching cost treats it as "free" to point
    at any new partner.

    Parameters
    ----------
    n
        Number of satellites.
    """

    n: int
    directions: np.ndarray = field(init=False)
    partners: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.directions = np.full((self.n, 3), np.nan, dtype=np.float64)
        self.partners = np.full(self.n, NO_LINK, dtype=np.int64)

    def update_from_realized(
        self,
        actions: np.ndarray,
        positions_t: np.ndarray,
    ) -> None:
        """Update pointing state after a realized joint action.

        For each satellite ``i``:

        - If ``actions[i] == j`` and ``actions[j] == i``, the link
          formed. Set ``directions[i]`` to the unit bearing toward
          ``j``'s position at this epoch.
        - Otherwise, keep the previous direction (NaN if never linked).

        Parameters
        ----------
        actions
            ``(n,)`` int array; ``-1`` means no request.
        positions_t
            ``(n, 3)`` ECI positions at this epoch (km).
        """
        for i in range(self.n):
            j = int(actions[i])
            if j == NO_LINK:
                continue
            if 0 <= j < self.n and int(actions[j]) == i:
                delta = positions_t[j] - positions_t[i]
                norm = np.linalg.norm(delta)
                if norm > 0:
                    self.directions[i] = delta / norm
                    self.partners[i] = j


# ---------------------------------------------------------------------------
# Abstract policy
# ---------------------------------------------------------------------------


class Policy(ABC):
    """Abstract base for a policy applied to a constellation.

    Subclasses implement :meth:`decide_for` (per-satellite decision);
    the default :meth:`decide` loops over satellites. Subclasses can
    override :meth:`decide` if a vectorized form is faster.

    Parameters
    ----------
    n
        Number of satellites.
    feasibility
        Boolean ``(n_epochs, n, n)`` feasibility tensor.
    positions
        ECI positions ``(n_epochs, n, 3)``. Used for switching-cost
        geometry.
    dt_s
        Epoch length in seconds.
    params
        :class:`PolicyParams` instance.
    rng
        Numpy ``Generator``; used by random-tie-break policies and the
        Random baseline. May be ``None`` for deterministic policies.
    windowed_laplacian
        Optional reference to the simulation's :class:`WindowedLaplacian`.
        The predictive policy uses it as the baseline for its value
        function. If ``None``, the policy falls back to an empty-graph
        baseline.
    """

    name: str = "abstract"

    def __init__(
        self,
        n: int,
        feasibility: np.ndarray,
        positions: np.ndarray,
        dt_s: float,
        params: PolicyParams,
        rng: Optional[np.random.Generator] = None,
        windowed_laplacian: Optional["WindowedLaplacian"] = None,
    ) -> None:
        if feasibility.shape[1] != n or feasibility.shape[2] != n:
            raise ValueError(f"feasibility shape {feasibility.shape} inconsistent with n={n}.")
        if positions.shape[1] != n:
            raise ValueError(f"positions shape {positions.shape} inconsistent with n={n}.")

        self.n = n
        self.feasibility = feasibility
        self.positions = positions
        self.dt_s = dt_s
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng(params.seed)
        self.pointing = PointingState(n)
        self.windowed_laplacian = windowed_laplacian
        self._diagnostics: dict[str, list] = {}

    # ---- Decision interface ------------------------------------------------

    @abstractmethod
    def decide_for(self, i: int, t: int) -> int:
        """Return satellite ``i``'s requested partner at epoch ``t``.

        Returns ``-1`` (:data:`NO_LINK`) to indicate no request.
        """

    def decide(self, t: int) -> np.ndarray:
        """Joint action at epoch ``t``: ``(n,)`` int array.

        Default implementation loops over satellites. Override for
        vectorized policies.
        """
        actions = np.full(self.n, NO_LINK, dtype=np.int64)
        for i in range(self.n):
            actions[i] = self.decide_for(i, t)
        return actions

    def step(self, t: int, actions: np.ndarray) -> None:
        """Hook called after the joint action is realized.

        Default behavior: update pointing state from the realized
        actions (which embed the mutual-choice rule).
        """
        self.pointing.update_from_realized(actions, self.positions[t])

    # ---- Diagnostics -------------------------------------------------------

    def diagnostics(self) -> dict[str, list]:
        """Return a copy of the diagnostics dict.

        Each policy records its own per-epoch counters here.
        """
        return {k: list(v) for k, v in self._diagnostics.items()}

    def _record(self, key: str, value) -> None:
        """Append a value to a diagnostics list, creating it if needed."""
        if key not in self._diagnostics:
            self._diagnostics[key] = []
        self._diagnostics[key].append(value)
