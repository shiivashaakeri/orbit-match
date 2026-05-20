# orbit-match/orbitmatch/policy/base.py
# Run: imported by other modules; not a runnable script.

"""Abstract policy interface and shared scaffolding.

A policy maps each satellite's information state to an action at every
epoch. In this codebase, a Policy object is constructed once per
simulation with the full feasibility tensor and pointing-cost machinery,
and is then driven by the simulation loop:

    policy = SomePolicy(n, feasibility, dt_s, params, rng)
    for t in range(n_epochs):
        actions = policy.decide(t)
        matching = actions_to_edges(actions)
        # ... apply matching to the world ...
        policy.step(t, actions)

The :class:`Policy` base provides:

- The shared parameter dataclass :class:`PolicyParams`.
- A standard ``__init__`` that consumes ``(n, feasibility, dt_s, params, rng)``.
- A pointing-state tracker that the switching cost depends on.
- A default ``decide(t)`` that loops over satellites calling
  ``decide_for(i, t)``. Subclasses can override ``decide(t)`` to
  vectorize. The per-satellite path is exposed for honesty and
  diagnostics.
- A ``step(t, actions)`` hook that updates pointing state after the
  joint action is realized.

Concrete policies live in sibling modules:
:mod:`orbitmatch.policy.predictive` (the §III rule),
:mod:`orbitmatch.policy.baselines` (greedy as a subclass; random as
a sibling), and :mod:`orbitmatch.policy.equilibrium` (full
best-response dynamics).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from orbitmatch.feasibility.predicates import FeasibilityParams  # noqa: F401
from orbitmatch.utils.logging_setup import get_logger

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
        Constant ``c`` in ``C^switch_{ij}(t) = c * angle(theta_i, theta_ij_hat)``.
        Set to 0 to disable the switching-cost term entirely.
    tie_break
        How to break ties in the argmax over candidate partners.
        ``"lowest_index"`` is deterministic and used in the paper;
        ``"random"`` uses the policy's RNG.
    seed
        Random seed for any policy that needs randomness (e.g. tie
        breaking, the random baseline). Optional.
    """

    H: int = 10
    T: int = 30
    switching_cost_scale: float = 1.0
    tie_break: str = "lowest_index"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.H <= 0:
            raise ValueError(f"H must be positive; got {self.H}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive; got {self.T}.")
        if self.switching_cost_scale < 0:
            raise ValueError(f"switching_cost_scale must be non-negative; got {self.switching_cost_scale}.")
        if self.tie_break not in {"lowest_index", "random"}:
            raise ValueError(f"tie_break must be 'lowest_index' or 'random'; got {self.tie_break!r}.")


# ---------------------------------------------------------------------------
# Pointing-state tracker
# ---------------------------------------------------------------------------


@dataclass
class PointingState:
    """Tracks each satellite's current pointing direction.

    The "current pointing direction" is the unit bearing to the partner
    the satellite is currently pointed at. If a satellite has no active
    link (the previous action was NO_LINK or the partner did not
    reciprocate), its pointing direction is :data:`np.nan`. The switching
    cost then treats it as "free" (zero) to point at any new partner —
    the satellite has no commitment to break.

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

        - If ``actions[i] == j`` and ``actions[j] == i``, the link formed.
          Set ``directions[i]`` to the unit bearing toward ``j``'s position
          at this epoch.
        - Otherwise (no link or non-mutual), keep the previous direction.
          The satellite has either committed pointing-time it can carry
          into the next epoch, or it has nothing to update.

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
                # Mutual link; update pointing.
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

    Subclasses implement :meth:`decide_for` (per-satellite decision); the
    default :meth:`decide` loops over satellites. Subclasses can override
    :meth:`decide` if a vectorized form is faster.

    Parameters
    ----------
    n
        Number of satellites.
    feasibility
        Boolean ``(n_epochs, n, n)`` feasibility tensor from
        :mod:`orbitmatch.feasibility.compute`.
    positions
        ECI positions ``(n_epochs, n, 3)`` from
        :mod:`orbitmatch.constellation.propagator`. The policy uses this
        for switching-cost geometry.
    dt_s
        Epoch length in seconds.
    params
        :class:`PolicyParams` instance.
    rng
        Numpy ``Generator``; used by random-tie-break policies and the
        Random baseline. May be ``None`` for deterministic policies.
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
        # Diagnostics collected over the run; subclasses extend as needed.
        self._diagnostics: dict[str, list] = {}

    # ---- Decision interface -------------------------------------------------

    @abstractmethod
    def decide_for(self, i: int, t: int) -> int:
        """Return satellite ``i``'s requested partner at epoch ``t``.

        Returns ``-1`` (:data:`NO_LINK`) to indicate no request.
        """

    def decide(self, t: int) -> np.ndarray:
        """Joint action at epoch ``t``: ``(n,)`` int array of partner indices.

        Default implementation loops over satellites. Override for
        vectorized policies (Random, Greedy in some forms).
        """
        actions = np.full(self.n, NO_LINK, dtype=np.int64)
        for i in range(self.n):
            actions[i] = self.decide_for(i, t)
        return actions

    def step(self, t: int, actions: np.ndarray) -> None:
        """Hook called after the joint action is realized.

        Default behavior: update pointing state from the realized actions
        (which embed the mutual-choice rule via :meth:`PointingState.update_from_realized`).
        Subclasses may extend to track additional state.
        """
        self.pointing.update_from_realized(actions, self.positions[t])

    # ---- Diagnostics --------------------------------------------------------

    def diagnostics(self) -> dict[str, list]:
        """Return a copy of the diagnostics dict.

        Each policy records its own per-epoch counters (deferral counts,
        BR-round counts, etc.) here.
        """
        return {k: list(v) for k, v in self._diagnostics.items()}

    def _record(self, key: str, value) -> None:
        """Append a value to a diagnostics list, creating the key if needed."""
        if key not in self._diagnostics:
            self._diagnostics[key] = []
        self._diagnostics[key].append(value)
