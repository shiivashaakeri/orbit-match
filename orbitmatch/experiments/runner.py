# orbit-match/orbitmatch/experiments/runner.py
# Run: imported by scripts; not a runnable script.

"""Central simulation driver.

`run_simulation(...)` is the function every paper script ends up calling.
Given a fully-specified constellation, policy, and simulation block, it:

1. Builds the Walker--Delta orbital elements.
2. Propagates positions on a uniform time grid.
3. Computes (or loads from cache) the feasibility tensor.
4. Constructs the WindowedLaplacian and WindowedUnion at the
   requested certificate window length T.
5. Instantiates the policy by name and runs it for n_epochs.
6. Records per-epoch lambda_2(Phi(t)), lambda_2(L^union_G(t; T)),
   the realized matching, and the policy's diagnostics dict.
7. Returns a frozen :class:`SimulationResult` carrying every trace
   needed to redraw any figure in the paper.

This module is pure: it does no disk persistence (other than the
feasibility cache hit, which is content-addressed and side-effect-
free with respect to results). Callers (scripts) are responsible
for saving the returned result via :func:`orbitmatch.utils.io.save_trace`.

Design
------
- :class:`SimulationResult` is a frozen dataclass with typed fields.
  Matchings are stored as a tuple of (k_t, 2) int64 arrays, which
  preserves jagged structure without padding. Numeric traces are
  numpy arrays of length ``n_epochs``.

- Policy class lookup is a small registry, keyed on the canonical
  policy names from PROJECT_PLAN.md. The runner intentionally does
  not auto-discover; new policies are added explicitly.

- ``equilibrium`` is currently a TODO: the class lives in
  ``policy.equilibrium`` which is still empty. Calling the runner
  with ``policy_name="equilibrium"`` raises NotImplementedError until
  that module is filled in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.feasibility.compute import load_or_compute_feasibility
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.graph.laplacian import actions_to_edges
from orbitmatch.graph.windowed import WindowedLaplacian, WindowedUnion
from orbitmatch.policy.adaptive import AdaptivePredictive
from orbitmatch.policy.base import Policy, PolicyParams
from orbitmatch.policy.baselines import GreedyMatching, RandomMatching
from orbitmatch.policy.equilibrium import EquilibriumMatching
from orbitmatch.policy.k_step import KStepPredictive
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import get_logger
from orbitmatch.utils.seeding import make_rng
from orbitmatch.utils.timing import timed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


_POLICY_REGISTRY: dict[str, type[Policy]] = {
    "predictive": PredictiveMatching,
    "greedy": GreedyMatching,
    "random": RandomMatching,
    "equilibrium": EquilibriumMatching,
    "k_step": KStepPredictive,
    "adaptive": AdaptivePredictive,
}


def get_policy_class(name: str) -> type[Policy]:
    """Return the Policy subclass registered under ``name``."""
    if name not in _POLICY_REGISTRY:
        raise KeyError(f"Unknown policy {name!r}. Known: {sorted(_POLICY_REGISTRY)}.")
    return _POLICY_REGISTRY[name]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Everything produced by one call to :func:`run_simulation`.

    Numeric traces are float64 arrays of length ``n_epochs``. The
    ``matchings`` field is a tuple of (k_t, 2) int64 arrays, one per
    epoch; ``k_t`` varies. The ``actions`` field is an ``(n_epochs, n)``
    int64 array holding the raw policy requests (before the mutual-
    choice rule reduces them to a matching).

    All metadata needed to reconstruct the run (constellation,
    feasibility thresholds, policy name and params, seeds, dt) lives
    in the ``metadata`` dict; scripts pass this through to
    :func:`save_trace`.
    """

    # Identity and provenance.
    policy_name: str
    config_label: str
    n: int
    n_epochs: int
    T: int
    dt_s: float

    # Per-epoch numeric traces (length n_epochs).
    lambda2_phi: np.ndarray
    lambda2_union: np.ndarray
    n_edges_per_epoch: np.ndarray
    n_union_edges_per_epoch: np.ndarray

    # Actions and realized matchings.
    actions: np.ndarray
    matchings: tuple[np.ndarray, ...]

    # Final-state objects, for one-shot rho/alpha_0 diagnostics in scripts.
    final_phi: np.ndarray
    final_union_adjacency: np.ndarray

    # Policy-specific diagnostics (e.g. PredictiveMatching's "deferrals").
    policy_diagnostics: dict[str, np.ndarray] = field(default_factory=dict)

    # Provenance metadata for save_trace.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Pack numeric fields into the dict shape expected by save_trace.

        Jagged matchings are stored as an object-dtype array so they
        survive the .npz round-trip.
        """
        arrays: dict[str, np.ndarray] = {
            "lambda2_phi": self.lambda2_phi,
            "lambda2_union": self.lambda2_union,
            "n_edges_per_epoch": self.n_edges_per_epoch,
            "n_union_edges_per_epoch": self.n_union_edges_per_epoch,
            "actions": self.actions,
            "matchings": np.asarray(self.matchings, dtype=object),
            "final_phi": self.final_phi,
            "final_union_adjacency": self.final_union_adjacency,
        }
        for k, v in self.policy_diagnostics.items():
            arrays[f"diag_{k}"] = v
        return arrays


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_simulation(
    walker: WalkerDeltaConfig,
    feasibility_params: FeasibilityParams,
    policy_name: str,
    policy_params: PolicyParams,
    *,
    n_epochs: int,
    dt_s: float,
    seed: int,
    config_label: str,
    feasibility_cache_dir: Optional[Any] = None,
) -> SimulationResult:
    """Run one policy on one constellation for n_epochs and return the trace.

    Parameters
    ----------
    walker
        Walker-Delta config (orbital geometry).
    feasibility_params
        Thresholds for the three feasibility predicates.
    policy_name
        One of ``{"predictive", "greedy", "random", "equilibrium"}``.
    policy_params
        :class:`PolicyParams` with ``H``, ``T``, ``c``, ``epsilon``, etc.
        The ``T`` field is the certificate window length used by the
        WindowedLaplacian / WindowedUnion.
    n_epochs
        Number of simulation epochs to run.
    dt_s
        Epoch length in seconds.
    seed
        Integer seed used to construct the policy's RNG.
    config_label
        Short identifier used in the feasibility cache filename
        (e.g. ``"small"`` or ``"medium"``).
    feasibility_cache_dir
        Optional override for the feasibility cache directory. Defaults
        to ``data/processed/`` via the existing utility.

    Returns
    -------
    SimulationResult
        Frozen dataclass with all traces and metadata. See class doc.
    """
    PolicyCls = get_policy_class(policy_name)

    # ---- 1. Orbital propagation --------------------------------------------
    elements = walker.initial_elements()
    times_s = make_time_grid(duration_s=n_epochs * dt_s, dt_s=dt_s, inclusive_end=False)
    if len(times_s) != n_epochs:
        # make_time_grid uses floor; for our setup the two agree, but guard anyway.
        log.warning("Time grid length %d != requested n_epochs %d; using time grid length.", len(times_s), n_epochs)
        n_epochs = len(times_s)
    positions = propagate_keplerian(elements, times_s)

    # ---- 2. Feasibility tensor (cached) ------------------------------------
    feasibility, cache_path = load_or_compute_feasibility(
        positions,
        dt_s,
        feasibility_params,
        config_label=f"walker_{config_label}_{walker.M}_{walker.P}_{walker.F}_alt{int(walker.altitude_km)}_inc{int(walker.inclination_deg)}",
        cache_dir=feasibility_cache_dir,
    )

    n = walker.M
    T = policy_params.T

    # ---- 3. Windowed objects -----------------------------------------------
    windowed_phi = WindowedLaplacian(n, window_length=T)
    windowed_union = WindowedUnion(n, window_length=T)

    # ---- 4. Policy ---------------------------------------------------------
    rng = make_rng(seed=seed)
    policy = PolicyCls(
        n=n,
        feasibility=feasibility,
        positions=positions,
        dt_s=dt_s,
        params=policy_params,
        rng=rng,
        windowed_laplacian=windowed_phi,
    )

    log.info(
        "Running %s for %d epochs (n=%d, T=%d, seed=%d)",
        policy_name,
        n_epochs,
        n,
        T,
        seed,
    )

    # ---- 5. Main loop ------------------------------------------------------
    lambda2_phi = np.zeros(n_epochs, dtype=np.float64)
    lambda2_union = np.zeros(n_epochs, dtype=np.float64)
    n_edges_per_epoch = np.zeros(n_epochs, dtype=np.int64)
    n_union_edges_per_epoch = np.zeros(n_epochs, dtype=np.int64)
    actions_history = np.full((n_epochs, n), -1, dtype=np.int64)
    matchings_list: list[np.ndarray] = []

    with timed(f"run_simulation[{policy_name}]"):
        for t in range(n_epochs):
            actions = policy.decide(t)
            matching = actions_to_edges(actions)

            windowed_phi.push(matching)
            windowed_union.push(matching)

            lambda2_phi[t] = windowed_phi.lambda_2()
            lambda2_union[t] = windowed_union.lambda_2()
            n_edges_per_epoch[t] = matching.shape[0]
            n_union_edges_per_epoch[t] = windowed_union.n_edges

            actions_history[t] = actions
            matchings_list.append(matching)

            policy.step(t, actions)

    # ---- 6. Diagnostics ----------------------------------------------------
    diag_dict = policy.diagnostics()
    policy_diagnostics: dict[str, np.ndarray] = {
        k: np.asarray(v) for k, v in diag_dict.items()
    }

    # ---- 7. Pack ------------------------------------------------------------
    metadata = {
        "policy_name": policy_name,
        "config_label": config_label,
        "walker": {
            "M": walker.M,
            "P": walker.P,
            "F": walker.F,
            "altitude_km": walker.altitude_km,
            "inclination_deg": walker.inclination_deg,
        },
        "feasibility_params": asdict(feasibility_params),
        "policy_params": asdict(policy_params),
        "n_epochs": n_epochs,
        "dt_s": dt_s,
        "seed": seed,
        "feasibility_cache": str(cache_path),
    }

    result = SimulationResult(
        policy_name=policy_name,
        config_label=config_label,
        n=n,
        n_epochs=n_epochs,
        T=T,
        dt_s=dt_s,
        lambda2_phi=lambda2_phi,
        lambda2_union=lambda2_union,
        n_edges_per_epoch=n_edges_per_epoch,
        n_union_edges_per_epoch=n_union_edges_per_epoch,
        actions=actions_history,
        matchings=tuple(matchings_list),
        final_phi=windowed_phi.phi.copy(),
        final_union_adjacency=windowed_union.adjacency.copy(),
        policy_diagnostics=policy_diagnostics,
        metadata=metadata,
    )

    log.info(
        "%s done: final lambda2(Phi)=%.4f, lambda2(L_union_G)=%.4f, %d union edges",
        policy_name,
        float(lambda2_phi[-1]),
        float(lambda2_union[-1]),
        int(n_union_edges_per_epoch[-1]),
    )
    return result
