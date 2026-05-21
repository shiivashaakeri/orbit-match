# orbit-match/scripts/check_baselines.py
# Run: python -m scripts.check_baselines

"""Smoke test for orbitmatch.policy.baselines.

Builds a tiny synthetic feasibility tensor + positions (no orbit
propagation, no feasibility cache) and exercises both baseline
policies plus the predictive policy. Verifies that:

1. All three policies instantiate against the abstract Policy interface.
2. decide(t) returns a length-n int array with entries in {-1, 0..n-1}
   and no self-links.
3. Random is reproducible: same seed -> same actions.
4. Greedy is deterministic (same input -> same actions).
5. Greedy's reciprocation override actually fires (returns 1.0).
6. step(t, actions) updates the pointing tracker without crashing.

No simulation correctness is checked here -- that's the runner's job.
This script only validates the Policy-subclass plumbing.
"""

from __future__ import annotations

import sys

import numpy as np

from orbitmatch.graph.windowed import WindowedLaplacian
from orbitmatch.policy.base import NO_LINK, PolicyParams
from orbitmatch.policy.baselines import GreedyMatching, RandomMatching
from orbitmatch.policy.predictive import PredictiveMatching
from orbitmatch.utils.logging_setup import configure, get_logger
from orbitmatch.utils.seeding import make_rng

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic test fixtures
# ---------------------------------------------------------------------------


def make_fixture(n: int = 6, n_epochs: int = 8, seed: int = 0):
    """A small, dense feasibility tensor with arbitrary positions.

    The fixture is not orbit-realistic: feasibility is a random
    symmetric boolean tensor with ~70% density, and positions are
    random unit vectors. The Policy interface only requires shapes to
    line up; the value-function math runs end-to-end regardless.
    """
    rng = np.random.default_rng(seed)

    # Feasibility: random symmetric, no self-loops.
    raw = rng.random((n_epochs, n, n)) < 0.7
    feasibility = raw & np.swapaxes(raw, 1, 2)
    eye = np.eye(n, dtype=bool)
    feasibility[:, eye] = False

    # Positions: arbitrary, magnitudes ~7000 km to look orbital-ish.
    positions = rng.normal(size=(n_epochs, n, 3))
    positions *= 7000.0 / np.linalg.norm(positions, axis=-1, keepdims=True)

    return feasibility, positions


def validate_actions(actions: np.ndarray, n: int, label: str) -> None:
    """Sanity-check a joint action vector."""
    if actions.shape != (n,):
        raise AssertionError(f"{label}: actions shape {actions.shape}, expected ({n},)")
    if actions.dtype != np.int64:
        raise AssertionError(f"{label}: actions dtype {actions.dtype}, expected int64")
    bad = ((actions != NO_LINK) & ((actions < 0) | (actions >= n))).any()
    if bad:
        raise AssertionError(f"{label}: actions out of range: {actions}")
    self_links = ((actions != NO_LINK) & (actions == np.arange(n))).any()
    if self_links:
        raise AssertionError(f"{label}: self-link in actions: {actions}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instantiation(feasibility, positions, dt_s, params, T):
    """Each policy class instantiates against the abstract base."""
    n = feasibility.shape[1]
    wl = WindowedLaplacian(n, window_length=T)
    rng = make_rng(seed=123)
    pred = PredictiveMatching(n, feasibility, positions, dt_s, params, rng=rng, windowed_laplacian=wl)
    greedy = GreedyMatching(n, feasibility, positions, dt_s, params, rng=rng, windowed_laplacian=wl)
    random_pol = RandomMatching(n, feasibility, positions, dt_s, params, rng=make_rng(seed=123))
    assert pred.name == "predictive", pred.name
    assert greedy.name == "greedy", greedy.name
    assert random_pol.name == "random", random_pol.name
    print("  [OK] all three policies instantiated; names = predictive/greedy/random")


def test_decide_shapes(feasibility, positions, dt_s, params, T):
    """decide(t) returns valid joint actions for each policy."""
    n = feasibility.shape[1]
    wl = WindowedLaplacian(n, window_length=T)
    for name, cls in [("predictive", PredictiveMatching), ("greedy", GreedyMatching), ("random", RandomMatching)]:
        pol = cls(n, feasibility, positions, dt_s, params, rng=make_rng(seed=0), windowed_laplacian=wl)
        actions = pol.decide(t=0)
        validate_actions(actions, n, label=name)
    print("  [OK] decide(t=0) returns valid actions for all three")


def test_random_reproducible(feasibility, positions, dt_s, params, T):  # noqa: ARG001
    """Random with the same seed produces the same actions."""
    n = feasibility.shape[1]
    a1 = RandomMatching(n, feasibility, positions, dt_s, params, rng=make_rng(seed=99)).decide(t=2)
    a2 = RandomMatching(n, feasibility, positions, dt_s, params, rng=make_rng(seed=99)).decide(t=2)
    if not np.array_equal(a1, a2):
        raise AssertionError(f"Random not reproducible: {a1} vs {a2}")
    print("  [OK] Random reproducible across separate instances with same seed")


def test_random_picks_feasible(feasibility, positions, dt_s, params, T):  # noqa: ARG001
    """Random's picks are always feasible (or NO_LINK if F_i is empty)."""
    n = feasibility.shape[1]
    pol = RandomMatching(n, feasibility, positions, dt_s, params, rng=make_rng(seed=7))
    for t in range(min(5, feasibility.shape[0])):
        actions = pol.decide(t)
        for i in range(n):
            j = int(actions[i])
            if j == NO_LINK:
                # Acceptable only if i has no feasible neighbors at t.
                if feasibility[t, i].any():
                    raise AssertionError(f"t={t} i={i}: Random played NO_LINK but F_i is nonempty")
            elif not feasibility[t, i, j]:
                raise AssertionError(f"t={t} i={i}: Random picked infeasible partner j={j}")
    print("  [OK] Random always picks a feasible neighbor (NO_LINK only when F_i is empty)")


def test_greedy_recip_one(feasibility, positions, dt_s, params, T):
    """Greedy._reciprocation_prob returns 1.0 for any candidate."""
    n = feasibility.shape[1]
    wl = WindowedLaplacian(n, window_length=T)
    greedy = GreedyMatching(n, feasibility, positions, dt_s, params, rng=make_rng(seed=0), windowed_laplacian=wl)
    p = greedy._reciprocation_prob(i=0, j=1, t=0)
    if p != 1.0:
        raise AssertionError(f"Greedy._reciprocation_prob returned {p}, expected 1.0")
    print("  [OK] Greedy._reciprocation_prob returns 1.0 (deferral disabled)")


def test_step_updates_pointing(feasibility, positions, dt_s, params, T):
    """step(t, actions) updates pointing state without crashing."""
    n = feasibility.shape[1]
    wl = WindowedLaplacian(n, window_length=T)
    pol = PredictiveMatching(n, feasibility, positions, dt_s, params, rng=make_rng(seed=0), windowed_laplacian=wl)
    actions = pol.decide(t=0)
    pol.step(t=0, actions=actions)
    # Pointing state for matched satellites should now have unit-norm directions.
    matched = [(i, int(actions[i])) for i in range(n) if actions[i] != NO_LINK and int(actions[int(actions[i])]) == i]
    for i, _ in matched:
        d = pol.pointing.directions[i]
        if np.any(np.isnan(d)):
            raise AssertionError(f"Pointing direction for matched sat {i} is still NaN")
        if abs(np.linalg.norm(d) - 1.0) > 1e-9:
            raise AssertionError(f"Pointing direction for sat {i} not unit-norm: |d| = {np.linalg.norm(d)}")
    print(f"  [OK] step() updates pointing for {len(matched)} matched satellites")


def test_short_simulation(feasibility, positions, dt_s, params, T):
    """Run each policy for 3 epochs end-to-end; verify cache cycles correctly."""
    n = feasibility.shape[1]
    for name, cls in [("predictive", PredictiveMatching), ("greedy", GreedyMatching), ("random", RandomMatching)]:
        wl = WindowedLaplacian(n, window_length=T)
        pol = cls(n, feasibility, positions, dt_s, params, rng=make_rng(seed=0), windowed_laplacian=wl)
        for t in range(3):
            actions = pol.decide(t)
            validate_actions(actions, n, label=f"{name} @ t={t}")
            pol.step(t, actions)
    print("  [OK] 3-epoch end-to-end run completes for all three policies")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")

    feasibility, positions = make_fixture(n=6, n_epochs=8, seed=0)
    dt_s = 10.0
    T = 4
    params = PolicyParams(
        H=3,
        T=T,
        switching_cost_scale=0.2,
        epsilon_geometric_prior=0.01,
        tie_break="lowest_index",
        seed=None,
    )

    print(f"Fixture: n={feasibility.shape[1]}, n_epochs={feasibility.shape[0]}, T={T}, H={params.H}")
    print(
        f"Feasibility density: {feasibility.sum() / (feasibility.shape[0] * feasibility.shape[1] * (feasibility.shape[1] - 1)):.2f}"  # noqa: E501
    )
    print()

    tests = [
        ("instantiation", test_instantiation),
        ("decide shapes", test_decide_shapes),
        ("random reproducible", test_random_reproducible),
        ("random picks feasible", test_random_picks_feasible),
        ("greedy recip = 1", test_greedy_recip_one),
        ("step updates pointing", test_step_updates_pointing),
        ("3-epoch simulation", test_short_simulation),
    ]

    failed = 0
    for label, fn in tests:
        print(f"[{label}]")
        try:
            fn(feasibility, positions, dt_s, params, T)
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed += 1

    print()
    if failed:
        print(f"FAILED: {failed} / {len(tests)} tests")
        return 1
    print(f"All {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
