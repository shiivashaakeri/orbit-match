# orbit-match/orbitmatch/utils/seeding.py
# Run: imported by other modules; not a runnable script.

"""Centralized random-number generation for reproducible experiments.

All randomness in the project flows through this module. Modules that need
randomness call :func:`make_rng` with a seed, never ``np.random.seed`` or
the global ``np.random`` API.

Design
------
We use NumPy's modern :class:`numpy.random.Generator` interface (via
``np.random.default_rng``), not the legacy global ``np.random``. Generators
are explicit, thread-safe, and don't share state across modules.

For experiments that need multiple independent streams (e.g. one RNG for
initial pointing directions, another for the random-matching baseline),
:func:`spawn_rngs` splits a parent seed into deterministic child seeds
using NumPy's :class:`SeedSequence` machinery.

Every RNG creation is logged so that traces saved to disk can be
matched back to the seed that produced them.

Usage
-----
::

    from orbitmatch.utils.seeding import make_rng, spawn_rngs

    rng = make_rng(seed=42)
    x = rng.normal(size=10)

    # For multiple independent streams:
    rngs = spawn_rngs(parent_seed=42, n_streams=3, names=["init", "random_policy", "noise"])
    init_rng = rngs["init"]
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


def make_rng(seed: int) -> np.random.Generator:
    """Create a new NumPy Generator from an integer seed.

    Parameters
    ----------
    seed
        Non-negative integer seed. Logged at INFO level so traces can be
        traced back to the seed that produced them.

    Returns
    -------
    numpy.random.Generator
        A fresh generator. Independent of any other generator created
        elsewhere in the project.

    Raises
    ------
    ValueError
        If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative; got {seed}.")
    log.info("Creating RNG with seed=%d", seed)
    return np.random.default_rng(seed)


def spawn_rngs(
    parent_seed: int,
    n_streams: int | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, np.random.Generator]:
    """Spawn multiple independent RNGs from a single parent seed.

    Uses :class:`numpy.random.SeedSequence` to derive child seeds in a
    way that is statistically independent and deterministic: the same
    ``parent_seed`` and ``names`` always yield the same child RNGs.

    Parameters
    ----------
    parent_seed
        Non-negative integer.
    n_streams
        Number of streams to spawn. Ignored if ``names`` is given;
        otherwise required.
    names
        Optional human-readable names for the streams. If given, the
        returned dict uses these as keys; otherwise keys are
        ``"stream_0"``, ``"stream_1"``, ...

    Returns
    -------
    dict[str, numpy.random.Generator]
        Named, independent generators.

    Raises
    ------
    ValueError
        If neither ``n_streams`` nor ``names`` is given, or if both are
        given with inconsistent lengths.
    """
    if names is None and n_streams is None:
        raise ValueError("Must provide either `n_streams` or `names`.")
    if names is not None and n_streams is not None and len(names) != n_streams:
        raise ValueError(f"Length of `names` ({len(names)}) does not match `n_streams` ({n_streams}).")

    if names is None:
        names = [f"stream_{i}" for i in range(n_streams)]
    n_streams = len(names)

    parent_seq = np.random.SeedSequence(parent_seed)
    child_seqs = parent_seq.spawn(n_streams)

    log.info(
        "Spawning %d RNG streams from parent_seed=%d: %s",
        n_streams,
        parent_seed,
        list(names),
    )

    return {name: np.random.default_rng(seq) for name, seq in zip(names, child_seqs)}
