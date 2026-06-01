# orbit-match/orbitmatch/utils/timing.py
# Run: imported by other modules; not a runnable script.

"""Timing utilities.

A context manager and a decorator for measuring wall-clock time of code
blocks and functions, with consistent logging.

Usage
-----
As a context manager::

    from orbitmatch.utils.timing import timed

    with timed("feasibility computation"):
        F = compute_feasibility(positions, dt, params)

As a decorator::

    from orbitmatch.utils.timing import timeit

    @timeit
    def run_simulation(...):
        ...

Both emit a single log line at completion: ``"feasibility computation: 4.221 s"``.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)

F = TypeVar("F", bound=Callable)


@contextmanager
def timed(label: str, level: str = "INFO") -> Iterator[dict[str, float]]:
    """Measure wall-clock time of a code block.

    Parameters
    ----------
    label
        Human-readable identifier for the block; appears in the log line.
    level
        Logging level for the completion message. Use ``"DEBUG"`` to
        suppress under default settings.

    Yields
    ------
    dict
        A single-key dict ``{"elapsed_s": float}``. The value is written
        on exit, so callers can read it after the ``with`` block exits::

            with timed("step") as t:
                ...
            print(t["elapsed_s"])
    """
    holder: dict[str, float] = {"elapsed_s": float("nan")}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        elapsed = time.perf_counter() - start
        holder["elapsed_s"] = elapsed
        log.log(_level_to_int(level), "%s: %s", label, _format_duration(elapsed))


def timeit(fn: F) -> F:
    """Decorate a function to log its wall-clock execution time.

    The function's qualified name appears as the label.

    Example
    -------
    ::

        @timeit
        def expensive_step(x):
            ...
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            log.info("%s: %s", fn.__qualname__, _format_duration(elapsed))

    return wrapper  # type: ignore[return-value]


def _format_duration(seconds: float) -> str:
    """Format a duration with a unit that keeps numbers readable.

    - ``< 1 ms``  -> microseconds with 1 decimal
    - ``< 1 s``   -> milliseconds with 2 decimals
    - ``< 60 s``  -> seconds with 3 decimals
    - ``>= 60 s`` -> minutes and seconds
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, secs = divmod(seconds, 60.0)
    return f"{int(minutes)}m {secs:.1f}s"


def _level_to_int(level: str) -> int:
    """Convert a level string to its numeric value (DEBUG, INFO, etc.)."""
    import logging  # noqa: PLC0415

    return logging.getLevelName(level.upper())
