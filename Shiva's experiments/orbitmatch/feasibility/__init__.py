# orbit-match/orbitmatch/feasibility/__init__.py
"""Inter-satellite link feasibility computation."""

from orbitmatch.feasibility.compute import (
    compute_feasibility,
    feasibility_union,
    feasible_neighbors,
    load_or_compute_feasibility,
)
from orbitmatch.feasibility.predicates import (
    FeasibilityParams,
    line_of_sight,
    pointing_rate_ok,
    range_ok,
)

__all__ = [
    "FeasibilityParams",
    "compute_feasibility",
    "feasibility_union",
    "feasible_neighbors",
    "line_of_sight",
    "load_or_compute_feasibility",
    "pointing_rate_ok",
    "range_ok",
]
