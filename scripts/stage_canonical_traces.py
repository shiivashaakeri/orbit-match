# orbit-match/scripts/stage_canonical_traces.py
# Run: python -m scripts.stage_canonical_traces

"""Stage the canonical traces for paper figure rendering.

The render_paper_figures.py script reads from per-figure directories
under results/. Different experiments wrote traces to different
directories (lambda2_traces/, k_ablation/, k_gs_validation/). This
script copies the validated/correct traces into results/canonical/
under predictable filenames so the renderer has a single source.

Idempotent. Re-runnable. Will overwrite existing canonical traces.

Canonical layout
----------------
results/canonical/fig1/        Headline lambda2_traces (4 policies, 3 seeds)
results/canonical/fig3/        k-ablation key policies + ordering remark traces
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from orbitmatch.utils.io import RESULTS_ROOT
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


# (source_subdir, source_filename, dest_subdir, dest_filename)
COPIES = [
    # ---- Fig 1: lambda2_traces, 4 policies x 3 seeds ----
    ("lambda2_traces", "trace_medium_predictive_T574_H10_seed42.npz", "fig1", "predictive_seed42.npz"),
    ("lambda2_traces", "trace_medium_predictive_T574_H10_seed43.npz", "fig1", "predictive_seed43.npz"),
    ("lambda2_traces", "trace_medium_predictive_T574_H10_seed44.npz", "fig1", "predictive_seed44.npz"),
    ("lambda2_traces", "trace_medium_greedy_T574_H10_seed42.npz",     "fig1", "greedy_seed42.npz"),
    ("lambda2_traces", "trace_medium_greedy_T574_H10_seed43.npz",     "fig1", "greedy_seed43.npz"),
    ("lambda2_traces", "trace_medium_greedy_T574_H10_seed44.npz",     "fig1", "greedy_seed44.npz"),
    ("lambda2_traces", "trace_medium_random_T574_H10_seed42.npz",     "fig1", "random_seed42.npz"),
    ("lambda2_traces", "trace_medium_random_T574_H10_seed43.npz",     "fig1", "random_seed43.npz"),
    ("lambda2_traces", "trace_medium_random_T574_H10_seed44.npz",     "fig1", "random_seed44.npz"),
    # The validated k1_gs traces are the right "iterated BR" reference
    # for Fig 1. We use seed 42/43/44 with identity ordering.
    ("k_gs_validation", "trace_k1_gs_seed42_identity.npz",            "fig1", "k1_gs_seed42.npz"),
    ("k_gs_validation", "trace_k1_gs_seed43_identity.npz",            "fig1", "k1_gs_seed43.npz"),
    ("k_gs_validation", "trace_k1_gs_seed44_identity.npz",            "fig1", "k1_gs_seed44.npz"),

    # ---- Fig 3: update-order ablation ----
    # The "headline" comparison is k1_sync (= predictive) vs k1_gs identity.
    # Greedy is included as the baseline. Adaptive as an optional curve.
    # All on seed 42 since k1_gs is seed-deterministic (F11).
    ("k_ablation",      "trace_medium_k1_sync_seed42.npz",            "fig3", "k1_sync_seed42.npz"),
    ("k_gs_validation", "trace_k1_gs_seed42_identity.npz",            "fig3", "k1_gs_identity_seed42.npz"),
    ("k_gs_validation", "trace_k1_gs_seed42_reversed.npz",            "fig3", "k1_gs_reversed_seed42.npz"),
    ("k_gs_validation", "trace_k1_gs_seed42_random_p7.npz",           "fig3", "k1_gs_random_seed42.npz"),
    # We do NOT include greedy in fig3 since k1_sync == predictive == greedy
    # at the realized-edges level (F2). Including greedy would be redundant.
    ("k_ablation",      "trace_medium_adaptive_seed42.npz",           "fig3", "adaptive_seed42.npz"),
    ("k_ablation",      "trace_medium_equilibrium_seed42.npz",        "fig3", "equilibrium_seed42.npz"),
]


def main() -> int:
    configure(level="WARNING")

    canonical_root = RESULTS_ROOT / "canonical"
    canonical_root.mkdir(exist_ok=True)

    n_copied = 0
    n_missing = 0
    for src_subdir, src_name, dest_subdir, dest_name in COPIES:
        src = RESULTS_ROOT / src_subdir / src_name
        dest_dir = canonical_root / dest_subdir
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / dest_name
        if not src.exists():
            print(f"[MISS] {src_subdir}/{src_name} -> {dest_subdir}/{dest_name}")
            n_missing += 1
            continue
        shutil.copy2(src, dest)
        print(f"[OK]   {src_subdir}/{src_name} -> {dest_subdir}/{dest_name}")
        n_copied += 1

    print()
    print(f"Copied {n_copied} traces; {n_missing} missing.")
    print(f"Canonical traces are at: {canonical_root}")
    if n_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
