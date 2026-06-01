# orbit-match/orbitmatch/experiments/sweep.py
# Run: imported by scripts; not a runnable script.

"""Parameter sweep driver.

A *sweep* is a set of simulation runs over a Cartesian product of
override dicts and integer seeds, e.g.

    overrides = [{"H": 1}, {"H": 5}, {"H": 10}, {"H": 20}, {"H": 50}]
    seeds     = [42, 43, 44, 45, 46]

producing 25 (run, diagnostics) pairs. The same mechanism handles
H sweeps, T sweeps, scaling sweeps (where the "override" is a swap
of the WalkerDeltaConfig), and any future single-axis ablation.

What this module does
---------------------
- Expands a sweep spec into a flat list of (override, seed) jobs.
- For each job: applies the override to a base config, runs the
  simulation, builds the DiagnosticsReport, saves the trace.
- Honors the "never re-run" rule: if a trace file already exists at
  the canonical path, the job is skipped (set ``force=True`` to
  override).
- Returns a list of :class:`SweepRecord` carrying override label,
  seed, file path, and report. The full SimulationResult lives on
  disk; only the report is held in memory for plotting / tables.

What this module does NOT do
----------------------------
- Plot anything (that belongs in ``plotting/paper_plots.py``).
- Decide which sweep to run (that belongs in ``scripts/run_*.py``).
- Group / aggregate across seeds (that belongs to the caller; the
  records contain enough information for any grouping).

Conventions
-----------
The sweep file name follows PROJECT_PLAN.md Sec 5.3:
    trace_{config}_{policy}_{params_hash}_seed{seed}.npz
where ``params_hash`` is a short content hash over (override, base
policy params). Two distinct override dicts always produce distinct
filenames; the same override always produces the same filename.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.experiments.diagnostics import DiagnosticsReport, build_report, compute_alpha_0
from orbitmatch.experiments.runner import run_simulation
from orbitmatch.feasibility.compute import compute_feasibility
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.policy.base import PolicyParams
from orbitmatch.utils.io import load_trace, save_trace
from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepRecord:
    """One (override, seed) result in a sweep.

    Holds the override label and seed plus the diagnostics report.
    The full SimulationResult is on disk at :attr:`trace_path`; load
    it via :func:`load_trace` when plots need more than the report.
    """

    override_label: str  # human-readable, e.g. "H=10"
    override_key: str  # canonical key, e.g. "H_10"
    seed: int
    config_label: str  # constellation label
    policy_name: str
    trace_path: Path
    report: DiagnosticsReport
    skipped_existing: bool  # True if the trace was loaded from cache


# ---------------------------------------------------------------------------
# Sweep specification helpers
# ---------------------------------------------------------------------------


def expand_overrides(sweep: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a single-axis sweep dict into a flat list of overrides.

    The sweep dict is expected to be single-axis (one key, list of
    values). Multi-axis sweeps are not supported here -- if you want
    e.g. (H, T) grid, write a script that calls this twice or builds
    its own list directly.

    Parameters
    ----------
    sweep
        ``{axis_name: [v1, v2, ...]}``.

    Returns
    -------
    list of dict
        One ``{axis_name: v_i}`` per value.
    """
    if len(sweep) != 1:
        raise ValueError(f"expand_overrides expects a single-axis sweep; got {len(sweep)} axes: {list(sweep)}.")
    ((axis, values),) = sweep.items()
    return [{axis: v} for v in values]


def override_label(override: dict[str, Any]) -> tuple[str, str]:
    """Produce (human-readable, canonical) labels for an override dict.

    Example
    -------
    {"H": 10}            -> ("H=10",     "H_10")
    {"T": 60}            -> ("T=60",     "T_60")
    {"walker": cfg}      -> ("n=24",     "n_24")    (uses cfg.M)
    """
    if len(override) != 1:
        # Multi-key overrides: concatenate.
        parts_h = []
        parts_c = []
        for k, v in sorted(override.items()):
            h, c = override_label({k: v})
            parts_h.append(h)
            parts_c.append(c)
        return ", ".join(parts_h), "__".join(parts_c)

    ((k, v),) = override.items()
    if k == "walker" and isinstance(v, WalkerDeltaConfig):
        return f"n={v.M}", f"n_{v.M}"
    return f"{k}={v}", f"{k}_{v}"


# ---------------------------------------------------------------------------
# Internal: apply override to a config
# ---------------------------------------------------------------------------


def _apply_override(
    *,
    base_walker: WalkerDeltaConfig,
    base_policy_params: PolicyParams,
    base_n_epochs: int,
    override: dict[str, Any],
) -> tuple[WalkerDeltaConfig, PolicyParams, int]:
    """Return (walker, policy_params, n_epochs) with override applied.

    Supported override keys:
    - ``"H"``: integer, replaces ``policy_params.H``.
    - ``"T"``: integer, replaces ``policy_params.T``.
    - ``"switching_cost_scale"``: float, replaces the c in policy_params.
    - ``"epsilon_geometric_prior"``: float, replaces epsilon.
    - ``"walker"``: WalkerDeltaConfig, replaces the constellation.
    - ``"n_epochs"``: integer, replaces n_epochs.

    Unsupported keys raise ``KeyError``.
    """
    walker = base_walker
    pp_dict = asdict(base_policy_params)
    n_epochs = base_n_epochs

    POLICY_KEYS = {"H", "T", "switching_cost_scale", "epsilon_geometric_prior", "tie_break", "seed"}

    for k, v in override.items():
        if k == "walker":
            if not isinstance(v, WalkerDeltaConfig):
                raise TypeError(f"override['walker'] must be WalkerDeltaConfig; got {type(v).__name__}")
            walker = v
        elif k == "n_epochs":
            n_epochs = int(v)
        elif k in POLICY_KEYS:
            pp_dict[k] = v
        else:
            raise KeyError(f"Unsupported override key: {k!r}")

    return walker, PolicyParams(**pp_dict), n_epochs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    base_walker: WalkerDeltaConfig,
    feasibility_params: FeasibilityParams,
    base_policy_name: str,
    base_policy_params: PolicyParams,
    base_n_epochs: int,
    dt_s: float,
    overrides: Iterable[dict[str, Any]],
    seeds: Iterable[int],
    config_label: str,
    results_dir: Path,
    force: bool = False,
    share_alpha_0: bool = True,
) -> list[SweepRecord]:
    """Run a sweep over (override, seed) jobs and return SweepRecords.

    Parameters
    ----------
    base_walker, feasibility_params, base_policy_name, base_policy_params,
    base_n_epochs, dt_s
        Defaults that each override may modify.
    overrides
        Iterable of override dicts. Use :func:`expand_overrides` to
        produce these from a sweep YAML.
    seeds
        Iterable of integer seeds.
    config_label
        Short identifier used in the feasibility cache filename and
        as the ``config`` field in the trace filename.
    results_dir
        Directory to save traces to. Created if missing.
    force
        If True, recompute even when the trace file already exists.
    share_alpha_0
        If True, compute alpha_0 once per *unique constellation* in the
        sweep and pass it into every diagnostics build. Major speedup
        for sweeps that don't vary the constellation (H, T sweeps).
        Ignored for sweeps that do vary it (scaling sweep): in that
        case alpha_0 must be recomputed per override.

    Returns
    -------
    list of SweepRecord
        One record per (override, seed) job.
    """
    overrides = list(overrides)
    seeds = list(seeds)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Sweep: %d overrides x %d seeds = %d jobs (results_dir=%s)",
        len(overrides),
        len(seeds),
        len(overrides) * len(seeds),
        results_dir,
    )

    # Cache alpha_0 per constellation if requested.
    alpha_0_cache: dict[str, tuple[float, int]] = {}

    records: list[SweepRecord] = []
    for ov in overrides:
        h_label, c_label = override_label(ov)
        for seed in seeds:
            rec = _run_one_job(
                base_walker=base_walker,
                feasibility_params=feasibility_params,
                base_policy_name=base_policy_name,
                base_policy_params=base_policy_params,
                base_n_epochs=base_n_epochs,
                dt_s=dt_s,
                override=ov,
                override_label=h_label,
                override_key=c_label,
                seed=seed,
                config_label=config_label,
                results_dir=results_dir,
                force=force,
                alpha_0_cache=alpha_0_cache if share_alpha_0 else None,
            )
            records.append(rec)

    log.info(
        "Sweep complete: %d records (%d new, %d loaded from cache)",
        len(records),
        sum(1 for r in records if not r.skipped_existing),
        sum(1 for r in records if r.skipped_existing),
    )
    return records


def _trace_filename(config_label: str, policy_name: str, override_key: str, seed: int) -> str:
    """Build the canonical trace filename per PROJECT_PLAN Sec 5.3."""
    return f"trace_{config_label}_{policy_name}_{override_key}_seed{seed}.npz"


def _run_one_job(
    *,
    base_walker: WalkerDeltaConfig,
    feasibility_params: FeasibilityParams,
    base_policy_name: str,
    base_policy_params: PolicyParams,
    base_n_epochs: int,
    dt_s: float,
    override: dict[str, Any],
    override_label: str,
    override_key: str,
    seed: int,
    config_label: str,
    results_dir: Path,
    force: bool,
    alpha_0_cache: Optional[dict[str, tuple[float, int]]],
) -> SweepRecord:
    """Run one (override, seed) job. Internal helper."""
    walker, policy_params, n_epochs = _apply_override(
        base_walker=base_walker,
        base_policy_params=base_policy_params,
        base_n_epochs=base_n_epochs,
        override=override,
    )

    # If the override replaces the constellation, use its label in the filename.
    eff_config_label = f"{config_label}_n{walker.M}" if "walker" in override else config_label

    trace_filename = _trace_filename(eff_config_label, base_policy_name, override_key, seed)
    trace_path = results_dir / trace_filename

    if trace_path.exists() and not force:
        # Hit: load the trace, rebuild the report.
        log.info("Loading cached trace: %s", trace_filename)
        report = _load_report_from_trace(trace_path)
        return SweepRecord(
            override_label=override_label,
            override_key=override_key,
            seed=seed,
            config_label=eff_config_label,
            policy_name=base_policy_name,
            trace_path=trace_path,
            report=report,
            skipped_existing=True,
        )

    # Miss: run and save.
    log.info("Running [%s, seed=%d] -> %s", override_label, seed, trace_filename)
    result = run_simulation(
        walker=walker,
        feasibility_params=feasibility_params,
        policy_name=base_policy_name,
        policy_params=policy_params,
        n_epochs=n_epochs,
        dt_s=dt_s,
        seed=seed,
        config_label=eff_config_label,
    )

    # alpha_0 cache: keyed on the constellation label so each unique
    # constellation in the sweep computes alpha_0 only once.
    precomputed: Optional[tuple[float, int]] = None
    if alpha_0_cache is not None:
        cache_key = walker.short_label
        if cache_key not in alpha_0_cache:
            # Compute the feasibility tensor again (cached on disk; cheap)
            # so we can call build_report.
            feas = _build_feasibility_for(walker, feasibility_params, n_epochs, dt_s)
            alpha_0_cache[cache_key] = compute_alpha_0(feas, T_0=policy_params.T)
        precomputed = alpha_0_cache[cache_key]
        feas = _build_feasibility_for(walker, feasibility_params, n_epochs, dt_s)
    else:
        feas = _build_feasibility_for(walker, feasibility_params, n_epochs, dt_s)

    report = build_report(result, feas, T_0=policy_params.T, precomputed_alpha_0=precomputed)

    # Save the trace with the report folded into metadata.
    metadata = dict(result.metadata)
    metadata["override"] = {k: (v.short_label if isinstance(v, WalkerDeltaConfig) else v) for k, v in override.items()}
    metadata["override_label"] = override_label
    metadata["report"] = report.to_metadata()
    save_trace(trace_path, arrays=result.to_arrays(), metadata=metadata)

    return SweepRecord(
        override_label=override_label,
        override_key=override_key,
        seed=seed,
        config_label=eff_config_label,
        policy_name=base_policy_name,
        trace_path=trace_path,
        report=report,
        skipped_existing=False,
    )


def _build_feasibility_for(
    walker: WalkerDeltaConfig,
    feasibility_params: FeasibilityParams,
    n_epochs: int,
    dt_s: float,
) -> np.ndarray:
    """Compute (or load-from-cache) the feasibility tensor for a walker.

    Cheaper than re-running propagation if the disk cache is warm.
    """
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * dt_s, dt_s=dt_s)
    positions = propagate_keplerian(elements, times)
    return compute_feasibility(positions, dt_s, feasibility_params)


def _load_report_from_trace(trace_path: Path) -> DiagnosticsReport:
    """Reconstruct a DiagnosticsReport from a previously saved trace."""
    _, manifest = load_trace(trace_path)
    rep_dict = manifest.get("user", {}).get("report")
    if rep_dict is None:
        raise ValueError(f"Trace {trace_path} has no 'report' in its manifest; cannot reconstruct.")
    return DiagnosticsReport(**rep_dict)
