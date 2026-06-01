# orbit-match/orbitmatch/utils/io.py
# Run: imported by other modules; not a runnable script.

"""File persistence for experiments.

All on-disk artifacts in the project flow through this module. We use two
formats:

- ``.npz``  for array-heavy data (trajectories, feasibility tensors, matchings)
- ``.parquet`` for tabular results (sweeps, summaries, tables)

We do not use pickle anywhere: it breaks across Python versions, is unsafe to
load from untrusted sources, and silently embeds class definitions.

Each saved file carries a *manifest*: a JSON-serializable dict of metadata
(timestamp, git commit if available, package version, parameter dict) so that
months from now we can answer "what produced this file?" without re-deriving
it from the filename.

Naming convention
-----------------
Files are named ``{prefix}_{config}_{policy}_{params}_{seed}.{ext}`` where
``{params}`` is a short hash of the parameter dict. The full parameter dict
lives in the manifest, not the filename — filenames stay readable.

Usage
-----
::

    from orbitmatch.utils.io import save_trace, load_trace, results_dir

    save_trace(
        path=results_dir("lambda2_traces") / "medium_predictive_seed42.npz",
        arrays={"lambda2": lambda2_trace, "matchings": matching_history},
        metadata={"policy": "predictive", "config": "medium", "seed": 42, "H": 10},
    )

    arrays, metadata = load_trace(path)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project root and standard directories
# ---------------------------------------------------------------------------

# This file lives at orbit-match/orbitmatch/utils/io.py
# Project root is three parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_TRACES = PROJECT_ROOT / "data" / "traces"
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_ROOT = PROJECT_ROOT / "figures"
TABLES_ROOT = PROJECT_ROOT / "tables"


def project_root() -> Path:
    """Return the absolute path of the project root."""
    return PROJECT_ROOT


def results_dir(experiment: str) -> Path:
    """Return ``results/{experiment}/``, creating it if needed.

    Parameters
    ----------
    experiment
        One of ``"lambda2_traces"``, ``"horizon_ablation"``, ``"scaling"``,
        ``"robustness"``, ``"diagnostics"``.
    """
    path = RESULTS_ROOT / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(kind: str = "paper") -> Path:
    """Return ``figures/{kind}/``, creating it if needed.

    Parameters
    ----------
    kind
        Either ``"paper"`` or ``"diagnostics"``.
    """
    path = FIGURES_ROOT / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def tables_dir() -> Path:
    """Return ``tables/``, creating it if needed."""
    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    return TABLES_ROOT


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    """Best-effort lookup of the current git commit hash. Returns None if unknown."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _package_version() -> str:
    """Return the installed orbitmatch version, or '0.0.0+unknown'."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("orbitmatch")
    except Exception:  # pragma: no cover
        return "0.0.0+unknown"


def _build_manifest(user_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap user metadata with provenance fields."""
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "orbitmatch_version": _package_version(),
        "user": dict(user_metadata),
    }
    return manifest


def params_hash(params: Mapping[str, Any], length: int = 8) -> str:
    """Compute a short, stable hash of a parameter dict.

    Useful for cache filenames so that distinct parameter sets land in
    distinct files. Hash is deterministic across runs: the same dict
    (after JSON-canonical serialization) always yields the same string.

    Parameters
    ----------
    params
        Any JSON-serializable mapping.
    length
        Number of hex characters to keep (default 8).
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]


# ---------------------------------------------------------------------------
# Array I/O (.npz)
# ---------------------------------------------------------------------------


def save_trace(
    path: str | os.PathLike,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    compress: bool = True,
) -> Path:
    """Save arrays and a JSON manifest to an ``.npz`` file.

    The manifest is stored alongside the arrays under the reserved key
    ``"__manifest__"``: a single-element object-dtype array holding a
    JSON string of the metadata wrapped with provenance fields.

    Parameters
    ----------
    path
        Destination path. Should end in ``.npz``.
    arrays
        Mapping of array name -> ``numpy.ndarray``. Names must not collide
        with the reserved key ``"__manifest__"``.
    metadata
        User metadata (parameters, config name, seed, etc.). Must be
        JSON-serializable.
    compress
        Use ``np.savez_compressed`` if True (default), ``np.savez`` otherwise.

    Returns
    -------
    pathlib.Path
        The path the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "__manifest__" in arrays:
        raise ValueError("'__manifest__' is a reserved array name and cannot be used.")

    manifest = _build_manifest(metadata)
    manifest_json = np.array(json.dumps(manifest), dtype=object)

    payload = dict(arrays)
    payload["__manifest__"] = manifest_json

    saver = np.savez_compressed if compress else np.savez
    saver(path, **payload)

    log.info("Saved trace to %s (%d arrays, %.1f KB)", path, len(arrays), path.stat().st_size / 1024)
    return path


def load_trace(path: str | os.PathLike) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load arrays and manifest from an ``.npz`` file written by ``save_trace``.

    Parameters
    ----------
    path
        Source path.

    Returns
    -------
    arrays
        Mapping of array name -> ``numpy.ndarray``. The reserved
        ``"__manifest__"`` key is stripped.
    manifest
        Parsed metadata dict, including provenance fields and the
        original user metadata under ``manifest["user"]``.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as f:
        keys = list(f.keys())
        arrays = {k: f[k] for k in keys if k != "__manifest__"}
        if "__manifest__" in keys:
            manifest = json.loads(str(f["__manifest__"]))
        else:
            log.warning("Loaded %s but it has no __manifest__; metadata will be empty.", path)
            manifest = {}
    log.info("Loaded trace from %s (%d arrays)", path, len(arrays))
    return arrays, manifest


# ---------------------------------------------------------------------------
# Tabular I/O (.parquet)
# ---------------------------------------------------------------------------


def save_table(path: str | os.PathLike, df, metadata: Mapping[str, Any] | None = None) -> Path:
    """Save a pandas DataFrame to parquet, with optional metadata sidecar.

    Metadata, if given, is written as a JSON sidecar next to the parquet
    file with the same stem and a ``.manifest.json`` suffix. Parquet
    itself supports table-level metadata, but a sidecar is easier to
    inspect by hand.

    Parameters
    ----------
    path
        Destination path. Should end in ``.parquet``.
    df
        pandas DataFrame.
    metadata
        Optional JSON-serializable metadata.

    Returns
    -------
    pathlib.Path
        The path the parquet file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False)

    if metadata is not None:
        manifest = _build_manifest(metadata)
        sidecar = path.with_suffix(".manifest.json")
        sidecar.write_text(json.dumps(manifest, indent=2))

    log.info("Saved table to %s (%d rows)", path, len(df))
    return path


def load_table(path: str | os.PathLike) -> tuple[Any, dict[str, Any]]:
    """Load a parquet file with optional manifest sidecar.

    Returns
    -------
    df
        pandas DataFrame.
    manifest
        Parsed manifest dict, or ``{}`` if no sidecar was found.
    """
    import pandas as pd  # local import keeps module-level imports cheap  # noqa: PLC0415

    path = Path(path)
    df = pd.read_parquet(path)  # noqa: PD901

    sidecar = path.with_suffix(".manifest.json")
    manifest = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    log.info("Loaded table from %s (%d rows)", path, len(df))
    return df, manifest
