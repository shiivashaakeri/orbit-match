# orbit-match/scripts/load_configs.py
# Run: python -m scripts.load_configs

"""Parse every config in configs/ and print the resolved values.

Sanity check that the seven YAML files round-trip cleanly into the
existing dataclasses (WalkerDeltaConfig, FeasibilityParams, PolicyParams)
and that the `T: orbital_period` sentinel resolves to the expected
number of epochs.

Run this before doing anything else with the configs. If it fails, the
problem is in the config files, not in any downstream code.

No network, no caching, no simulations -- just parse and validate.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
from orbitmatch.feasibility.predicates import FeasibilityParams
from orbitmatch.policy.base import PolicyParams
from orbitmatch.utils.io import PROJECT_ROOT
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)

CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a Python dict."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping, got {type(data).__name__}.")
    return data


def build_walker(cfg: dict[str, Any]) -> WalkerDeltaConfig:
    """Construct a WalkerDeltaConfig from the `walker:` block."""
    return WalkerDeltaConfig(
        M=int(cfg["M"]),
        P=int(cfg["P"]),
        F=int(cfg["F"]),
        altitude_km=float(cfg["altitude_km"]),
        inclination_deg=float(cfg["inclination_deg"]),
        name=str(cfg.get("name", "")),
    )


def build_feasibility(cfg: dict[str, Any]) -> FeasibilityParams:
    """Construct FeasibilityParams. Converts deg/s -> rad/s if needed."""
    if "rate_max_deg_per_s" in cfg and "rate_max_rad_per_s" in cfg:
        raise ValueError("Specify exactly one of rate_max_deg_per_s or rate_max_rad_per_s, not both.")
    if "rate_max_deg_per_s" in cfg:
        rate_rad_s = float(np.deg2rad(cfg["rate_max_deg_per_s"]))
    elif "rate_max_rad_per_s" in cfg:
        rate_rad_s = float(cfg["rate_max_rad_per_s"])
    else:
        rate_rad_s = float(np.deg2rad(1.0))  # default 1 deg/s
    return FeasibilityParams(
        atm_buffer_km=float(cfg.get("atm_buffer_km", 80.0)),
        range_max_km=float(cfg.get("range_max_km", 8000.0)),
        rate_max_rad_per_s=rate_rad_s,
    )


def resolve_T(T_spec: Any, walker: WalkerDeltaConfig, dt_s: float) -> int:
    """Resolve the certificate window T to an integer number of epochs.

    Accepts either an int (returned as-is) or the string sentinel
    "orbital_period", which is replaced by ceil(T_orb_s / dt_s).
    """
    if isinstance(T_spec, int):
        return T_spec
    if isinstance(T_spec, str) and T_spec == "orbital_period":
        return math.ceil(walker.orbital_period_s / dt_s)
    raise ValueError(f"Invalid T spec: {T_spec!r}. Use an integer or 'orbital_period'.")


def build_policy_params(cfg: dict[str, Any], T_epochs: int) -> PolicyParams:
    """Construct PolicyParams from the `policy:` block plus a resolved T."""
    return PolicyParams(
        H=int(cfg["H"]),
        T=T_epochs,
        switching_cost_scale=float(cfg["switching_cost_scale"]),
        epsilon_geometric_prior=float(cfg["epsilon_geometric_prior"]),
        tie_break=str(cfg["tie_break"]),
        seed=cfg.get("seed"),
    )


def n_epochs_for(walker: WalkerDeltaConfig, duration_periods: float, dt_s: float) -> int:
    """Compute n_epochs = ceil(duration_periods * T_orb_s / dt_s)."""
    return math.ceil(duration_periods * walker.orbital_period_s / dt_s)


# ---------------------------------------------------------------------------
# Per-file validators
# ---------------------------------------------------------------------------


def check_constellation(path: Path) -> WalkerDeltaConfig:
    cfg = load_yaml(path)
    walker = build_walker(cfg["walker"])
    feas = build_feasibility(cfg["feasibility"])
    print(f"\n[{path.name}]")
    print(f"  label:      {cfg.get('label', '(missing)')}")
    print(
        f"  walker:     M={walker.M}, P={walker.P}, F={walker.F}, alt={walker.altitude_km:.0f} km, inc={walker.inclination_deg:.0f} deg"  # noqa: E501
    )
    print(f"  period:     {walker.orbital_period_s:.1f} s ({walker.orbital_period_s / 60:.2f} min)")
    print(f"  T_orb @ dt=10s: {math.ceil(walker.orbital_period_s / 10.0)} epochs")
    print(
        f"  feasibility: atm={feas.atm_buffer_km:.0f}, range={feas.range_max_km:.0f}, rate={np.rad2deg(feas.rate_max_rad_per_s):.2f} deg/s"  # noqa: E501
    )
    return walker


def check_policy(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    # Build with a placeholder T so dataclass validation runs end-to-end.
    params = build_policy_params(cfg["policy"], T_epochs=30)
    print(f"\n[{path.name}]")
    print(f"  name:    {cfg.get('name', '(missing)')}")
    print(
        f"  policy:  H={params.H}, c={params.switching_cost_scale}, eps={params.epsilon_geometric_prior}, tie_break={params.tie_break}"  # noqa: E501
    )
    print("  (T not set here -- supplied per experiment)")
    return cfg


def check_experiment(path: Path) -> None:
    cfg = load_yaml(path)
    constellation_path = CONFIGS_DIR / cfg["constellation"]
    policy_path = CONFIGS_DIR / cfg["policy"]
    walker_cfg = load_yaml(constellation_path)["walker"]
    walker = build_walker(walker_cfg)
    policy_block = load_yaml(policy_path)["policy"]
    sim = cfg["simulation"]
    T_epochs = resolve_T(sim["T"], walker, float(sim["dt_s"]))
    n_epochs = n_epochs_for(walker, float(sim["duration_periods"]), float(sim["dt_s"]))
    params = build_policy_params(policy_block, T_epochs)
    print(f"\n[{path.name}]")
    print(f"  constellation: {cfg['constellation']} -> {walker.short_label}")
    print(f"  policy file:   {cfg['policy']}")
    print(f"  policies run:  {cfg.get('policies', [])}")
    print(f"  duration:      {sim['duration_periods']} periods = {n_epochs} epochs at dt={sim['dt_s']}s")
    print(
        f"  T (resolved):  {T_epochs} epochs ({T_epochs * sim['dt_s']:.0f} s = {T_epochs * sim['dt_s'] / 60:.2f} min)"
    )
    print(f"  seeds:         {sim['seeds']}")
    print(f"  full params:   {asdict(params)}")


def check_sweep_horizon_or_T(path: Path) -> None:
    cfg = load_yaml(path)
    base = cfg["base"]
    walker_cfg = load_yaml(CONFIGS_DIR / base["constellation"])["walker"]
    walker = build_walker(walker_cfg)
    sim = base["simulation"]
    dt_s = float(sim["dt_s"])
    swept_keys = list(cfg["sweep"].keys())
    print(f"\n[{path.name}]")
    print(f"  base constellation: {base['constellation']}")
    print(f"  base policy file:   {base['policy']}")
    print(f"  policy name:        {base['policy_name']}")
    print(f"  duration:           {sim['duration_periods']} periods")
    if "T" in sim:
        T_epochs = resolve_T(sim["T"], walker, dt_s)
        print(f"  T (fixed):          {T_epochs} epochs ({sim['T']!r})")
    else:
        print("  T:                  swept (no fixed value in base)")
    print(f"  seeds:              {sim['seeds']}")
    print(f"  swept axes:         {swept_keys}")
    for k, vs in cfg["sweep"].items():
        print(f"    {k}: {vs}  ({len(vs)} values)")
    print(f"  total runs (seeds * grid): {len(sim['seeds']) * sum(len(v) for v in cfg['sweep'].values())}")


def check_sweep_scaling(path: Path) -> None:
    cfg = load_yaml(path)
    policy_block = load_yaml(CONFIGS_DIR / cfg["policy"])["policy"]
    feas = build_feasibility(cfg["feasibility"])
    sim = cfg["simulation"]
    dt_s = float(sim["dt_s"])
    print(f"\n[{path.name}]")
    print(f"  policy file:    {cfg['policy']}")
    print(f"  policies run:   {cfg['policies']}")
    print(
        f"  feasibility:    atm={feas.atm_buffer_km:.0f}, range={feas.range_max_km:.0f}, rate={np.rad2deg(feas.rate_max_rad_per_s):.2f} deg/s"  # noqa: E501
    )
    print(f"  duration:       {sim['duration_periods']} periods at dt={dt_s}s")
    print(f"  seeds:          {sim['seeds']}")
    print("  constellations:")
    for entry in cfg["constellations"]:
        w = build_walker(entry)
        T_epochs = resolve_T(sim["T"], w, dt_s)
        n_eps = n_epochs_for(w, float(sim["duration_periods"]), dt_s)
        print(
            f"    n={w.M:3d}: P={w.P}, F={w.F}, period={w.orbital_period_s / 60:.2f} min, T={T_epochs} epochs, sim={n_eps} epochs"  # noqa: E501
        )
    # Build a fake PolicyParams with T from the first constellation to validate the policy block.
    first_T = resolve_T(sim["T"], build_walker(cfg["constellations"][0]), dt_s)
    params = build_policy_params(policy_block, first_T)
    print(f"  policy params:  {asdict(params)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    configure(level="WARNING")  # quiet the orbitmatch loggers; we're printing our own.

    print(f"Loading configs from {CONFIGS_DIR}")
    if not CONFIGS_DIR.exists():
        print(f"ERROR: configs directory does not exist: {CONFIGS_DIR}", file=sys.stderr)
        return 1

    expected = [
        "constellation_small.yaml",
        "constellation_medium.yaml",
        "policy_default.yaml",
        "experiment_main.yaml",
        "sweep_horizon.yaml",
        "sweep_T.yaml",
        "sweep_scaling.yaml",
    ]
    missing = [n for n in expected if not (CONFIGS_DIR / n).exists()]
    if missing:
        print(f"ERROR: missing configs: {missing}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print("CONSTELLATIONS")
    print("=" * 72)
    check_constellation(CONFIGS_DIR / "constellation_small.yaml")
    check_constellation(CONFIGS_DIR / "constellation_medium.yaml")

    print("\n" + "=" * 72)
    print("POLICY")
    print("=" * 72)
    check_policy(CONFIGS_DIR / "policy_default.yaml")

    print("\n" + "=" * 72)
    print("EXPERIMENT")
    print("=" * 72)
    check_experiment(CONFIGS_DIR / "experiment_main.yaml")

    print("\n" + "=" * 72)
    print("SWEEPS")
    print("=" * 72)
    check_sweep_horizon_or_T(CONFIGS_DIR / "sweep_horizon.yaml")
    check_sweep_horizon_or_T(CONFIGS_DIR / "sweep_T.yaml")
    check_sweep_scaling(CONFIGS_DIR / "sweep_scaling.yaml")

    print("\n" + "=" * 72)
    print("All 7 configs parsed and validated successfully.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
