"""Central configuration for the satellite ISL formation experiment.

Every tunable lives here so a simulation can be reconfigured without touching
the package internals. ``run.py`` maps command-line flags onto these fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class SimConfig:
    # --- Constellation geometry (Walker-Delta / Starlink-like shell) ---
    planes: int = 24                  # number of orbital planes (RAAN slots)
    sats_per_plane: int = 66          # satellites per plane (true-anomaly slots)
    altitude_km: float = 550.0        # shell altitude above Earth surface
    inclination_deg: float = 53.0     # orbital inclination

    # --- Link feasibility ---
    isl_range_km: float = 2000.0      # max inter-satellite link range
    fov_angle_deg: float = 60.0       # full angular width of each terminal cone

    # --- Time discretization ---
    epochs: int = 7                   # number of timesteps simulated
    t_end_min: float = 1.0            # propagation horizon (minutes from epoch)

    # --- Game / policy parameters ---
    window: int = 10                  # windowed-union horizon T (epochs)
    alpha: float = 0.5                # slewing-cost weight
    bridge_bonus: float = 1e3         # finite utility for a component-bridging link

    # --- Methods to run (subset of: game, furthest, greedy, mdmd) ---
    methods: list[str] = field(default_factory=lambda: ["game", "furthest"])

    # --- Run bookkeeping ---
    quick: bool = False               # record whether this was a quick run
    save_graphs: bool = False         # pickle realized graphs alongside metrics
    out_dir: str | None = None        # results dir (run.py defaults to a timestamp)

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def isl_range_m(self) -> float:
        """ISL range in meters (the units the geometry code works in)."""
        return self.isl_range_km * 1000.0

    @property
    def semimajor_axis_km(self) -> float:
        """Circular-orbit semimajor axis = Earth radius + altitude."""
        return 6378.137 + self.altitude_km

    @property
    def n_satellites(self) -> int:
        return self.planes * self.sats_per_plane

    @property
    def times(self) -> np.ndarray:
        """Epoch offsets (minutes) at which the constellation is propagated."""
        return np.linspace(0.0, self.t_end_min, self.epochs)

    def to_dict(self) -> dict:
        """JSON-serializable snapshot of the configuration (for results/)."""
        d = asdict(self)
        d["n_satellites"] = self.n_satellites
        return d
