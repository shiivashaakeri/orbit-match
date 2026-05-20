# orbit-match/orbitmatch/constellation/walker_delta.py
# Run: imported by other modules; not a runnable script.

"""Walker--Delta constellation geometry.

Builds the initial orbital elements for a Walker--Delta constellation in
the standard ``i:M/P/F`` notation:

- ``M`` total satellites, distributed evenly across
- ``P`` orbital planes (so ``M/P`` satellites per plane), with
- ``F`` the phasing parameter relating the inter-plane phase offset.

All planes share the same inclination ``i`` and circular altitude. Planes
are separated in right ascension of ascending node (RAAN) by ``360/P``
degrees. Within a plane, satellites are separated by ``360 P / M``
degrees in true anomaly. Across planes, satellite ``k`` in plane ``p+1``
leads satellite ``k`` in plane ``p`` by an additional ``360 F / M``
degrees of true anomaly.

Reference
---------
Walker, J. G. "Satellite constellations." *Journal of the British
Interplanetary Society*, 37:559--572, 1984.

Usage
-----
::

    from orbitmatch.constellation.walker_delta import WalkerDeltaConfig

    config = WalkerDeltaConfig(M=60, P=6, F=2, altitude_km=550.0, inclination_deg=53.0)
    elements = config.initial_elements()    # one row per satellite
    print(elements.shape)                   # (60, 6) -- (a, e, i, RAAN, omega, nu)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)

# Earth gravitational parameter (km^3 / s^2) and mean equatorial radius (km)
MU_EARTH = 398600.4418
R_EARTH = 6378.137


@dataclass(frozen=True)
class WalkerDeltaConfig:
    """Parameters defining a Walker--Delta constellation.

    Parameters
    ----------
    M
        Total number of satellites.
    P
        Number of orbital planes. Must divide ``M`` evenly.
    F
        Phasing parameter, integer in ``0, 1, ..., P-1``.
    altitude_km
        Circular orbit altitude above the Earth's equator (km).
    inclination_deg
        Orbit inclination, in degrees.
    name
        Optional human-readable label, used in logs and filenames.
    """

    M: int
    P: int
    F: int
    altitude_km: float
    inclination_deg: float
    name: str = field(default="")

    def __post_init__(self) -> None:
        if self.M <= 0:
            raise ValueError(f"M must be positive; got {self.M}.")
        if self.P <= 0 or self.P > self.M:
            raise ValueError(f"P must be in 1..M; got P={self.P}, M={self.M}.")
        if self.M % self.P != 0:
            raise ValueError(f"P must divide M; got M={self.M}, P={self.P}.")
        if not (0 <= self.F < self.P):
            raise ValueError(f"F must be in 0..P-1; got F={self.F}, P={self.P}.")
        if self.altitude_km <= 0:
            raise ValueError(f"altitude_km must be positive; got {self.altitude_km}.")
        if not (0.0 <= self.inclination_deg <= 180.0):
            raise ValueError(f"inclination_deg must be in [0, 180]; got {self.inclination_deg}.")

    @property
    def n(self) -> int:
        """Alias for ``M`` (the number of satellites)."""
        return self.M

    @property
    def sats_per_plane(self) -> int:
        """Number of satellites per orbital plane."""
        return self.M // self.P

    @property
    def semi_major_axis_km(self) -> float:
        """Semi-major axis of the circular orbits (km)."""
        return R_EARTH + self.altitude_km

    @property
    def orbital_period_s(self) -> float:
        """Kepler orbital period (seconds) for a circular orbit at the given altitude."""
        a = self.semi_major_axis_km
        return float(2.0 * np.pi * np.sqrt(a**3 / MU_EARTH))

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion (rad/s) for the circular orbit."""
        return float(np.sqrt(MU_EARTH / self.semi_major_axis_km**3))

    @property
    def short_label(self) -> str:
        """Compact identifier suitable for filenames, e.g. ``walker_60_6_2``."""
        base = "walker"
        if self.name:
            base = f"{base}_{self.name}"
        return f"{base}_{self.M}_{self.P}_{self.F}_alt{int(self.altitude_km)}_inc{int(self.inclination_deg)}"

    def initial_elements(self) -> np.ndarray:
        """Return the initial Keplerian orbital elements for all satellites.

        For each satellite ``k`` in plane ``p`` (0-indexed), the elements
        are computed as:

        - semi-major axis ``a`` (km), shared
        - eccentricity ``e = 0`` (circular)
        - inclination ``i`` (deg), shared
        - RAAN ``Omega = p * 360 / P`` (deg)
        - argument of periapsis ``omega = 0`` (deg; undefined for circular,
          conventionally zero)
        - true anomaly ``nu = k * 360 P / M + p * 360 F / M`` (deg)

        Returns
        -------
        numpy.ndarray
            Array of shape ``(M, 6)``, dtype ``float64``. Columns are
            ``(a, e, i, RAAN, omega, nu)`` in (km, [], deg, deg, deg, deg).
            Satellites are indexed plane-major: rows ``0..M/P-1`` belong
            to plane 0, rows ``M/P..2M/P-1`` to plane 1, and so on.
        """
        M, P, F = self.M, self.P, self.F
        S = self.sats_per_plane  # satellites per plane
        a = self.semi_major_axis_km
        inc = self.inclination_deg

        elements = np.zeros((M, 6), dtype=np.float64)

        for p in range(P):
            raan = (360.0 * p) / P
            for k in range(S):
                idx = p * S + k
                nu = (360.0 * P * k) / M + (360.0 * F * p) / M
                nu = nu % 360.0

                elements[idx, 0] = a  # semi-major axis
                elements[idx, 1] = 0.0  # eccentricity
                elements[idx, 2] = inc  # inclination
                elements[idx, 3] = raan  # RAAN
                elements[idx, 4] = 0.0  # argument of periapsis
                elements[idx, 5] = nu  # true anomaly

        log.info(
            "Built Walker-Delta i:%d/%d/%d at alt=%.0f km, inc=%.1f deg (period %.1f min)",
            M,
            P,
            F,
            self.altitude_km,
            inc,
            self.orbital_period_s / 60.0,
        )
        return elements

    def plane_index(self, sat_index: int) -> int:
        """Return the plane index ``p`` for a given satellite index.

        Useful for grouping satellites by plane in diagnostic plots.
        """
        if not (0 <= sat_index < self.M):
            raise IndexError(f"sat_index {sat_index} out of range [0, {self.M}).")
        return sat_index // self.sats_per_plane

    def slot_index(self, sat_index: int) -> int:
        """Return the slot index ``k`` within the plane for a given satellite."""
        if not (0 <= sat_index < self.M):
            raise IndexError(f"sat_index {sat_index} out of range [0, {self.M}).")
        return sat_index % self.sats_per_plane
