"""Orbital model and SGP4 propagation.

Extracted from the notebook's cell 1. ``spacex_constellation`` is parameterized
so the constellation size/shell can be driven from :class:`config.SimConfig`
(the notebook hard-coded a 24x66 / 550 km / 53 deg Starlink shell).
"""

from __future__ import annotations

import numpy as np
from sgp4.api import Satrec, WGS84

EARTH_RADIUS_KM = 6378.137
MU_EARTH = 398600.4418  # Earth gravitational constant, km^3/s^2


class Satellite:
    """A single satellite described by its six Keplerian orbital elements."""

    def __init__(self, name, a, e, i, omega, arg_peri, true_anom):
        self.name = name
        self.a = a                  # semimajor axis (km)
        self.e = e                  # eccentricity
        self.i = i                  # inclination (deg)
        self.omega = omega          # RAAN (deg)
        self.arg_peri = arg_peri    # argument of pericenter (rad/deg as set)
        self.true_anom = true_anom  # true anomaly (deg)


def spacex_constellation(
    planes: int = 24,
    sats_per_plane: int = 66,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
):
    """Generate a Walker-Delta / Starlink-like shell of circular orbits.

    Planes are spread evenly in RAAN over [0, 360) and satellites evenly in
    true anomaly over [0, 360) within each plane.
    """
    constellation = []
    satellite_id = 1

    a = altitude_km + EARTH_RADIUS_KM
    e = 0
    i = inclination_deg
    omega = np.linspace(0, 360, planes, endpoint=False)
    arg_peri = 0
    true_anom = np.linspace(0, 360, sats_per_plane, endpoint=False)

    for idx in range(len(omega)):
        for anom in true_anom:
            name = f"Satellite-{satellite_id}"
            constellation.append(
                Satellite(name, a, e, i, omega[idx], arg_peri, anom)
            )
            satellite_id += 1
    return constellation


def build_constellation(cfg):
    """Construct the constellation described by a :class:`config.SimConfig`."""
    return spacex_constellation(
        planes=cfg.planes,
        sats_per_plane=cfg.sats_per_plane,
        altitude_km=cfg.altitude_km,
        inclination_deg=cfg.inclination_deg,
    )


def calculate_cartesian_coordinates(constellation_obj, t_offset_minutes):
    """Propagate the constellation with SGP4 to ``t_offset_minutes`` from epoch.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        ECI positions in meters.
    velocities : np.ndarray, shape (N, 3)
        ECI velocities in meters/second.
    """
    num_satellites = len(constellation_obj)
    positions = np.zeros((num_satellites, 3))
    velocities = np.zeros((num_satellites, 3))

    mu = MU_EARTH

    # Arbitrary base epoch (Julian Date) for the simulation.
    jd1 = 2460000.0
    jd2 = 0.0
    target_jd2 = jd2 + (t_offset_minutes / 1440.0)  # minutes -> days

    for i, sat in enumerate(constellation_obj):
        # 1. Degrees -> radians for SGP4.
        inclo = np.radians(sat.i)
        nodeo = np.radians(sat.omega)
        argpo = np.radians(sat.arg_peri)
        nu = np.radians(sat.true_anom)
        ecco = sat.e

        # 2. True anomaly -> mean anomaly.
        if ecco < 1e-8:  # effectively circular (Starlink)
            mo = nu
        else:
            E = 2 * np.arctan(np.sqrt((1 - ecco) / (1 + ecco)) * np.tan(nu / 2))
            mo = E - ecco * np.sin(E)

        # 3. Mean motion (rad/min).
        n_rad_per_sec = np.sqrt(mu / (sat.a ** 3))
        no_kozai = n_rad_per_sec * 60.0

        # 4. Initialize the satellite natively in SGP4.
        rec = Satrec()
        rec.sgp4init(
            WGS84,
            "i",
            i + 1,
            (jd1 + jd2) - 2433281.5,  # SGP4 epoch (days since 1949-12-31)
            0.0,   # bstar
            0.0,   # ndot
            0.0,   # nddot
            ecco,
            argpo,
            inclo,
            mo,
            no_kozai,
            nodeo,
        )

        # 5. Propagate.
        error_code, r, v = rec.sgp4(jd1, target_jd2)

        if error_code == 0:
            positions[i, :] = np.array(r) * 1000.0   # km -> m
            velocities[i, :] = np.array(v) * 1000.0  # km/s -> m/s
        else:
            print(f"SGP4 Propagation error {error_code} for satellite {sat.name}")

    return positions, velocities
