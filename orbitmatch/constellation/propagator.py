# orbit-match/orbitmatch/constellation/propagator.py
# Run: imported by other modules; not a runnable script.

"""Orbital propagation: initial elements -> ECI positions over time.

For the experiments in this paper we use Keplerian (two-body) propagation
of circular orbits. At 550 km altitude and over a 3-orbit horizon, J2 and
other perturbations introduce drift on the order of tens of meters per
satellite, which is negligible for feasibility computations evaluated on
the km scale (range threshold 5000 km, line-of-sight to a 6378 km Earth).

The :func:`propagate_keplerian` function is fully vectorized over time and
satellite indices: it takes the ``(M, 6)`` element array from
:class:`WalkerDeltaConfig.initial_elements` plus a 1-D time grid and
returns a ``(n_epochs, M, 3)`` array of ECI position vectors in km.

Frame
-----
The output is in an Earth-centered inertial (ECI) frame aligned with the
J2000 equinox. For Walker--Delta with planes spaced in RAAN, this means
the X-axis lies in the equatorial plane in the direction of the
ascending node of plane 0, and Z points along Earth's rotation axis.

Usage
-----
::

    from orbitmatch.constellation.walker_delta import WalkerDeltaConfig
    from orbitmatch.constellation.propagator import propagate_keplerian
    import numpy as np

    config = WalkerDeltaConfig(M=60, P=6, F=2, altitude_km=550.0, inclination_deg=53.0)
    elements = config.initial_elements()

    times_s = np.arange(0, 3 * config.orbital_period_s, 10.0)
    positions = propagate_keplerian(elements, times_s)
    # positions.shape == (len(times_s), 60, 3)
"""

from __future__ import annotations

import numpy as np

from orbitmatch.constellation.walker_delta import MU_EARTH
from orbitmatch.utils.logging_setup import get_logger
from orbitmatch.utils.timing import timed

log = get_logger(__name__)


def propagate_keplerian(
    elements: np.ndarray,
    times_s: np.ndarray,
) -> np.ndarray:
    """Propagate Keplerian elements to ECI positions over a time grid.

    Implements two-body propagation of circular orbits. Eccentricity is
    assumed zero (any nonzero entries in the eccentricity column are
    silently ignored with a warning). All satellites are propagated
    simultaneously via vectorized NumPy operations.

    Parameters
    ----------
    elements
        Array of shape ``(M, 6)``, dtype float, with columns
        ``(a, e, i, RAAN, omega, nu_0)``:

        - ``a``: semi-major axis (km)
        - ``e``: eccentricity (must be zero; circular orbits only)
        - ``i``: inclination (degrees)
        - ``RAAN``: right ascension of ascending node (degrees)
        - ``omega``: argument of periapsis (degrees, ignored for circular)
        - ``nu_0``: initial true anomaly at ``t = 0`` (degrees)

    times_s
        1-D array of times in seconds at which to evaluate positions.
        Times need not be uniformly spaced.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(times_s), M, 3)``, dtype float64, of ECI
        position vectors in km.
    """
    if elements.ndim != 2 or elements.shape[1] != 6:
        raise ValueError(f"elements must have shape (M, 6); got {elements.shape}.")
    if times_s.ndim != 1:
        raise ValueError(f"times_s must be 1-D; got shape {times_s.shape}.")

    M = elements.shape[0]
    n_epochs = times_s.shape[0]

    a = elements[:, 0]  # (M,)
    e = elements[:, 1]  # (M,)
    inc = np.deg2rad(elements[:, 2])  # (M,)
    raan = np.deg2rad(elements[:, 3])  # (M,)
    nu_0 = np.deg2rad(elements[:, 5])  # (M,)

    if np.any(e > 1e-9):
        log.warning(
            "Non-circular orbits detected (max e = %.3e); using circular approximation.",
            float(np.max(e)),
        )

    # Mean motion (rad/s), per satellite.
    n_mean = np.sqrt(MU_EARTH / a**3)  # (M,)

    # True anomaly at each epoch. For a circular orbit, true anomaly
    # equals mean anomaly equals eccentric anomaly, so:
    #   nu(t) = nu_0 + n * t
    with timed("propagate_keplerian", level="DEBUG"):
        # Shape: (n_epochs, M)
        nu_t = nu_0[None, :] + n_mean[None, :] * times_s[:, None]

        # Position in the orbital (perifocal) frame: x along ascending direction,
        # y rotated 90 deg in plane of orbit, z = 0.
        # For circular, r = a, so:
        x_pf = a[None, :] * np.cos(nu_t)  # (n_epochs, M)
        y_pf = a[None, :] * np.sin(nu_t)  # (n_epochs, M)

        # Rotate from perifocal to ECI using the rotation matrix
        #   R = R_z(RAAN) @ R_x(inclination) @ R_z(omega)
        # For circular orbits omega is conventionally zero, so the perifocal
        # x-axis points to the ascending node. We apply R_x then R_z:
        cos_i = np.cos(inc)  # (M,)
        sin_i = np.sin(inc)  # (M,)
        cos_raan = np.cos(raan)  # (M,)
        sin_raan = np.sin(raan)  # (M,)

        # After R_x(i): (x, y_pf * cos_i, y_pf * sin_i)
        # After R_z(RAAN): rotate (x, y_after_i) in the x-y plane.
        # Combine:
        #   X_eci = x_pf * cos_raan - (y_pf * cos_i) * sin_raan
        #   Y_eci = x_pf * sin_raan + (y_pf * cos_i) * cos_raan
        #   Z_eci = y_pf * sin_i
        # All broadcasts: (n_epochs, M)
        X_eci = x_pf * cos_raan[None, :] - (y_pf * cos_i[None, :]) * sin_raan[None, :]
        Y_eci = x_pf * sin_raan[None, :] + (y_pf * cos_i[None, :]) * cos_raan[None, :]
        Z_eci = y_pf * sin_i[None, :]

    positions = np.stack([X_eci, Y_eci, Z_eci], axis=-1)  # (n_epochs, M, 3)

    log.info(
        "Propagated %d satellites over %d epochs (Keplerian, circular).",
        M,
        n_epochs,
    )
    return positions


def make_time_grid(
    duration_s: float,
    dt_s: float,
    *,
    inclusive_end: bool = False,
) -> np.ndarray:
    """Construct a uniform time grid for a simulation.

    Parameters
    ----------
    duration_s
        Total simulation duration in seconds.
    dt_s
        Epoch length in seconds.
    inclusive_end
        If True, include ``duration_s`` as the last point; if False
        (default), stop just before it.

    Returns
    -------
    numpy.ndarray
        1-D array of times in seconds, starting at 0.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive; got {duration_s}.")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive; got {dt_s}.")

    n = int(np.floor(duration_s / dt_s)) + 1 if inclusive_end else int(np.floor(duration_s / dt_s))

    times = np.arange(n, dtype=np.float64) * dt_s
    log.info(
        "Built time grid: %d epochs, dt=%.2f s, duration=%.2f s",
        n,
        dt_s,
        times[-1] if n > 0 else 0.0,
    )
    return times
