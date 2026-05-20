# orbit-match/orbitmatch/plotting/theme.py
# Run: imported by plotting modules; not a runnable script.

"""Visual theme for all figures.

Centralizes the earthy color palette, typography, and matplotlib defaults.
Plotting code outside this module never sets colors, fonts, or rcParams
directly — it pulls them from here.

Two entry points:

- :func:`apply_theme` sets the matplotlib rcParams globally; call it once
  per script before any plotting.
- :data:`COLORS` and :data:`POLICY_COLORS` provide the color constants.

Usage
-----
::

    from orbitmatch.plotting.theme import apply_theme, POLICY_COLORS

    apply_theme()

    fig, ax = plt.subplots()
    ax.plot(t, lambda2, color=POLICY_COLORS["predictive"], label="Predictive matching")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt

from orbitmatch.utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """Earthy color palette used across all figures.

    Hex values are fixed and not to be edited mid-project. If a new role
    is needed, add a new field; do not repurpose an existing one.
    """

    # Categorical colors (for distinguishing policies, configs, etc.)
    burgundy: str = "#7A2922"
    copper: str = "#B87333"
    olive: str = "#5C6B4A"
    warmbrown: str = "#7B5C3E"

    # Neutrals
    parchment: str = "#D4C4A8"  # background, light fills
    near_black: str = "#2C2C2A"  # body text, primary axis
    warm_gray: str = "#9C9A92"  # gridlines, secondary text

    # Semantic (used sparingly, only when domain meaning is clear)
    deep_red: str = "#5A1A14"  # error / lower bound
    forest: str = "#3D4F38"  # success / theorem guarantee

    def cycle(self) -> list[str]:
        """Default categorical cycle for non-policy plots."""
        return [self.burgundy, self.copper, self.olive, self.warmbrown]


COLORS = Palette()


# Mapping from policy name (must match the strings used in policy modules)
# to color. Used in every plot that compares policies.
POLICY_COLORS: dict[str, str] = {
    "predictive": COLORS.burgundy,
    "equilibrium": COLORS.copper,
    "greedy": COLORS.olive,
    "random": COLORS.warmbrown,
}

# Policy linestyles. The theorem guarantee line uses a dashed style.
POLICY_LINESTYLES: dict[str, str] = {
    "predictive": "-",
    "equilibrium": "-",
    "greedy": "--",
    "random": ":",
}

# Marker symbols for sparse-point plots (scaling, ablations).
POLICY_MARKERS: dict[str, str] = {
    "predictive": "o",
    "equilibrium": "s",
    "greedy": "^",
    "random": "D",
}


# ---------------------------------------------------------------------------
# Figure dimensions (IEEE conference column widths)
# ---------------------------------------------------------------------------


FIG_WIDTH_SINGLE_COL = 3.5  # inches; standard IEEE single column
FIG_WIDTH_DOUBLE_COL = 7.16  # inches; standard IEEE double column (page width)
FIG_HEIGHT_DEFAULT = 2.3  # inches; readable aspect for single-column plots


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------


def apply_theme(
    context: Literal["paper", "diagnostic"] = "paper",
    use_latex: bool = False,
) -> None:
    """Apply project-wide matplotlib rcParams.

    Call this once at the top of any script that produces plots, before
    creating any figure.

    Parameters
    ----------
    context
        ``"paper"`` for publication-quality figures (small fonts, tight
        layout). ``"diagnostic"`` for exploratory plots (larger fonts,
        less aggressive whitespace).
    use_latex
        If True, render text with LaTeX. Adds compile-time but produces
        nicer math. Requires a working LaTeX installation on the system.
        Defaults to False for portability.
    """
    base = {
        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Axes
        "axes.edgecolor": COLORS.near_black,
        "axes.labelcolor": COLORS.near_black,
        "axes.titlecolor": COLORS.near_black,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        # Grid
        "grid.color": COLORS.warm_gray,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        # Ticks
        "xtick.color": COLORS.near_black,
        "ytick.color": COLORS.near_black,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.5,
        # Legend
        "legend.frameon": False,
        "legend.labelcolor": COLORS.near_black,
        "legend.handlelength": 1.6,
        "legend.borderpad": 0.2,
        "legend.columnspacing": 1.2,
        # Color cycle
        "axes.prop_cycle": mpl.cycler(color=COLORS.cycle()),
        # Font
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "text.color": COLORS.near_black,
    }

    if context == "paper":
        base.update(
            {
                "font.size": 9,
                "axes.labelsize": 9,
                "axes.titlesize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "figure.figsize": (FIG_WIDTH_SINGLE_COL, FIG_HEIGHT_DEFAULT),
            }
        )
    elif context == "diagnostic":
        base.update(
            {
                "font.size": 11,
                "axes.labelsize": 11,
                "axes.titlesize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.figsize": (6.0, 4.0),
            }
        )
    else:
        raise ValueError(f"Unknown context: {context!r}. Use 'paper' or 'diagnostic'.")

    if use_latex:
        base.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )

    plt.rcParams.update(base)
    log.info("Applied %s theme (latex=%s)", context, use_latex)


def policy_style(policy: str) -> dict[str, str]:
    """Return a dict of plot kwargs (color, linestyle, marker) for a policy.

    Convenient shorthand for ``ax.plot(..., **policy_style("predictive"))``.

    Raises
    ------
    KeyError
        If ``policy`` is not one of the recognized policy names.
    """
    if policy not in POLICY_COLORS:
        raise KeyError(f"Unknown policy {policy!r}. Known: {sorted(POLICY_COLORS.keys())}")
    return {
        "color": POLICY_COLORS[policy],
        "linestyle": POLICY_LINESTYLES[policy],
        "marker": POLICY_MARKERS[policy],
    }
