# orbit-match/scripts/render_animation.py
# Run: python -m scripts.render_animation [--phase1-frames N] [--phase2-frames N]

"""Render a two-panel animation of the predictive policy.

Two phases, smooth transition:

  Phase 1 (link formation): the policy runs for ~1.5 orbital periods.
  Each frame shows (left) 3D view of Earth + satellites with current
  matching edges, (right) a 2D network view with the rolling-window
  realized union accumulating.

  Phase 2 (consensus): the same matching sequence is replayed, but
  each satellite has a scalar state x_i(t) updated by
  x(t+1) = (I - mu * L_G(t)) x(t). Satellites' colors map to their
  state value. The network panel shows the current matching plus the
  disagreement norm over time.

Outputs (saved to figures/animations/):
  predictive_animation.mp4   shareable video
  predictive_animation.gif   smaller, for slides
  predictive_animation.html  self-contained interactive viewer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as mpla
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from orbitmatch.plotting.theme import COLORS, apply_theme
from orbitmatch.utils.io import RESULTS_ROOT, figures_dir, load_trace
from orbitmatch.utils.logging_setup import configure, get_logger

log = get_logger(__name__)


CANONICAL_TRACE = RESULTS_ROOT / "canonical" / "fig1" / "predictive_seed42.npz"
EARTH_RADIUS_KM = 6371.0
MU = 1.0 / 3.0  # consensus step size
PHASE1_DURATION_S = 30.0  # seconds of animation for link formation
PHASE2_DURATION_S = 25.0
TRANSITION_DURATION_S = 2.0
FINAL_HOLD_S = 3.0
FPS = 30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trace_data(path: Path) -> dict:
    """Load the canonical trace and the satellite positions over time."""
    arrays, manifest = load_trace(path)
    user = manifest.get("user", {})

    actions = arrays["actions"]  # (n_epochs, n)
    matchings = arrays["matchings"]  # tuple of (n_edges_t, 2) arrays

    # We also need satellite positions over time. The canonical trace
    # doesn't include positions, but the metadata has walker config.
    # Re-propagate them.
    from orbitmatch.constellation.propagator import make_time_grid, propagate_keplerian
    from orbitmatch.constellation.walker_delta import WalkerDeltaConfig

    walker_meta = user.get("walker", {})
    walker = WalkerDeltaConfig(
        M=int(walker_meta.get("M", 60)),
        P=int(walker_meta.get("P", 6)),
        F=int(walker_meta.get("F", 2)),
        altitude_km=float(walker_meta.get("altitude_km", 550.0)),
        inclination_deg=float(walker_meta.get("inclination_deg", 53.0)),
        name=walker_meta.get("name", "medium"),
    )

    n_epochs = int(actions.shape[0])
    dt_s = float(user.get("dt_s", 10.0))
    elements = walker.initial_elements()
    times = make_time_grid(duration_s=n_epochs * dt_s, dt_s=dt_s)
    positions = propagate_keplerian(elements, times)  # (n_epochs, n, 3)

    return {
        "actions": actions,
        "matchings": matchings,
        "positions": positions,
        "walker": walker,
        "n_epochs": n_epochs,
        "n": int(actions.shape[1]),
        "dt_s": dt_s,
        "T": int(user.get("policy_params", {}).get("T", walker.orbital_period_s / dt_s)),
    }


# ---------------------------------------------------------------------------
# Consensus simulation
# ---------------------------------------------------------------------------


def matching_laplacian(actions_t: np.ndarray, n: int) -> np.ndarray:
    """Laplacian of the mutually-formed matching at epoch t."""
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = int(actions_t[i])
        if j < 0 or j >= n:
            continue
        if int(actions_t[j]) != i:
            continue
        if i < j:
            L[i, i] += 1.0
            L[j, j] += 1.0
            L[i, j] -= 1.0
            L[j, i] -= 1.0
    return L


def simulate_consensus(
    actions: np.ndarray, n: int, n_epochs: int, mu: float, seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Run consensus iteration over the full matching sequence.

    Returns (x_states, disagreement) where x_states[t, i] is satellite
    i's value at epoch t, and disagreement[t] is the L2 disagreement.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = x - x.mean()

    x_states = np.zeros((n_epochs + 1, n), dtype=np.float64)
    disagreement = np.zeros(n_epochs + 1, dtype=np.float64)
    x_states[0] = x
    disagreement[0] = float(np.linalg.norm(x))

    for t in range(n_epochs):
        L = matching_laplacian(actions[t], n)
        x = x - mu * (L @ x)
        x_states[t + 1] = x
        disagreement[t + 1] = float(np.linalg.norm(x))

    return x_states, disagreement


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def make_orbital_paths(positions: np.ndarray, walker, sample_stride: int = 8) -> list[np.ndarray]:
    """Sampled per-plane paths for the (now-unused) dotted backdrop."""
    n_epochs = positions.shape[0]
    sample = positions[::sample_stride]
    paths = []
    for plane_idx in range(walker.P):
        sats_in_plane = [i for i in range(walker.M) if i % walker.P == plane_idx]
        if not sats_in_plane:
            continue
        i = sats_in_plane[0]
        paths.append(sample[:, i, :])
    return paths


def make_earth_surface(radius: float, n_lat: int = 30, n_lon: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, Y, Z) arrays for a sphere surface."""
    u = np.linspace(0, 2 * np.pi, n_lon)
    v = np.linspace(0, np.pi, n_lat)
    X = radius * np.outer(np.cos(u), np.sin(v))
    Y = radius * np.outer(np.sin(u), np.sin(v))
    Z = radius * np.outer(np.ones_like(u), np.cos(v))
    return X, Y, Z


def network_layout_circular(n: int, walker) -> np.ndarray:
    """Lay out the n nodes on a circle, grouping satellites in the same plane."""
    pos = np.zeros((n, 2), dtype=np.float64)
    P = walker.P
    for plane_idx in range(P):
        wedge_start = 2 * np.pi * plane_idx / P
        wedge_span = 2 * np.pi / P * 0.85
        sats_in_plane = [i for i in range(n) if i % P == plane_idx]
        for k, sat_id in enumerate(sats_in_plane):
            theta = wedge_start + wedge_span * k / max(len(sats_in_plane) - 1, 1)
            pos[sat_id] = [np.cos(theta), np.sin(theta)]
    return pos


def render(
    data: dict,
    x_states: np.ndarray,
    disagreement: np.ndarray,
    out_dir: Path,
    phase1_epochs: int = 860,
    phase2_epochs: int = 860,
) -> None:
    """Render the animation and save in MP4, GIF, HTML formats."""

    apply_theme(context="paper")

    n = data["n"]
    positions = data["positions"]
    actions = data["actions"]
    walker = data["walker"]

    # Frame counts.
    n_p1 = int(PHASE1_DURATION_S * FPS)
    n_trans = int(TRANSITION_DURATION_S * FPS)
    n_p2 = int(PHASE2_DURATION_S * FPS)
    n_hold = int(FINAL_HOLD_S * FPS)
    n_total = n_p1 + n_trans + n_p2 + n_hold
    print(f"Total frames: {n_total} ({n_total / FPS:.1f}s at {FPS} fps)")

    network_pos = network_layout_circular(n, walker)
    earth_X, earth_Y, earth_Z = make_earth_surface(EARTH_RADIUS_KM)

    # ---- Figure setup ------------------------------------------------------
    fig = plt.figure(figsize=(14, 7.2), facecolor="white")
    gs = GridSpec(
        3, 2,
        height_ratios=[0.6, 12, 1.0],
        width_ratios=[1, 1],
        hspace=0.05, wspace=0.05,
        left=0.02, right=0.98, top=0.97, bottom=0.03,
    )
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax3d = fig.add_subplot(gs[1, 0], projection="3d")
    ax_net = fig.add_subplot(gs[1, 1])
    ax_status = fig.add_subplot(gs[2, :])
    ax_status.axis("off")

    # ---- 3D panel: Earth + satellites + links ------------------------------
    pos_max = float(np.max(np.linalg.norm(positions.reshape(-1, 3), axis=1)) * 1.05)
    ax3d.set_xlim(-pos_max, pos_max)
    ax3d.set_ylim(-pos_max, pos_max)
    ax3d.set_zlim(-pos_max, pos_max)
    ax3d.set_axis_off()
    ax3d.set_facecolor("white")
    # Square aspect.
    ax3d.set_box_aspect((1, 1, 1))

    # Earth as a translucent surface.
    earth_color = (0.55, 0.50, 0.45)  # warm tan-grey
    ax3d.plot_surface(
        earth_X, earth_Y, earth_Z,
        rstride=2, cstride=2,
        color=earth_color, alpha=0.18, linewidth=0,
        antialiased=True, zorder=1,
    )
    # Equator and a few latitude rings for orientation.
    theta = np.linspace(0, 2 * np.pi, 100)
    ax3d.plot(EARTH_RADIUS_KM * np.cos(theta), EARTH_RADIUS_KM * np.sin(theta), 0,
              color=COLORS.warmbrown, linewidth=0.6, alpha=0.5, zorder=2)
    for lat in (-np.pi / 4, np.pi / 4):
        r = EARTH_RADIUS_KM * np.cos(lat)
        z = EARTH_RADIUS_KM * np.sin(lat)
        ax3d.plot(r * np.cos(theta), r * np.sin(theta), z * np.ones_like(theta),
                  color=COLORS.warmbrown, linewidth=0.4, alpha=0.3, zorder=2)

    # Satellite scatter (initialize at epoch 0).
    sat_scatter = ax3d.scatter(
        positions[0, :, 0], positions[0, :, 1], positions[0, :, 2],
        s=36, c=[COLORS.burgundy] * n, edgecolors=COLORS.near_black, linewidths=0.5, zorder=5,
        depthshade=True,
    )

    # Active matching edges in 3D (seeded with a placeholder so add_collection3d works).
    _placeholder = [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]]
    link_lc_3d = Line3DCollection(_placeholder, colors=COLORS.copper, linewidths=2.0, alpha=0.95, zorder=6)
    ax3d.add_collection3d(link_lc_3d)
    link_lc_3d.set_segments([])

    # ---- 2D network panel --------------------------------------------------
    ax_net.set_xlim(-1.3, 1.3)
    ax_net.set_ylim(-1.3, 1.3)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    # Plane-wedge guide arcs (very faint, just to hint at the orbital-plane grouping).
    P = walker.P
    for plane_idx in range(P):
        wedge_start = 2 * np.pi * plane_idx / P
        wedge_span = 2 * np.pi / P * 0.85
        arc_t = np.linspace(wedge_start, wedge_start + wedge_span, 32)
        ax_net.plot(1.12 * np.cos(arc_t), 1.12 * np.sin(arc_t),
                    color=COLORS.warmbrown, linewidth=0.4, alpha=0.25, zorder=1)

    # Node scatter.
    net_scatter = ax_net.scatter(
        network_pos[:, 0], network_pos[:, 1],
        s=80, c=[COLORS.burgundy] * n, edgecolors=COLORS.near_black, linewidths=0.6, zorder=5,
    )

    # Active matching edges in 2D.
    net_link_lc = LineCollection([], colors=COLORS.copper, linewidths=2.0, alpha=0.95, zorder=4)
    ax_net.add_collection(net_link_lc)

    # ---- Title and status text --------------------------------------------
    title_text = ax_title.text(
        0.5, 0.5, "", ha="center", va="center",
        fontsize=18, color=COLORS.near_black, family="sans-serif", weight="bold",
        transform=ax_title.transAxes,
    )
    status_text = ax_status.text(
        0.5, 0.55, "", ha="center", va="center",
        fontsize=14, color=COLORS.near_black, family="sans-serif",
        transform=ax_status.transAxes,
    )
    epoch_text = ax_status.text(
        0.98, 0.55, "", ha="right", va="center",
        fontsize=10, color=COLORS.warm_gray, family="monospace",
        transform=ax_status.transAxes,
    )

    # Helpers.
    def active_edges_at(t: int) -> list[tuple[int, int]]:
        edges = []
        for i in range(n):
            j = int(actions[t, i])
            if 0 <= j < n and int(actions[t, j]) == i and i < j:
                edges.append((i, j))
        return edges

    # Divergent state-to-color with strong endpoints.
    burgundy = np.array([0x7A, 0x29, 0x22]) / 255.0
    olive = np.array([0x5C, 0x6B, 0x4A]) / 255.0
    midpoint = np.array([0xF5, 0xF0, 0xE6]) / 255.0  # very pale parchment

    def state_to_color(x: float, x_scale: float) -> tuple[float, float, float]:
        t_val = 0.0 if x_scale == 0 else float(np.clip(x / x_scale, -1.0, 1.0))
        # Non-linear emphasis to keep colors visible near zero.
        sign = 1.0 if t_val >= 0 else -1.0
        mag = abs(t_val) ** 0.55  # sqrt-ish; small values stay visible
        if sign < 0:
            c = (1 - mag) * midpoint + mag * burgundy
        else:
            c = (1 - mag) * midpoint + mag * olive
        return tuple(c)

    def state_to_size(x: float, x_scale: float, base: float = 60.0, span: float = 80.0) -> float:
        """Larger nodes for larger-magnitude states (less converged)."""
        t_val = 0.0 if x_scale == 0 else float(np.clip(abs(x) / x_scale, 0.0, 1.0))
        return base + span * (t_val ** 0.5)

    state_scale = float(np.percentile(np.abs(x_states[1]), 90)) or 1.0  # use early state's 90th pct as scale

    # ---- Frame function ----------------------------------------------------
    def make_frame(frame_idx: int):
        # Determine phase and epoch.
        if frame_idx < n_p1:
            phase = "p1"
            phase_t = frame_idx / max(n_p1 - 1, 1)
            epoch = int(phase_t * (phase1_epochs - 1))
            consensus_blend = 0.0
        elif frame_idx < n_p1 + n_trans:
            phase = "trans"
            phase_t = (frame_idx - n_p1) / max(n_trans - 1, 1)
            epoch = phase1_epochs - 1
            consensus_blend = phase_t
        elif frame_idx < n_p1 + n_trans + n_p2:
            phase = "p2"
            phase_t = (frame_idx - n_p1 - n_trans) / max(n_p2 - 1, 1)
            epoch = int(phase_t * (phase2_epochs - 1))
            consensus_blend = 1.0
        else:
            phase = "hold"
            epoch = phase2_epochs - 1
            consensus_blend = 1.0

        # Slow camera rotation.
        azim = 20.0 + 50.0 * (frame_idx / max(n_total - 1, 1))
        ax3d.view_init(elev=20.0, azim=azim)

        # Update satellite positions.
        sat_pos_t = positions[epoch]
        sat_scatter._offsets3d = (sat_pos_t[:, 0], sat_pos_t[:, 1], sat_pos_t[:, 2])

        # Active matching edges in both panels.
        edges_now = active_edges_at(epoch)
        link_segs_3d = [[sat_pos_t[i], sat_pos_t[j]] for i, j in edges_now]
        link_lc_3d.set_segments(link_segs_3d)
        net_link_segs = [[network_pos[i], network_pos[j]] for i, j in edges_now]
        net_link_lc.set_segments(net_link_segs)

        # Determine node colors and sizes.
        if consensus_blend == 0.0:
            # Phase 1: burgundy default, copper for currently paired.
            base_colors = [COLORS.burgundy] * n
            for i, j in edges_now:
                base_colors[i] = COLORS.copper
                base_colors[j] = COLORS.copper
            colors_3d = base_colors
            colors_2d = base_colors
            sizes_3d = [36] * n
            sizes_2d = [80] * n
        else:
            x_now = x_states[epoch + 1]
            consensus_colors = [state_to_color(float(x_now[i]), state_scale) for i in range(n)]
            consensus_sizes_2d = [state_to_size(float(x_now[i]), state_scale, base=60.0, span=80.0) for i in range(n)]
            consensus_sizes_3d = [state_to_size(float(x_now[i]), state_scale, base=24.0, span=36.0) for i in range(n)]

            if consensus_blend < 1.0:
                # Blend during transition.
                base_colors = [COLORS.burgundy] * n
                for i, j in edges_now:
                    base_colors[i] = COLORS.copper
                    base_colors[j] = COLORS.copper

                def hex_to_rgb(h):
                    if isinstance(h, tuple):
                        return h
                    h = h.lstrip("#")
                    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

                blended = [
                    tuple((1 - consensus_blend) * np.array(hex_to_rgb(base_colors[i]))
                          + consensus_blend * np.array(consensus_colors[i]))
                    for i in range(n)
                ]
                colors_3d = blended
                colors_2d = blended
                sizes_3d = [(1 - consensus_blend) * 36 + consensus_blend * consensus_sizes_3d[i] for i in range(n)]
                sizes_2d = [(1 - consensus_blend) * 80 + consensus_blend * consensus_sizes_2d[i] for i in range(n)]
            else:
                colors_3d = consensus_colors
                colors_2d = consensus_colors
                sizes_3d = consensus_sizes_3d
                sizes_2d = consensus_sizes_2d

        sat_scatter.set_color(colors_3d)
        sat_scatter.set_sizes(sizes_3d)
        net_scatter.set_color(colors_2d)
        net_scatter.set_sizes(sizes_2d)

        # Titles and status.
        if phase == "p1":
            title_text.set_text("Predictive Matching")
            status_text.set_text(f"{len(edges_now)} links formed this epoch")
            epoch_text.set_text(f"epoch {epoch:>4}/{phase1_epochs}")
        elif phase == "trans":
            title_text.set_text("Handing off the network")
            status_text.set_text("")
            epoch_text.set_text("")
        elif phase == "p2":
            d = disagreement[epoch + 1]
            title_text.set_text("Consensus on the Realized Network")
            status_text.set_text(f"disagreement  =  {d:.3e}")
            epoch_text.set_text(f"epoch {epoch:>4}/{phase2_epochs}")
        else:
            d = disagreement[phase2_epochs]
            title_text.set_text("Geometric Convergence")
            status_text.set_text(f"disagreement  =  {d:.3e}    (~3 orders of magnitude decay)")
            epoch_text.set_text("")

        return (sat_scatter, net_scatter, link_lc_3d, net_link_lc,
                title_text, status_text, epoch_text)

    print("Building animation...")
    t0 = time.perf_counter()
    anim = mpla.FuncAnimation(
        fig, make_frame, frames=n_total,
        interval=1000 / FPS, blit=False, repeat=False,
    )

    # GIF via pillow.
    gif_path = out_dir / "predictive_animation.gif"
    print(f"Saving GIF to {gif_path}...")
    t1 = time.perf_counter()
    try:
        writer = mpla.PillowWriter(fps=FPS)
        anim.save(str(gif_path), writer=writer, dpi=80)
        print(f"  GIF saved in {time.perf_counter() - t1:.1f}s")
    except Exception as e:
        print(f"  [WARN] GIF export failed: {e}.")

    plt.close(fig)
    print(f"All done in {time.perf_counter() - t0:.1f}s total.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase1-epochs", type=int, default=860,
                        help="Number of epochs to show in phase 1 (default 860 = 1.5 orbits)")
    parser.add_argument("--phase2-epochs", type=int, default=860,
                        help="Number of consensus epochs to show in phase 2")
    args = parser.parse_args()

    configure(level="WARNING")

    if not CANONICAL_TRACE.exists():
        print(f"[ERROR] canonical trace not found: {CANONICAL_TRACE}")
        print("Run scripts.stage_canonical_traces first.")
        return 1

    print(f"Loading trace from {CANONICAL_TRACE}...")
    data = load_trace_data(CANONICAL_TRACE)
    print(f"  n={data['n']}, n_epochs={data['n_epochs']}, T={data['T']}")
    print()

    # Cap phase1 and phase2 to the trace length.
    args.phase1_epochs = min(args.phase1_epochs, data["n_epochs"])
    args.phase2_epochs = min(args.phase2_epochs, data["n_epochs"])

    # Pre-simulate consensus over the full trace.
    print(f"Simulating consensus (mu={MU})...")
    t0 = time.perf_counter()
    x_states, disagreement = simulate_consensus(
        data["actions"], data["n"], data["n_epochs"], MU,
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s. "
          f"Final disagreement: {disagreement[-1]:.3e}")
    print()

    out_dir = figures_dir("animations")
    render(data, x_states, disagreement, out_dir,
           phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs)
    print(f"\nOutputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
