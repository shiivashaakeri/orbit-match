#!/usr/bin/env python3
"""Full best-response experiment: sweep every knob, tabulate, plot, benchmark.

Sweeps the optimism parameter ``rho`` against both initializations (cold / warm)
under simultaneous best-response dynamics, and benchmarks the result against the
matching game and the centralized ceiling on the *same* constellation and
trajectory (apples to apples).

For each (init, rho) cell it reports, on the windowed union:
  - final log tau and lambda_2 (connectivity),
and the best-response diagnostics, averaged over epochs:
  - mean rounds to settle, % epochs converged, % epochs cycled,
  - mean reciprocation-failure rate,
  - mean empirical epsilon-Nash gap.

The headline contrasts: cold-start + strict (rho=0) collapses to the
coordination-failure trap; optimism (rho->1) escapes it at the cost of
reciprocation failures; and the whole BR family is measured against matching.

Examples
--------
    python sweep_br.py                       # 60-sat, 6000 km (fast, connects)
    python sweep_br.py --full                # 1584-sat, 2000 km (matches the paper run)
    python sweep_br.py --rhos 0 0.5 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from config import SimConfig
from satgame.best_response import run_best_response
from satgame.graph import union_series
from satgame.metrics import logtau
from satgame.simulate import precompute_trajectories, run_simulation


def final_union(graphs, window):
    """Connectivity of the final windowed-union graph."""
    U = union_series(graphs, window)[-1]
    connected = nx.number_connected_components(U) == 1
    return {
        "logtau": logtau(U),
        "lambda2": nx.algebraic_connectivity(U) if connected else 0.0,
        "edges": U.number_of_edges(),
    }


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rhos", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    p.add_argument("--inits", nargs="+", default=["cold", "warm"],
                   choices=["cold", "warm"])
    p.add_argument("--planes", type=int, default=6)
    p.add_argument("--sats-per-plane", type=int, default=10)
    p.add_argument("--isl-range-km", type=float, default=6000.0)
    p.add_argument("--fov-angle", type=float, default=60.0)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--full", action="store_true",
                   help="1584-sat / 2000 km shell to match the main run (slow)")
    p.add_argument("--no-eps", action="store_true",
                   help="skip the epsilon-Nash measurement (faster)")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.full:
        planes, sats, rng = 24, 66, 2000.0
    else:
        planes, sats, rng = a.planes, a.sats_per_plane, a.isl_range_km

    cfg = SimConfig(
        planes=planes, sats_per_plane=sats, isl_range_km=rng,
        fov_angle_deg=a.fov_angle, epochs=a.epochs, window=a.window,
        alpha=a.alpha, methods=["game", "centralized"],
    )
    out = Path(a.out) if a.out else Path("results") / (
        "br_sweep_" + time.strftime("%Y%m%d-%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)

    print(f"BR sweep: {cfg.n_satellites} sats, {cfg.epochs} epochs, {rng:g} km; "
          f"rhos={a.rhos}, inits={a.inits}")
    traj = precompute_trajectories(cfg, progress=True)

    # --- Reference: matching game + centralized on the same trajectory. ---
    ref_graphs = run_simulation(cfg, trajectory=traj, progress=True)
    reference = {m: final_union(ref_graphs[m], cfg.window) for m in ref_graphs}
    print(f"  reference  matching(game) logtau={reference['game']['logtau']:.3f}  "
          f"centralized logtau={reference['centralized']['logtau']:.3f}")

    # --- BR grid. ---
    rows = []
    curves = {init: {"rho": [], "logtau": [], "lambda2": [], "recip": [],
                     "eps": [], "rounds": [], "conv": []} for init in a.inits}
    for init in a.inits:
        for rho in a.rhos:
            t0 = time.time()
            res = run_best_response(cfg, rho, init, traj, measure_eps=not a.no_eps)
            fu = final_union(res["graphs"], cfg.window)
            d = res["diags"]
            row = {
                "init": init, "rho": rho,
                "final_logtau": fu["logtau"], "final_lambda2": fu["lambda2"],
                "union_edges": fu["edges"],
                "mean_rounds": mean(x["rounds"] for x in d),
                "pct_converged": 100.0 * mean(1.0 if x["converged"] else 0.0 for x in d),
                "pct_cycled": 100.0 * mean(1.0 if x["cycled"] else 0.0 for x in d),
                "mean_recip_failure": mean(x["recip_failure"] for x in d),
                "mean_epsilon": mean(x.get("epsilon", float("nan")) for x in d),
            }
            rows.append(row)
            c = curves[init]
            c["rho"].append(rho); c["logtau"].append(fu["logtau"])
            c["lambda2"].append(fu["lambda2"]); c["recip"].append(row["mean_recip_failure"])
            c["eps"].append(row["mean_epsilon"]); c["rounds"].append(row["mean_rounds"])
            c["conv"].append(row["pct_converged"])
            print(f"  {init:>4} rho={rho:<4g} logtau={fu['logtau']:8.3f} "
                  f"recipfail={row['mean_recip_failure']:.2f} "
                  f"eps={row['mean_epsilon']:.3f} rounds={row['mean_rounds']:.1f} "
                  f"conv={row['pct_converged']:.0f}% ({time.time()-t0:.0f}s)")

    # --- Reference rows appended to the table. ---
    for m in ("game", "centralized"):
        rows.append({
            "init": "-", "rho": m, "final_logtau": reference[m]["logtau"],
            "final_lambda2": reference[m]["lambda2"], "union_edges": reference[m]["edges"],
            "mean_rounds": "", "pct_converged": "", "pct_cycled": "",
            "mean_recip_failure": "", "mean_epsilon": "",
        })

    # --- Write the table. ---
    cols = ["init", "rho", "final_logtau", "final_lambda2", "union_edges",
            "mean_rounds", "pct_converged", "pct_cycled", "mean_recip_failure",
            "mean_epsilon"]
    with open(out / "br_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "br_config.json").write_text(json.dumps(
        {"planes": planes, "sats_per_plane": sats, "isl_range_km": rng,
         "epochs": a.epochs, "window": a.window, "alpha": a.alpha,
         "rhos": a.rhos, "inits": a.inits}, indent=2))

    # --- Plots: 2x2 panels vs rho, with matching/centralized reference lines. ---
    _STY = {"cold": ("#dc2626", "o", "-"), "warm": ("#2563eb", "s", "-")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Best-response sweep ({cfg.n_satellites} sats, {rng:g} km) — "
                 f"simultaneous updates", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    finite_vals = []
    for init in a.inits:
        col, mk, ls = _STY[init]
        rl = list(zip(curves[init]["rho"], curves[init]["logtau"]))
        fr = [r for r, l in rl if math.isfinite(l)]
        fl = [l for r, l in rl if math.isfinite(l)]
        finite_vals += fl
        ax.plot(fr, fl, marker=mk, color=col, ls=ls, lw=2, label=f"BR {init}")
    ax.axhline(reference["game"]["logtau"], color="#059669", ls="--",
               label="matching game")
    ax.axhline(reference["centralized"]["logtau"], color="#7c3aed", ls=":",
               label="centralized")
    # Show the coordination-failure trap (log τ = -inf, empty graph) explicitly,
    # as markers pinned to a "trap floor" below the finite data instead of dropping.
    ref_vals = finite_vals + [reference["game"]["logtau"],
                              reference["centralized"]["logtau"]]
    ref_vals = [v for v in ref_vals if math.isfinite(v)]
    if ref_vals:
        ymin, ymax = min(ref_vals), max(ref_vals)
        pad = 0.10 * (ymax - ymin if ymax > ymin else 1.0)
        trap_y = ymin - pad
        trapped_rho = []
        for init in a.inits:
            col, _, _ = _STY[init]
            tr = [r for r, l in zip(curves[init]["rho"], curves[init]["logtau"])
                  if not math.isfinite(l)]
            if tr:
                ax.scatter(tr, [trap_y] * len(tr), marker="x", s=90, color=col,
                           zorder=5)
                trapped_rho += tr
        if trapped_rho:
            ax.set_ylim(trap_y - pad, ymax + pad)
            ax.axhline(trap_y, color="0.6", ls=":", lw=1)
            ax.annotate("empty graph (coordination-failure trap)",
                        xy=(min(trapped_rho), trap_y),
                        xytext=(min(trapped_rho) + 0.12, trap_y + 0.4 * pad),
                        fontsize=9, color="0.3",
                        arrowprops=dict(arrowstyle="->", color="0.5"))
    ax.set_title("Final windowed-union log τ (connectivity)")
    ax.set_xlabel("ρ (optimism)"); ax.set_ylabel("log τ"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for init in a.inits:
        col, mk, ls = _STY[init]
        ax.plot(curves[init]["rho"], curves[init]["recip"], marker=mk, color=col,
                ls=ls, lw=2, label=f"BR {init}")
    ax.set_title("Reciprocation-failure rate")
    ax.set_xlabel("ρ (optimism)"); ax.set_ylabel("fraction wasted"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for init in a.inits:
        col, mk, ls = _STY[init]
        ax.plot(curves[init]["rho"], curves[init]["eps"], marker=mk, color=col,
                ls=ls, lw=2, label=f"BR {init}")
    ax.set_title("Empirical ε-Nash gap (max unilateral gain)")
    ax.set_xlabel("ρ (optimism)"); ax.set_ylabel("ε"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for init in a.inits:
        col, mk, ls = _STY[init]
        ax.plot(curves[init]["rho"], curves[init]["rounds"], marker=mk, color=col,
                ls=ls, lw=2, label=f"BR {init} (rounds)")
    ax.set_title("Rounds to settle (cap = 50)")
    ax.set_xlabel("ρ (optimism)"); ax.set_ylabel("mean rounds"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "br_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Console table. ---
    print(f"\n{'init':>5} {'rho':>6} {'logtau':>9} {'lambda2':>9} {'recipfail':>10} "
          f"{'eps':>8} {'rounds':>7} {'conv%':>6}")
    for r in rows:
        rho = r["rho"] if isinstance(r["rho"], str) else f"{r['rho']:.2f}"
        rf = r["mean_recip_failure"]; ep = r["mean_epsilon"]
        mr = r["mean_rounds"]; cv = r["pct_converged"]
        print(f"{r['init']:>5} {rho:>6} {r['final_logtau']:>9.3f} "
              f"{r['final_lambda2']:>9.4f} "
              f"{(f'{rf:.2f}' if rf != '' else '-'):>10} "
              f"{(f'{ep:.3f}' if ep != '' else '-'):>8} "
              f"{(f'{mr:.1f}' if mr != '' else '-'):>7} "
              f"{(f'{cv:.0f}' if cv != '' else '-'):>6}")
    print(f"\nResults written to {out}/")
    return out


if __name__ == "__main__":
    main()
