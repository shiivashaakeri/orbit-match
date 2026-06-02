#!/usr/bin/env python3
"""Slewing-weight (alpha) sweep for the matching game.

Runs the game policy at several alpha values against a shared trajectory and the
greedy-furthest reference, and reports how the slewing weight trades off
windowed-union connectivity against link churn (the size of the windowed-union
edge set). Higher alpha penalizes repointing, so it should stabilize the
topology across epochs at some cost to connectivity.

Examples
--------
    python sweep.py                          # alpha in {0.5, 2, 5, 10}, full shell
    python sweep.py --quick                  # fast 60-sat smoke sweep
    python sweep.py --alphas 0.5 1 2 4 8 --isl-range-km 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from config import SimConfig
from satgame.metrics import series_metrics
from satgame.simulate import precompute_trajectories, run_simulation


def cumulative_union(graphs, n):
    """Per-epoch cumulative-union lambda_2 and union edge count."""
    union = nx.Graph()
    union.add_nodes_from(range(n))
    l2, edges = [], []
    for G in graphs:
        union.add_edges_from(G.edges())
        connected = nx.number_connected_components(union) == 1
        l2.append(nx.algebraic_connectivity(union) if connected else 0.0)
        edges.append(union.number_of_edges())
    return l2, edges


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 2.0, 5.0, 10.0])
    p.add_argument("--isl-range-km", type=float, default=2000.0)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--planes", type=int, default=None)
    p.add_argument("--sats-per-plane", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    planes = a.planes if a.planes is not None else (6 if a.quick else 24)
    sats = a.sats_per_plane if a.sats_per_plane is not None else (10 if a.quick else 66)
    base = dict(planes=planes, sats_per_plane=sats, isl_range_km=a.isl_range_km,
                epochs=a.epochs, window=a.window)

    out = Path(a.out) if a.out else Path("results") / (
        "sweep_alpha_" + time.strftime("%Y%m%d-%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)

    # Shared trajectory + furthest reference (computed once).
    cfg0 = SimConfig(methods=["furthest"], **base)
    n = cfg0.n_satellites
    print(f"Trajectory: {n} sats, {cfg0.epochs} epochs, {a.isl_range_km:g} km; "
          f"alphas={a.alphas}")
    traj = precompute_trajectories(cfg0, progress=True)
    fref = run_simulation(cfg0, trajectory=traj, progress=False)["furthest"]
    f_l2, f_ue = cumulative_union(fref, n)
    f_ms = series_metrics(fref)

    rows = []
    union_curves = {}
    for al in a.alphas:
        cfg = SimConfig(methods=["game"], alpha=al, **base)
        t0 = time.time()
        g = run_simulation(cfg, trajectory=traj, progress=False)["game"]
        ms = series_metrics(g)
        l2, ue = cumulative_union(g, n)
        union_curves[al] = l2
        print(f"  alpha={al:<5g} final_union_l2={l2[-1]:.4f} union_edges={ue[-1]:5d} "
              f"mean_snapshot_edges={sum(ms['edges']) / len(ms['edges']):.0f} "
              f"({time.time() - t0:.0f}s)")
        rows.append({
            "alpha": al,
            "final_union_lambda2": l2[-1],
            "union_edges": ue[-1],
            "mean_snapshot_edges": sum(ms["edges"]) / len(ms["edges"]),
            "mean_snapshot_lambda2": sum(ms["lambda2"]) / len(ms["lambda2"]),
        })

    rows.append({
        "alpha": "furthest",
        "final_union_lambda2": f_l2[-1],
        "union_edges": f_ue[-1],
        "mean_snapshot_edges": sum(f_ms["edges"]) / len(f_ms["edges"]),
        "mean_snapshot_lambda2": sum(f_ms["lambda2"]) / len(f_ms["lambda2"]),
    })

    with open(out / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "sweep_config.json").write_text(json.dumps({**base, "alphas": a.alphas}, indent=2))

    # Figure 1: cumulative-union lambda_2 vs epoch, one curve per alpha.
    ep = list(range(cfg0.epochs))
    plt.figure(figsize=(8, 5))
    for al in a.alphas:
        plt.plot(ep, union_curves[al], marker="o", label=f"game α={al:g}")
    plt.plot(ep, f_l2, "k--", marker="s", label="furthest")
    plt.xlabel("epoch")
    plt.ylabel("cumulative-union λ₂")
    plt.title("Windowed-union connectivity vs slewing weight α")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "union_lambda2_vs_alpha.png", dpi=150)
    plt.close()

    # Figure 2: final union lambda_2 and union edge count (churn) vs alpha.
    als = list(a.alphas)
    fl2 = [r["final_union_lambda2"] for r in rows if r["alpha"] != "furthest"]
    fue = [r["union_edges"] for r in rows if r["alpha"] != "furthest"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(als, fl2, "b-o", label="final union λ₂")
    ax1.axhline(f_l2[-1], color="b", ls=":", alpha=0.6, label="furthest λ₂")
    ax1.set_xlabel("α (slewing weight)")
    ax1.set_ylabel("final cumulative-union λ₂", color="b")
    ax2 = ax1.twinx()
    ax2.plot(als, fue, "r-s", label="union edges (churn)")
    ax2.axhline(f_ue[-1], color="r", ls=":", alpha=0.6)
    ax2.set_ylabel("windowed-union edge count", color="r")
    ax1.set_title("Connectivity / link-churn tradeoff vs α")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tradeoff_vs_alpha.png", dpi=150)
    plt.close(fig)

    print(f"Results written to {out}/")
    return out


if __name__ == "__main__":
    main()
