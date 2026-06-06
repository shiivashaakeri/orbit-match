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
from satgame.graph import union_series
from satgame.metrics import logtau
from satgame.simulate import precompute_trajectories, run_simulation


def union_curve(graphs, window):
    """Per-epoch windowed-union lambda_2, log tau, and union edge count.

    Evaluated on the windowed-union graph G^cup(t; T) the game targets (same
    quantity ``run.py`` reports), so the sweep is consistent with the main run.
    log tau is the optimized objective; it stays finite even when the union is
    not yet connected (spanning-forest fallback), unlike lambda_2.
    """
    series = union_series(graphs, window)
    l2, lt, edges = [], [], []
    for U in series:
        connected = nx.number_connected_components(U) == 1
        l2.append(nx.algebraic_connectivity(U) if connected else 0.0)
        lt.append(logtau(U))
        edges.append(U.number_of_edges())
    return l2, lt, edges


def mean_snapshot_edges(graphs):
    """Average realized links per epoch (snapshot edge count) -- a churn proxy."""
    return sum(G.number_of_edges() for G in graphs) / max(len(graphs), 1)


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
    f_l2, f_lt, f_ue = union_curve(fref, cfg0.window)

    rows = []
    union_curves = {}
    logtau_curves = {}
    for al in a.alphas:
        cfg = SimConfig(methods=["game"], alpha=al, **base)
        t0 = time.time()
        g = run_simulation(cfg, trajectory=traj, progress=False)["game"]
        l2, lt, ue = union_curve(g, cfg.window)
        union_curves[al] = l2
        logtau_curves[al] = lt
        print(f"  alpha={al:<5g} final_union_l2={l2[-1]:.4f} "
              f"final_union_logtau={lt[-1]:.3f} union_edges={ue[-1]:5d} "
              f"mean_snapshot_edges={mean_snapshot_edges(g):.0f} "
              f"({time.time() - t0:.0f}s)")
        rows.append({
            "alpha": al,
            "final_union_lambda2": l2[-1],
            "final_union_logtau": lt[-1],
            "union_edges": ue[-1],
            "mean_snapshot_edges": mean_snapshot_edges(g),
        })

    rows.append({
        "alpha": "furthest",
        "final_union_lambda2": f_l2[-1],
        "final_union_logtau": f_lt[-1],
        "union_edges": f_ue[-1],
        "mean_snapshot_edges": mean_snapshot_edges(fref),
    })

    with open(out / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "sweep_config.json").write_text(json.dumps({**base, "alphas": a.alphas}, indent=2))

    # Figure 1: windowed-union lambda_2 and log tau vs epoch, one curve per alpha.
    ep = list(range(cfg0.epochs))
    fig1, (axa, axb) = plt.subplots(1, 2, figsize=(14, 5))
    for al in a.alphas:
        axa.plot(ep, union_curves[al], marker="o", label=f"game α={al:g}")
        axb.plot(ep, logtau_curves[al], marker="o", label=f"game α={al:g}")
    axa.plot(ep, f_l2, "k--", marker="s", label="furthest")
    axb.plot(ep, f_lt, "k--", marker="s", label="furthest")
    axa.set_xlabel("epoch")
    axa.set_ylabel("windowed-union λ₂")
    axa.set_title("Windowed-union λ₂ vs slewing weight α")
    axa.legend()
    axa.grid(alpha=0.3)
    axb.set_xlabel("epoch")
    axb.set_ylabel("windowed-union log τ")
    axb.set_title("Windowed-union log τ (objective) vs α")
    axb.legend()
    axb.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out / "union_lambda2_vs_alpha.png", dpi=150)
    plt.close(fig1)

    # Figure 2: final union lambda_2 and union edge count (churn) vs alpha.
    als = list(a.alphas)
    fl2 = [r["final_union_lambda2"] for r in rows if r["alpha"] != "furthest"]
    fue = [r["union_edges"] for r in rows if r["alpha"] != "furthest"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(als, fl2, "b-o", label="final union λ₂")
    ax1.axhline(f_l2[-1], color="b", ls=":", alpha=0.6, label="furthest λ₂")
    ax1.set_xlabel("α (slewing weight)")
    ax1.set_ylabel("final windowed-union λ₂", color="b")
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
