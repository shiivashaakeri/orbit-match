#!/usr/bin/env python3
"""Command-line driver for the satellite ISL formation experiment.

Examples
--------
    python run.py                          # full 1584-sat run, 2000 km ISL
    python run.py --isl-range-km 3000      # wider ISL range
    python run.py --quick                  # fast 60-sat smoke run
    python run.py --methods game furthest greedy mdmd --save-graphs

Each run writes config.json, metrics.csv, and figures to results/<timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

from config import SimConfig
from plots import plot_comparison, plot_union_lambda2
from satgame.metrics import series_metrics
from satgame.simulate import run_simulation

ALL_METHODS = ["game", "furthest", "greedy", "mdmd"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--isl-range-km", type=float, default=2000.0,
                   help="max inter-satellite link range in km (default: 2000)")
    p.add_argument("--fov-angle", type=float, default=60.0,
                   help="full angular width of each terminal cone, deg (default: 60)")
    p.add_argument("--epochs", type=int, default=7,
                   help="number of timesteps (default: 7)")
    p.add_argument("--t-end-min", type=float, default=1.0,
                   help="propagation horizon in minutes (default: 1)")
    p.add_argument("--window", type=int, default=10,
                   help="windowed-union horizon T in epochs (default: 10)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="slewing-cost weight (default: 0.5)")
    p.add_argument("--planes", type=int, default=None,
                   help="orbital planes (default: 24, or 6 with --quick)")
    p.add_argument("--sats-per-plane", type=int, default=None,
                   help="satellites per plane (default: 66, or 10 with --quick)")
    p.add_argument("--quick", action="store_true",
                   help="small 60-sat constellation for fast iteration")
    p.add_argument("--methods", nargs="+", default=["game", "furthest"],
                   choices=ALL_METHODS, metavar="METHOD",
                   help=f"methods to run (subset of {ALL_METHODS})")
    p.add_argument("--save-graphs", action="store_true",
                   help="pickle the realized graphs alongside metrics")
    p.add_argument("--out", type=str, default=None,
                   help="output directory (default: results/<timestamp>)")
    return p.parse_args(argv)


def build_config(args) -> SimConfig:
    # --quick shrinks the constellation unless the user set sizes explicitly.
    planes = args.planes if args.planes is not None else (6 if args.quick else 24)
    sats = args.sats_per_plane if args.sats_per_plane is not None else (
        10 if args.quick else 66
    )
    return SimConfig(
        planes=planes,
        sats_per_plane=sats,
        isl_range_km=args.isl_range_km,
        fov_angle_deg=args.fov_angle,
        epochs=args.epochs,
        t_end_min=args.t_end_min,
        window=args.window,
        alpha=args.alpha,
        methods=list(args.methods),
        quick=args.quick,
        save_graphs=args.save_graphs,
        out_dir=args.out,
    )


def write_metrics_csv(path, metrics_by_method):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "epoch", "lambda2", "resistance", "edges", "max_degree"])
        for method, m in metrics_by_method.items():
            for epoch in range(len(m["lambda2"])):
                w.writerow([method, epoch, m["lambda2"][epoch],
                            m["resistance"][epoch], m["edges"][epoch],
                            m["max_degree"][epoch]])


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path("results") / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration: {cfg.n_satellites} satellites, {cfg.epochs} epochs, "
          f"ISL range {cfg.isl_range_km:g} km, methods={cfg.methods}")
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    t0 = time.time()
    graphs = run_simulation(cfg)
    elapsed = time.time() - t0
    print(f"Simulation finished in {elapsed:.1f}s")

    metrics_by_method = {m: series_metrics(graphs[m]) for m in graphs}
    write_metrics_csv(out_dir / "metrics.csv", metrics_by_method)

    plot_comparison(metrics_by_method, out_dir / "comparison.png")
    if "game" in graphs and len(graphs["game"]) > 0:
        plot_union_lambda2(graphs["game"], out_dir / "union_lambda2.png")

    if cfg.save_graphs:
        with open(out_dir / "graphs.pkl", "wb") as f:
            pickle.dump(graphs, f)

    # Console summary.
    print(f"\n{'method':>9} {'lambda2 (last)':>15} {'resistance (last)':>18} "
          f"{'edges (last)':>13} {'max deg':>8}")
    for method, m in metrics_by_method.items():
        print(f"{method:>9} {m['lambda2'][-1]:>15.5f} {m['resistance'][-1]:>18.1f} "
              f"{m['edges'][-1]:>13d} {max(m['max_degree']):>8d}")
    print(f"\nResults written to {out_dir}/")
    return out_dir


if __name__ == "__main__":
    main()
