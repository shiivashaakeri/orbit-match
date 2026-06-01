# ACC2027 — Decentralized ISL Topology Formation as a Game

Python project extracted from `Satellite_network_formation_game.ipynb` (kept in
this folder as the reference). Each satellite carries four laser terminals
(front/behind/left/right) and forms inter-satellite links (ISLs) by a distributed
best-response policy in a potential game; baselines are greedy-closest,
greedy-furthest, and minimum-degree/maximum-distance (MDMD).

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run.py                          # full 1584-sat shell, 2000 km ISL range
python run.py --isl-range-km 3000      # wider range (slower)
python run.py --quick                  # 60-sat smoke run, seconds
python run.py --methods game furthest greedy mdmd --save-graphs
```

Each run writes a timestamped directory under `results/` containing
`config.json`, `metrics.csv`, `comparison.png`, and `union_lambda2.png`
(plus `graphs.pkl` with `--save-graphs`).

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--isl-range-km` | 2000 | max inter-satellite link range |
| `--fov-angle` | 60 | terminal cone width (deg) |
| `--epochs` | 7 | number of timesteps |
| `--t-end-min` | 1 | propagation horizon (minutes) |
| `--window` | 10 | windowed-union horizon T (epochs) |
| `--alpha` | 0.5 | slewing-cost weight |
| `--planes` / `--sats-per-plane` | 24 / 66 | constellation size |
| `--quick` | off | 6×10 = 60-sat constellation |
| `--methods` | game furthest | subset of `game furthest greedy mdmd` |

## Layout

```
config.py            # SimConfig: every tunable in one dataclass
run.py               # CLI: flags -> config -> simulate -> save metrics + figures
plots.py             # comparison figure + cumulative-union λ₂ analysis
satgame/
  constellation.py   # Satellite, spacex_constellation, SGP4 propagation
  geometry.py        # relative bearings + field-of-view feasibility
  graph.py           # empty-network + windowed-union G^cup(t;T)
  baselines.py       # greedy / furthest / MDMD baselines
  game.py            # game_theory_formation (the potential-game policy)
  metrics.py         # algebraic connectivity + effective resistance
  simulate.py        # trajectory precompute + epoch-by-epoch driver
results/             # per-run outputs (timestamped)
```

## The game policy (what `game.py` does)

Per terminal, satellite *i* ranks each feasible neighbor *j* by

```
score_i(j) = log(1 + R_ij) − α · angle(θ_prev, θ_ij)
```

where `R_ij` is the effective resistance between *i* and *j* estimated **locally**
on the union of *i*'s and *j*'s `est_radius`-hop neighborhoods in the
**windowed-union** graph `G^cup(t;T)`, measured **before** the link is added (the
exact marginal of log #spanning-trees); the second term is the slewing cost of
repointing the terminal.

Links are then realized by a **stable matching** (deferred acceptance) over the
terminal direction-pairs (front↔behind, left↔right). Matching enforces the
**mutual-choice** rule directly and avoids the wasted-terminal reciprocation
failure of a one-shot pick, while the per-direction structure caps realized
degree at 4 — matching the hardware budget and the baselines. This is the
"matching game" of the project writeups.

> The original notebook had a dead slewing term, read resistance after adding the
> edge, formed links unilaterally (degree could exceed 4), and optimized the
> per-epoch snapshot. The corrected policy fixes all four and realizes links by
> matching rather than an optimistic one-shot selection.
