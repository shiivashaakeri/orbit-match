# ACC2027 — Decentralized ISL Topology Formation as a Game

Python project extracted from `Satellite_network_formation_game.ipynb` (kept in
this folder as the reference). Each satellite carries four laser terminals
(front/behind/left/right) and forms inter-satellite links (ISLs) by a distributed
best-response policy in a potential game. The policy is compared against two
references: `furthest`, a decentralized greedy-furthest heuristic, and
`centralized`, a centralized greedy maximizer of the same `log τ` objective with
full global state -- the optimization ceiling. So `game` vs `centralized`
measures the price of decentralization plus local information.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run.py                          # full 1584-sat shell, 2000 km ISL range
python run.py --isl-range-km 3000      # wider range (slower)
python run.py --quick                  # 60-sat smoke run, seconds
python run.py --methods game furthest centralized --save-graphs
python validate_opt.py                 # exact-OPT check on a tiny instance
```

Each run writes a timestamped directory under `results/` containing
`config.json`, `metrics.csv`, `comparison.png`, and `union_lambda2.png`
(plus `graphs.pkl` with `--save-graphs`). Metrics (λ₂, effective resistance,
`log τ`) are evaluated on the windowed-union graph; `metrics.csv` also reports
`snap_max_degree`, the per-epoch realized degree, which stays ≤ 4 (the hardware
cap). `validate_opt.py` brute-forces the true optimum on a small constellation
and confirms the centralized greedy is a tight ceiling.

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
| `--methods` | game furthest centralized | subset of `game furthest centralized` |

## Layout

```
config.py            # SimConfig: every tunable in one dataclass
run.py               # CLI: flags -> config -> simulate -> save metrics + figures
sweep.py             # slewing-weight (alpha) sweep
validate_opt.py      # exact-OPT brute force on a tiny instance (ceiling check)
plots.py             # comparison figure + cumulative-union λ₂ / log τ analysis
satgame/
  constellation.py   # Satellite, spacex_constellation, SGP4 propagation
  geometry.py        # relative bearings + field-of-view feasibility
  graph.py           # empty-network + windowed-union / union-series helpers
  baselines.py       # greedy-furthest decentralized baseline
  game.py            # game_theory_formation (the decentralized matching game)
  centralized.py     # centralized_greedy_formation (CELF ceiling, exact marginals)
  metrics.py         # algebraic connectivity + effective resistance + log τ
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

## The centralized ceiling (what `centralized.py` does)

A single planner with full global state maximizes the **same** objective
`log τ(G^cup(t;T)) − α·Σ slew` under the **same** k=4 directional terminal
budget. It differs from the game in exactly two ways, which is the point:

- **Exact global marginals** — the `log τ` single-edge gain `log(1 + R_ij)` is
  computed on the full Laplacian pseudoinverse `L†` (rank-one Sherman–Morrison
  updates), not an h-hop local estimate.
- **Globally coordinated, re-evaluated greedy** — one priority order with
  marginals recomputed after every accepted edge (true submodular diminishing
  returns), realized by lazy greedy (CELF), rather than a linearized score
  realized by bilateral matching.

So `game` vs `centralized` isolates the price of decentralization + local
information. `validate_opt.py` brute-forces the true optimum on a tiny instance;
a representative run gives `furthest < game < centralized ≲ OPT` (centralized at
~0.91 of OPT, the greedy-vs-exact gap), confirming the ceiling is tight.
