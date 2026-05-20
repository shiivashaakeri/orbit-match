# `orbitmatch` — Project Plan

This document is the source of truth for the simulation companion to the ACC paper
*Predictive Inter-Satellite Link Formation as a Matching Game*. It captures
architecture, conventions, experiment design, and the running task list. Edit it
when decisions change.

Last reviewed: 2026-05-20 (theory revision: union-graph certificate, Kirchhoff-based value function).

---

## 1. Scope

The simulation has three jobs:

1. **Validate the analytical results of §IV.** Show empirically that the
   predictive matching policy achieves $\lambda_2(L^\cup_\mathcal{G}(t; T))
   \geq \rho \cdot \alpha_0$ for a representative LEO constellation, with
   $\rho$ in the theorem-predicted range. The certificate is on the
   *realized union graph*; the windowed Laplacian $\Phi(t)$ is tracked as
   a secondary metric for diagnostics.
2. **Compare against baselines.** Demonstrate that reciprocation prediction is
   doing real work — predictive matching outperforms greedy local value and
   random matching, and approaches the equilibrium ceiling.
3. **Produce four publication-quality figures** for §V and any tables that
   accompany them.

Out of scope: learned policies, real ephemerides, hardware-in-the-loop dynamics,
consensus protocol convergence experiments (Corollary 7 makes that a textbook
consequence once the certificate holds).

---

## 2. System model

### 2.1 Constellations

Walker–Delta in $i:M/P/F$ notation. Two configurations:

| Config  | $M$ | $P$ | $F$ | Altitude | Inclination | Per-plane | Period |
|---------|-----|-----|-----|----------|-------------|-----------|--------|
| Small   | 24  | 4   | 1   | 550 km   | 53°         | 6         | ~95.7 min |
| Medium  | 60  | 6   | 2   | 550 km   | 53°         | 10        | ~95.7 min |

Both configs use circular orbits and the standard Walker–Delta phase formula.
The medium config is the headline; the small config is for fast ablations and
ground truth (brute-force comparisons remain tractable).

Scaling experiment additionally uses $n \in \{12, 24, 48, 96\}$ — each a
Walker–Delta with $P = n/6$ planes (or similar). See `configs/sweep_scaling.yaml`.

### 2.2 Feasibility

Pair $(i,j)$ is feasible at epoch $t$ iff all three hold:

- **Line of sight.** Chord from $r_i(t)$ to $r_j(t)$ clears Earth plus
  atmospheric buffer of $h_\mathrm{atm} = 80$ km above the surface.
- **Range.** $\|r_i(t) - r_j(t)\| \leq d_\mathrm{max}$ with $d_\mathrm{max} = 8000$ km.
- **Pointing rate.** $\|\dot{\hat{u}}_{ij}(t)\|_2 \leq \omega_\mathrm{max}$ with
  $\omega_\mathrm{max} = 1°/\mathrm{s} \approx 0.01745\ \mathrm{rad/s}$.

The pointing-rate derivative is computed by centered finite differences on the
unit-bearing vector across one epoch.

### 2.3 Discretization

- Epoch length $\Delta t = 10$ s.
- Simulation horizon: 3 orbital periods, $\sim 1722$ epochs.

### 2.4 Switching cost

The raw switching cost is the angular slew $\angle(\theta_i(t-1),
\hat{\theta}_{ij}(t)) \in [0, \pi]$. The normalized form used by the
decision rule is

$$\tilde C^\text{switch}_{ij}(t) = \frac{\angle(\theta_i(t-1), \hat{\theta}_{ij}(t))}{\pi} \in [0, 1].$$

The scaling parameter $c \in [0, 1]$ is the maximum fraction of normalized
value a full $\pi$-radian slew can offset. Default $c = 0.2$. The actual
value used is reported in
the experiment log once tuned.

### 2.5 Value function: effective resistance with geometric prior

The policy ranks candidate edges by their marginal decrease in the
*Kirchhoff index* (sum of inverse non-zero Laplacian eigenvalues) of an
*evaluation graph*:

$$\Phi_\varepsilon(t) = \Phi(t) + \varepsilon \cdot L^\cup_\mathcal{F},$$

where $\Phi(t)$ is the windowed sum of realized matching Laplacians and
$\varepsilon L^\cup_\mathcal{F}$ is a soft geometric prior derived from
the feasibility union (common knowledge by Assumption that orbits are
known).

The raw value is

$$V_i(j; t) = \sum_{\tau=0}^{H-1} \big[\Omega(\Phi_\varepsilon(t+\tau)) - \Omega(\Phi_\varepsilon(t+\tau) + L_{(i,j)})\big] \cdot \mathbf{1}\{(i,j) \in \mathcal{F}(t+\tau)\}.$$

Larger drops = better connectivity contribution. The normalized form
$\tilde V_i = V_i / \max_k V_i(k;t)$ is applied at the decision
boundary; per-satellite normalization preserves argmax within each
satellite's candidate set.

The certificate (Theorem 6) is on the realized union graph
$\mathcal{G}^\cup(t; T)$, which contains no $\varepsilon$ term. The
regularization is purely a device of the value function.

---

## 3. Policy and hyperparameters

| Symbol           | Meaning                          | Default | Sweep range          |
|------------------|----------------------------------|---------|----------------------|
| $H$              | Lookahead horizon (epochs)       | 10      | {1, 5, 10, 20, 50}   |
| $T$              | Certificate window length        | 30      | {10, 30, 60, 100}    |
| $T_0$            | Feasibility window               | computed| n/a (geometry)       |
| $c$              | Switching cost scale             | 0.2     | {0, 0.1, 0.3, 0.5}   |
| $\varepsilon$    | Geometric-prior weight           | 0.01    | {0.001, 0.01, 0.1}   |
| One-step BR      | Reciprocation predictor          | fixed   | (extension only)     |

Reciprocation predictor is fixed at one-step throughout (per §III.C). The
"equilibrium" baseline runs best-response dynamics to convergence at each epoch,
which serves as our practical proxy for "centralized optimal" — defensible
because Theorem 6's $\rho_\text{cover}$ is defined against this ceiling anyway.

### 3.1 Baselines

| Name          | Description |
|---------------|-------------|
| `predictive`  | The §III policy with one-step $p_{ij}$. **Ours.** |
| `greedy`      | $a_i = \arg\max V_i$, ignores $p_{ij}$. Tests whether deferral helps. |
| `random`      | Uniform over $\mathcal{F}_i(t) \cup \{\varnothing\}$. Lower-bound reference. |
| `equilibrium` | Full BR dynamics to convergence per epoch. Practical ceiling. |

---

## 4. Experiments

### 4.1 Sanity checks (`scripts/run_sanity_checks.py`)

Diagnostic only; not in the paper. Run before any paper experiment, after any
change to feasibility or graph code.

| ID  | Check                                                              | Pass criterion                                      |
|-----|--------------------------------------------------------------------|-----------------------------------------------------|
| S1  | Feasibility-union graph $\mathcal{F}^\cup(t; T_0)$ connected       | $\lambda_2(L^\cup_\mathcal{F}) > 0$ for all $t$, $T_0 = T_\mathrm{orb}$ |
| S2  | $\alpha_0$ empirically computed                                    | Reported in `results/diagnostics/alpha_0.npz`       |
| S3  | Deferral mechanism fires                                           | $\geq 5\%$ of epochs have $p_{ij^\star} = 0$ for some $i$ |
| S4  | BR dynamics monotone improvement                                   | $W_t$ non-decreasing across rounds in `equilibrium` |
| S5  | Single-orbit visualization                                         | `figures/diagnostics/orbit_animation.png` looks reasonable |

### 4.2 Paper experiments

| Plot  | Script                                | Config        | Output                                  | Description |
|-------|---------------------------------------|---------------|-----------------------------------------|-------------|
| **1** | `scripts.run_lambda2_traces`          | medium (60)   | `figures/paper/fig1_lambda2_traces.pdf` | $\lambda_2(L^\cup_\mathcal{G}(t; T))$ over time for all 4 policies, with $\rho \alpha_0$ reference line. $\lambda_2(\Phi(t))$ shown as secondary curve. |
| **2** | `scripts.run_horizon_ablation`        | small (24)    | `figures/paper/fig2_horizon_ablation.pdf` | Time-averaged $\bar{\lambda}_2$ vs. $H$, 5 seeds, error bars |
| **3** | `scripts.run_scaling`                 | $n$ varies    | `figures/paper/fig3_scaling.pdf`        | $\bar{\lambda}_2 / \alpha_0$ vs. $n$, predictive vs. greedy |
| **4** | `scripts.run_robustness` *(optional)* | medium (60)   | `figures/paper/fig4_robustness.pdf`     | $\lambda_2(L^\cup_\mathcal{G}(t; T))$ before/after 5-sat dropout, recovery comparison |

Plot 4 runs locally but only goes in the paper if §V has space.

### 4.3 Tables

| Table | Script | Output | Description |
|-------|--------|--------|-------------|
| 1 | `scripts.render_tables` | `tables/tab1_summary.tex` | Per-policy mean/std/min of $\lambda_2(L^\cup_\mathcal{G})$ and $\lambda_2(\Phi)$ for both configs |
| 2 | `scripts.render_tables` | `tables/tab2_efficiency.tex` | Empirical $\rho_\mathrm{match}$, $\rho_\mathrm{cover}$, $\rho$ vs. theorem bound |

---

## 5. Package architecture

```
orbitmatch/
├── constellation/
│   ├── walker_delta.py       Walker–Delta geometry & TLE/Keplerian setup
│   └── propagator.py         skyfield wrapper, batched propagation
├── feasibility/
│   ├── predicates.py         LOS, range, pointing-rate (single-pair, vectorized)
│   └── compute.py            Batched (n_epochs, n, n) feasibility tensor
├── graph/
│   ├── laplacian.py          Matching → Laplacian; sparse-friendly
│   ├── windowed.py           Windowed Laplacian Φ(t)
│   └── spectral.py           λ_2 with caching; small dense vs. large sparse
├── policy/
│   ├── base.py               Policy interface (abstract)
│   ├── predictive.py         PredictiveMatching (ours)
│   ├── baselines.py          Greedy, Random
│   └── equilibrium.py        Full best-response dynamics
├── experiments/
│   ├── runner.py             run_simulation(constellation, policy, params)
│   ├── diagnostics.py        Deferral counter, BR-round counter, etc.
│   └── sweep.py              Parameter sweeps (H, n, c)
├── plotting/
│   ├── theme.py              Earthy color palette, matplotlib rcParams
│   ├── paper_plots.py        plot_lambda2_traces, plot_horizon_ablation, ...
│   └── diagnostic_plots.py   Orbit animations, feasibility heatmaps
└── utils/
    ├── io.py                 Save/load .npz, .parquet; never pickle
    ├── seeding.py            np.random.default_rng with logged seeds
    ├── timing.py             @timed context manager
    └── logging_setup.py      Project-wide logger
```

### 5.1 Data flow

```
configs/*.yaml
    │
    ▼
scripts/run_*.py
    │
    ├──▶ orbitmatch.constellation ──▶ positions  (cached in data/processed/)
    │
    ├──▶ orbitmatch.feasibility   ──▶ F[t,i,j]  (cached in data/processed/)
    │
    ├──▶ orbitmatch.policy        ──▶ a(t), G(t)
    │
    ├──▶ orbitmatch.graph         ──▶ Φ(t), λ_2(Φ(t))
    │
    └──▶ orbitmatch.experiments   ──▶ traces, diagnostics
                                          │
                                          ▼
                                  results/**/*.npz
                                          │
                                          ▼
                          orbitmatch.plotting → figures/paper/*.pdf
```

### 5.2 Caching policy

- **Constellation positions and feasibility tensors** are deterministic given
  the config — cache in `data/processed/` with a content-hashed filename, e.g.
  `positions_walker_24_4_1_alt550_inc53_dt10_horizon3orb_{hash8}.npz`.
- **Simulation traces** save *everything* needed to redraw plots: per-epoch
  $\lambda_2(L^\cup_\mathcal{G}(t; T))$, $\lambda_2(\Phi(t))$, realized matchings as edge lists, diagnostics, policy
  config, seed. Saved to `results/{experiment}/{policy}_{config}_{hash}.npz`.
- **Never re-run a simulation that already has a saved trace.** All
  `render_*.py` scripts read from `results/` and produce figures; they do not
  re-simulate.

### 5.3 Naming convention for saved files

`{prefix}_{config}_{policy}_{params}_{seed}.npz`

Example:
`trace_medium_predictive_H10_T30_c1.0_seed42.npz`

Stick to this format. The `utils.io` module provides helpers `save_trace` and
`load_trace` that enforce it.

---

## 6. Plotting conventions

### 6.1 Color palette (earthy, no neon)

Hex codes locked in `orbitmatch/plotting/theme.py`. Five-color palette plus
neutrals:

| Role          | Name      | Hex       |
|---------------|-----------|-----------|
| Primary       | burgundy  | `#7A2922` |
| Secondary     | copper    | `#B87333` |
| Tertiary      | olive     | `#5C6B4A` |
| Quaternary    | warmbrown | `#7B5C3E` |
| Background    | parchment | `#D4C4A8` |
| Body text     | near-black| `#2C2C2A` |
| Grid          | warm-gray | `#9C9A92` |

Policy-to-color assignment (consistent across all plots):

| Policy        | Color     |
|---------------|-----------|
| `predictive`  | burgundy  |
| `equilibrium` | copper    |
| `greedy`      | olive     |
| `random`      | warmbrown |
| Theorem line  | near-black (dashed) |

### 6.2 Figure parameters

- Single-column width: 3.5 in. Double-column width: 7.16 in.
- Default for paper: single-column unless explicitly two-column.
- Font: serif body, 9 pt for axis labels, 8 pt for ticks, 8 pt for legends.
- Line width: 1.2 pt for main curves, 0.8 pt for grid.
- Grid: light parchment, behind axes.
- Saved at 600 dpi PDF (vector preferred where possible).

### 6.3 Tables

LaTeX tables generated by `scripts/render_tables.py`. Use `booktabs` style
(`\toprule`, `\midrule`, `\bottomrule`). No vertical rules. Numbers right-aligned.
Three significant figures unless precision matters for a specific cell.

---

## 7. Workflow

For Shiva (collaborator-side):

1. Pull latest.
2. `pip install -e ".[dev]"` if dependencies changed.
3. Run the relevant script: `python -m scripts.run_xxx`.
4. Inspect `figures/diagnostics/` and `results/diagnostics/`.
5. Commit results files alongside the figures so re-renders are deterministic.

For Claude (in-conversation):

1. Open `docs/PROJECT_PLAN.md` to refresh on conventions.
2. Open `docs/EXPERIMENTS_LOG.md` to see what's been run.
3. Write or modify code in `orbitmatch/`.
4. Wait for Shiva to run and report.
5. Update `docs/EXPERIMENTS_LOG.md` with results.

---

## 8. Open questions and parking lot

- [ ] Verify $\alpha_0$ scales with $n$ as expected — affects scaling plot interpretation.
- [ ] Decide whether to brute-force the optimal matching on the small config for
      a tighter ceiling than `equilibrium`. Probably not worth it.
- [ ] Pointing-rate threshold $\omega_\text{max}$: 1°/s is conservative.
      A reviewer might ask why not 0.5°/s. Sensitivity ablation if space allows.
- [x] Switching cost $c$ now in $[0, 1]$ after normalizing $\tilde C^\text{switch}$ by $\pi$. Clean interpretation as max fraction of value offset by full slew.

---

## 9. Milestones

| Milestone                         | Target                  | Status |
|-----------------------------------|-------------------------|--------|
| M1: Foundation modules            | constellation, feasibility, graph | not started |
| M2: Sanity checks pass            | scripts/run_sanity_checks | not started |
| M3: Policy implementations        | all four policies        | not started |
| M4: Plot 1 reproducible           | run_lambda2_traces       | not started |
| M5: All four plots reproducible   | end of §V draft          | not started |
| M6: §V written, paper compiles    | submit to coauthors      | not started |
| M7: Submission                    | ACC deadline             | ~late Sep 2026 |

