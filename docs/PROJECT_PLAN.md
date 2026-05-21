# Project Plan: ACC 2027 Companion Simulation

Living document. Updated whenever scope, policies, or experiment design changes.

Last revised: 2026-05-20

---

## 1. Scope

Build the simulation companion to the ACC 2027 paper *"Predictive
Inter-Satellite Link Formation as a Matching Game"* (Li, Shakeri,
Mesbahi). The simulation supports:

- Validation of the connectivity certificate (Theorem 6) on
  representative LEO geometries.
- Empirical comparison of the predictive policy against natural
  baselines and theoretically-motivated variants.
- The figures and numerical observations in Section V of the paper.

Out of scope for ACC 2027: hardware-in-the-loop, RF/laser link-budget
modeling, atmospheric channel effects, attack/Byzantine robustness.

---

## 2. Repository structure

```
orbit-match/
├── configs/                YAML configs (constellation, policy, experiments)
├── data/                   Raw inputs and cached intermediates
│   └── processed/          Cached feasibility tensors, etc.
├── orbitmatch/             Library code
│   ├── constellation/      Walker-Delta builder, Keplerian propagator
│   ├── feasibility/        Per-epoch feasibility predicates and tensor
│   ├── graph/              Laplacians, spectral helpers, windowed objects
│   ├── policy/             All policies (see Sec 3)
│   ├── experiments/        Runner, diagnostics, sweep machinery
│   ├── plotting/           Theme, paper plots, diagnostic plots
│   └── utils/              io, logging, seeding, timing
├── scripts/                Executable entry points (run_*, check_*, render_*)
├── tests/                  pytest unit tests (light coverage)
├── results/                Saved traces and diagnostics
├── figures/                Rendered PDFs (paper/, diagnostics/)
├── docs/                   This doc, EXPERIMENTS_LOG.md, FINDINGS.md, NOTATION.md
└── paper/                  predictive_matching_acc.tex
```

---

## 3. Policies

The repository implements six policies. Three are paper-bound, three
are reference baselines or motivated by the paper's §IV.D Remark.

| Policy             | Class                         | Role                                                                                  |
|--------------------|-------------------------------|---------------------------------------------------------------------------------------|
| `predictive`       | `PredictiveMatching`          | The paper's headline policy. One-step BR with the indicator $p_{ij}$ of eq. 17.       |
| `greedy`           | `GreedyMatching`              | Baseline: $p_{ij} \equiv 1$. No reciprocation prediction.                             |
| `random`           | `RandomMatching`              | Baseline: uniformly random feasible partner per epoch.                                |
| `equilibrium`      | `EquilibriumMatching`         | Gauss-Seidel BR to fixed point, warm-started from `predictive`. Reference ceiling.    |
| `k_step`           | `KStepPredictive`             | Exactly $k$ rounds of BR. `mode ∈ {sync, gauss_seidel}`. Used for the §V $k$-ablation. |
| `adaptive`         | `AdaptivePredictive`          | $k=1$ by default; escalates to $k_{\max}$ only when top-two score gap $< \delta$.      |

### Boundary cases (verified in `scripts/check_k_step.py` and `check_adaptive.py`)

- `k_step(k=0, mode=*)` ≡ `greedy`. Every satellite picks top-V; no BR.
- `k_step(k=1, mode=*)` ≡ `predictive`. One BR sweep against the level-0 profile.
- `k_step(k=∞, mode=gauss_seidel)` from level-0 ≈ `equilibrium`, but not identical
  (equilibrium warm-starts from level-1 = predictive's output, not level-0).
- `adaptive(k_max=1)` ≡ `predictive`. No room to escalate.
- `adaptive(δ=0)` ≡ `predictive`. No satellite is ambiguous (gap ≥ 0 always; strict $\delta = 0$ never triggers escalation).
- `adaptive(δ=∞)` ≈ `k_step(k=k_max, mode=gauss_seidel)` from level-1 warm-start.

### What the paper proposes vs. what we implement

The paper of §III proposes **one policy**: `predictive` with the
one-step BR indicator. The paper's §IV proves this policy achieves
the certificate. The paper's §IV.D Remark mentions the existence of
the level-$k$ family ("k-step predictors implement k rounds, in the
limit converging to NE") without developing or naming members of it.

We implement the level-$k$ family explicitly because (i) the paper's
own Remark calls it out, (ii) §V benefits from quantifying how much
predictive leaves on the table by stopping at $k=1$, and (iii) the
`adaptive` variant turned out to be a strong middle ground that's
worth claiming in the paper. None of this requires new theory:
Theorem 6 holds for any truncation depth.

---

## 4. Experiments

### 4.1 Sanity checks (run before any paper experiment)

`scripts/run_sanity_checks.py` runs five checks on the canonical
constellation:

| Check | What it verifies | Status |
|---|---|---|
| S1 | Feasibility-union $\lambda_2 > 0$ for all $t$ at $T_0 = T_\text{orb}$ | ✅ Passes both configs |
| S2 | $\alpha_0$ computed and saved | ✅ |
| S3 | Predictive deferral mechanism fires on $\geq 5\%$ of epochs | ✅ Fires on ~100% on medium |
| S4 | BR dynamics non-decreasing in $W_t$ within each epoch | ✅ |
| S5 | Single-orbit visualization saved | ✅ |

### 4.2 Paper experiments

| Plot | Script | Status | Purpose |
|---|---|---|---|
| Plot 1 (Fig 1) | `scripts/run_lambda2_traces.py` | ✅ Run; rendered | $\lambda_2$ traces for 4 policies; the headline. |
| Plot 2 (Fig 2) | `scripts/run_horizon_ablation.py` | ⏳ Stub | $\bar\lambda_2$ vs lookahead horizon $H$. |
| Plot 3 (Fig 3) | `scripts/run_k_ablation.py` | ✅ Run (first pass) | $\bar\lambda_2$ and request efficiency vs depth $k$. |
| Plot 4 (Fig 4) | `scripts/run_scaling.py` | ⏳ Stub | $\rho$ vs constellation size $n$. |
| Plot 5 (optional) | `scripts/run_robustness.py` | ⏳ Stub | $\lambda_2$ recovery after satellite dropout. |

Plot 3 was previously slotted for scaling. The k-ablation produced
results we want in the paper (F1, F6, F7, F8 in `FINDINGS.md`), so
Plot 3 was repurposed. Scaling moves to Plot 4 or supplementary.

### 4.3 Headline experiment configuration

Medium Walker (60/6/2), 2 orbital periods, 3 seeds {42, 43, 44},
$T = T_\text{orb} = 574$ epochs, $H = 10$, $c = 0.2$,
$\varepsilon = 0.01$, dt = 10 s.

Wall-clock per job (after `_compute_top_value_partners` caching fix):
~3 min predictive/greedy/k_step(k≤3), ~5 min equilibrium, ~30 s random.

The earlier 3-period default was cut to 2 to fit the laptop budget
(~10 min total for 12 jobs vs 15-20 min at 3 periods). Returning to
3 periods is a nice-to-have but not blocking.

---

## 5. Conventions

### 5.1 Code

- Frozen dataclasses for all configuration/result types.
- `get_logger(__name__)` for all module-level loggers.
- No em-dashes anywhere in source code or docs.
- Earthy color palette (burgundy `#7A2922`, copper `#B87333`,
  olive `#5C6B4A`, warmbrown `#7B5C3E`, parchment `#D4C4A8`,
  near-black `#2C2C2A`). Defined in `orbitmatch/plotting/theme.py`;
  applied via `apply_theme(context='paper' | 'diagnostic')`.
- Type hints with `from __future__ import annotations`.
- One-screen check scripts for each new module.

### 5.2 Experiments

- **Never re-run a simulation that already has a saved trace.** The
  `run_*.py` scripts always look for cached traces first, and only
  regenerate when the cached manifest's `n_epochs`, `T`, and `H`
  match the current config (the `cached_trace_matches` invariant).
  `--force` overrides.

- **The feasibility tensor is precomputed once per (constellation, dt, params)
  tuple** and cached at `data/processed/feas_*.npz`. All experiments share
  this cache.

- **$\alpha_0$ is precomputed once per (constellation, $T_0$)** and reused
  across policies in the same experiment. Saves ~5 minutes per medium
  config sweep.

### 5.3 Trace filenames

```
results/{experiment_name}/trace_{config_label}_{policy}_T{T}_H{H}_seed{seed}.npz
```

The filename encodes every run-defining parameter. The trace's
manifest contains the full `PolicyParams` and the `DiagnosticsReport`
fields so a downstream consumer can reconstruct the run without
needing the YAML.

### 5.4 Sanity-check scripts vs unit tests

We maintain two test surfaces:

- **`scripts/check_*.py`**: per-module smoke tests, runnable directly.
  These exercise typical usage and verify boundary cases (e.g.
  `check_k_step.py` confirms $k = 0 \equiv$ greedy, $k = 1 \equiv$
  predictive). They are the primary integration tests.

- **`tests/test_*.py`**: pytest unit tests for individual functions.
  Light coverage; primarily for things that benefit from regression
  testing (graph functions, propagator math, etc.). Stubs currently;
  expand opportunistically.

---

## 6. Plotting

Theme module (`orbitmatch/plotting/theme.py`) is the single source of
truth for colors, fonts, and IEEE figure dimensions. Every plotting
function takes data in memory and returns `(fig, ax)`; no I/O. Saving
is the caller's job.

Paper plots (`paper_plots.py`): `plot_lambda2_traces`,
`plot_horizon_ablation`, `plot_scaling`, `plot_robustness`. Sized for
IEEE single column (3.5") or double (7.16"). 9pt labels, 8pt ticks,
1.2pt lines, 600 DPI PDF.

Diagnostic plots (`diagnostic_plots.py`): `plot_orbit_projection`,
`plot_feasibility_heatmap`, `plot_lambda2_phi_trace`,
`plot_deferral_histogram`, `plot_edge_count_trace`,
`plot_br_rounds_trace`. Diagnostic-context theme (larger fonts).

---

## 7. Performance

After today's optimization work:

- **Medium config full 2-period run, 12 jobs, single seed-set**: 636 s
- **Medium config 6-policy k-ablation, single seed**: 470 s
- **Predictive single-job wall-clock**: ~3 min (down from
  unacceptably-slow before the `_compute_top_value_partners` caching)
- **Feasibility cache hit**: near-instant after the first run on a
  given constellation

The remaining cost dominance is the per-edge eigendecomposition in
`_compute_value_matrix` (~210 × 1148 = 240k Kirchhoff computations
per job on medium). Sherman-Morrison rank-1 updates could cut this
~10×; parked as future optimization, not blocking ACC.

---

## 8. What is NOT yet built

- `scripts/run_horizon_ablation.py` (Plot 2 data)
- `scripts/run_scaling.py` (Plot 4 data)
- `scripts/run_robustness.py` (Plot 5 data, optional)
- `scripts/render_tables.py` (LaTeX tables for §V)
- `tests/test_*.py` (proper pytest coverage)
- Paper §V (numerical experiments), §VI (conclusion), Appendix B
  (Theorem 6 full proof)
- Gauss-Seidel variant of `k_step` (currently sync-only); decision
  pending after the second k-ablation run completes

---

## 9. Milestones

| Milestone | Description | Status |
|---|---|---|
| M1 | Repository skeleton, conventions, theme, constellation/feasibility infra | ✅ |
| M2 | Policy library: predictive, greedy, random, equilibrium | ✅ |
| M2.5 | Policy extensions: k_step (sync), adaptive | ✅ |
| M3 | Sanity checks (S1-S5) | ✅ |
| M4 | Headline experiment (Plot 1) | ✅ |
| M4.5 | k-ablation (Plot 3, first pass) | ✅ |
| M5 | Gauss-Seidel k_step variant + re-run ablation | ⏳ |
| M6 | Remaining ablations (Plot 2 horizon, Plot 4 scaling) | ⏳ |
| M7 | Render all paper figures + LaTeX tables | ⏳ |
| M8 | §V text written, §VI conclusion, Appendix B proof | ⏳ |
| M9 | Final paper submission to ACC 2027 (deadline ~late Sep 2026) | ⏳ |

### Recent decisions affecting this plan

- **k-ablation became Plot 3** (was scaling). Scaling moved to Plot 4.
- **Equilibrium reframed as "iterated-BR reference"** rather than a
  co-equal algorithm. Predictive is the paper's single proposed
  policy; equilibrium and k_step are diagnostic/analytic.
- **Headline framing shifted from "predictive beats greedy on $\lambda_2$"
  to "predictive achieves certificate with 37% fewer requests"** (per
  Findings F2 and F4). Greedy ties predictive on $\lambda_2$ but
  wastes 77% of requests.
- **Adaptive policy added** as the "deployable middle ground" between
  predictive and equilibrium; captures most of equilibrium's benefit
  at 1.28× $k=1$ cost. May get a paragraph in §V and a mention in §VI.

---

## 10. Cross-references

- Dated run records: `docs/EXPERIMENTS_LOG.md`
- Durable empirical findings: `docs/FINDINGS.md`
- Notation glossary: `docs/NOTATION.md`
- Paper source: `paper/predictive_matching_acc.tex`