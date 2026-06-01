# Project Plan: ACC 2027 Companion Simulation

Living document. Updated whenever scope, policies, or experiment design changes.

Last revised: 2026-05-21 (v3 — ACC scope locked; lever exploration and consensus added; journal extensions split out)

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

The repository implements many policies. **Only three appear in the ACC
paper** (predictive + greedy + random). The others were explored as
candidates for headline or refinement during development but were cut
for either model-amendment reasons or because they did not Pareto-dominate
the baseline. They are preserved in the code for reproducibility and
journal-extension work.

### ACC-paper policies

| Policy             | Class                         | Role in paper                                                                |
|--------------------|-------------------------------|------------------------------------------------------------------------------|
| `predictive`       | `PredictiveMatching`          | The paper's proposed policy. One-step BR with the indicator $p_{ij}$ (eq. 17). |
| `greedy`           | `GreedyMatching`              | Baseline: $p_{ij} \equiv 1$. Demonstrates request efficiency of predictive.    |
| `random`           | `RandomMatching`              | Baseline: uniformly random feasible partner. Sets the floor for $\rho$.        |

### Internal exploration (not in the ACC paper)

| Policy             | Class                         | Why explored, why cut                                                      |
|--------------------|-------------------------------|----------------------------------------------------------------------------|
| `equilibrium`      | `EquilibriumMatching`         | GS BR to convergence; *dominated* by k1_gs (F13). Cut for clarity. |
| `k_step`           | `KStepPredictive`             | Family with $k$ BR sweeps in either sync or GS mode. Synchronous family is degenerate (collapses at $k=1$). GS at $k=1$ gives zero waste but requires broadcast assumption. Out of ACC scope. |
| `adaptive`         | `AdaptivePredictive`          | Mid-range cost/quality variant. Beaten by k1_gs. Not paper-headline. |
| `level_k`          | `LevelKPredictive`            | Cognitive hierarchy. Collapses to level 1 on Walker (F16). |
| `lever`            | `LeverPredictive`             | Predictive + scarcity weighting + history-aware $p_{ij}$. Pareto-explored; baseline always best on coverage (F17). |

### Boundary cases (verified)

- `k_step(k=0, mode=*)` ≡ `greedy`. Every satellite picks top-V; no BR.
- `k_step(k=1, mode="sync")` ≡ `predictive`. One BR sweep against the level-0 profile.
- `level_k(level=0)` ≡ `greedy`.
- `level_k(level=1)` ≡ `predictive`.
- `adaptive(k_max=1)` ≡ `predictive`.
- `adaptive(δ=0)` ≡ `predictive`.
- `lever(scarcity_beta=0, history_gamma=0)` ≡ `predictive`.

### What the paper proposes vs. what we implement

The paper proposes **one policy**: `predictive`. The paper presents
this policy with two baselines (greedy, random) and validates Theorem
6 empirically.

All other policies above were explored during the development cycle
documented in `EXPERIMENTS_LOG.md`. Findings F11-F17 in `FINDINGS.md`
summarize what each variant taught us. None of them survived to the
ACC paper, but the exploration validates that the baseline policy is
on the Pareto frontier of the design space (F17).

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
| Fig 1 ($\lambda_2$ traces) | `scripts/run_lambda2_traces.py` | ✅ Run; rendered | Certificate validation: predictive saturates $\rho$ on Walker. 3 policies (predictive, greedy, random), 3 seeds. |
| Table 1 (request efficiency) | inline in `render_paper_figures.py` (will move to its own script) | ⏳ Numbers measured, LaTeX rendering pending | Per-epoch metrics: requests, edges, waste %. |
| Fig 2 (consensus) | `scripts/run_consensus.py` | ✅ Run; rendered | Empirical demonstration of the corollary: geometric consensus decay on predictive's realized network. Single curve. |

Plot 3 (k-ablation), Plot 4 (scaling), and Plot 5 (robustness) from
earlier drafts have been removed from the ACC paper plan. The
k-ablation findings are absorbed into §V.E as a sentence-level
observation (F17); scaling and robustness move to journal-extension
material (see §11 below).

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

## 8. What is NOT yet built (for the ACC paper)

- `scripts/render_paper_figures.py` updated for the final layout:
  Fig 1 ($\lambda_2$ traces, three policies only — drop k1_gs curve),
  Table 1 (LaTeX table for §V.C), Fig 2 (already done).
- §V text written (~1.5 pages of prose around Fig 1, Table 1, Fig 2)
- §VI conclusion text
- Appendix A (Lemma 2 full potential-game proof; currently referenced
  but absent from the .tex)
- Appendix B (Theorem 6 full proof; currently referenced but absent)
- `tests/test_*.py` (proper pytest coverage; stubs only)

---

## 9. Milestones

| Milestone | Description | Status |
|---|---|---|
| M1 | Repository skeleton, conventions, theme, constellation/feasibility infra | ✅ |
| M2 | Core policies: predictive, greedy, random | ✅ |
| M2.5 | Internal exploration policies: equilibrium, k_step, adaptive, level_k, lever | ✅ |
| M3 | Sanity checks (S1-S5) | ✅ |
| M4 | Headline experiment (Fig 1) | ✅ |
| M4.5 | Lever exploration, level-k ablation, enhancement ablation | ✅ |
| M5 | Consensus corollary demonstration (Fig 2) | ✅ |
| M6 | Render final paper figures + LaTeX Table 1 | ⏳ |
| M7 | §V text, §VI conclusion, Appendices A and B | ⏳ |
| M8 | Final paper submission to ACC 2027 (deadline ~late Sep 2026) | ⏳ |

### Recent decisions affecting this plan

- **ACC scope locked to 3 policies** (predictive, greedy, random) plus
  the design-space-exploration remark.
- **k1_gs (sequential predictive) cut for ACC.** Empirically the
  strongest variant, but requires a broadcast assumption that changes
  the model. Reserved for journal extension.
- **Equilibrium, adaptive, k_step, level_k, lever all cut for ACC.**
  None Pareto-dominated baseline; all are internal exploration
  recorded in FINDINGS for posterity.
- **Plots 3, 4, 5 dropped from ACC.** k-ablation absorbed as one
  sentence in §V.E. Scaling and robustness move to journal.

---

## 10. Cross-references

- Dated run records: `docs/EXPERIMENTS_LOG.md`
- Durable empirical findings: `docs/FINDINGS.md`
- Notation glossary: `docs/NOTATION.md`
- Journal-extension experimental menu: `docs/JOURNAL_EXTENSIONS.md`
- Paper source: `paper/predictive_matching_acc.tex`

---

## 11. Journal-extension scope

The ACC paper presents a tight 3-policy story. The full work has more.
Documented in `docs/JOURNAL_EXTENSIONS.md`: candidate experiments for a
journal version (T-AC, T-CST, or similar), grouped by category
(negative controls, theorem tightness, scaling, robustness, downstream
tasks, comparison with state-of-the-art, sensitivity, model extensions,
policy-design exploration, theoretical bounds vs empirics).