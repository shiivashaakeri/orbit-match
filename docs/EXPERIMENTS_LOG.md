# Experiments Log

Append a new entry every time a simulation is run. Newest first.

Format:
- Date, branch/commit, who ran it
- Script
- Config / parameters
- Outcome (one or two sentences)
- Artifacts (paths to results files and figures)

---

## 2026-05-20 — k1_gs validation: seed robustness + ordering sensitivity

- **Branch/commit:** main / current
- **Ran by:** Shiva (local laptop)
- **Script:** `scripts/validate_k1_gs.py`

### Question

The k-ablation showed `k1_gs` (k=1 Gauss-Seidel BR from level-0) is the
densest, zero-waste policy on medium config, beating equilibrium on
edges-per-epoch. Before reframing the paper around this finding, two
validations:

A. Robust across seeds?
B. Robust across satellite update orderings?

### Setup

Medium Walker, 2 orbital periods (1148 epochs), T=574, H=10, c=0.2,
eps=0.01. Six runs total, each ~64 s wall-clock.

### Results

Validation A (seeds {42, 43, 44}, identity ordering):

```
seed=42   req/ep 37.12   edges/ep 18.56   waste 0.0%   rho_mean 0.9988   rho_cover 0.9902
seed=43   req/ep 37.12   edges/ep 18.56   waste 0.0%   rho_mean 0.9988   rho_cover 0.9902
seed=44   req/ep 37.12   edges/ep 18.56   waste 0.0%   rho_mean 0.9988   rho_cover 0.9902

  edges/ep spread: 0.0000  rho_cover spread: 0.0000
```

Validation B (seed=42, three orderings):

```
identity    req/ep 37.12   edges/ep 18.56   rho_mean 0.9988   rho_cover 0.9902
reversed    req/ep 35.73   edges/ep 17.86   rho_mean 0.9994   rho_cover 0.9882
random_p7   req/ep 37.42   edges/ep 18.71   rho_mean 0.9998   rho_cover 0.9863

  edges/ep spread: 0.8441  rho_cover spread: 0.0039
```

### Outcomes

**Validation A: perfect.** k1_gs is fully deterministic given the
geometry. Seeds only matter for the random tie-break in
`_argmax_with_varnothing_preference`, which never fires here (scores
are distinct floats). The finding is deterministic, not statistical.

**Validation B: meaningful but small.** Different orderings yield
*different* NEs of $\Gamma_t$, but all are excellent (0% waste, very
high rho_mean). The spread is ~5% in edges/ep and 0.4% in rho_cover.
Random and reversed orderings slightly beat identity on rho_mean,
slightly lose on rho_cover. **The phenomenon: GS-from-level-0 has
multiple basins of attraction; ordering picks which one.**

**Verdict: finding holds.** k1_gs (Gauss-Seidel, from level-0, one
sweep) is the recommended deployable policy. Even the worst-ordering
NE (reversed: 17.86 edges/ep) beats equilibrium (16.46 edges/ep,
warm-started from level-1).

### Implications

- See FINDINGS F11 (k1_gs reaches NE in one sweep), F12 (different
  orderings yield different NEs), F13 (equilibrium is a *worse* NE),
  and F14 (adaptive interpretation revised).
- Reframe Fig 1 around: predictive (k1_sync), k1_gs, greedy, random.
  Drop or demote equilibrium per F13.
- Reframe Fig 3 around the *update-order axis* (sync vs GS at k=1)
  rather than the depth axis. Add a remark about NE multiplicity.

### Artifacts

- `results/k_gs_validation/trace_k1_gs_seed{42,43,44}_identity.npz`
- `results/k_gs_validation/trace_k1_gs_seed42_{identity,reversed,random_p7}.npz`

---

## 2026-05-20 — k-ablation, first run, synchronous BR + initial adaptive

- **Branch/commit:** main / current
- **Ran by:** Shiva (local laptop)
- **Script:** `scripts/run_k_ablation.py`

### Setup

Medium Walker (60/6/2), 2 orbital periods (1148 epochs), single seed 42,
T = T_orb = 574 epochs. Six policies: k_step at k in {1, 2, 3, 8} all
synchronous (Jacobi), equilibrium (Gauss-Seidel BR to convergence
warm-started from level-1), adaptive with delta_threshold=0.1, k_max=3.

### Results

```
policy        req/ep  edges/ep  waste%   rho_max  rho_mean  rho_cover
k1            38.11    6.79     64.4%    1.0000   0.9855    0.9529
k2            38.11    6.79     64.4%    1.0000   0.9855    0.9529
k3            38.11    6.79     64.4%    1.0000   0.9855    0.9529
k8            38.11    6.79     64.4%    1.0000   0.9855    0.9529
equilibrium   32.92   16.46      0.0%    1.0000   0.9993    0.9843
adaptive      36.51   10.67     41.6%    1.0000   0.9980    0.9843

Adaptive: mean ambiguous frac = 13.83% (max 40.00%, min 0.00%)
         cost vs k=1 = 1.28x (with k_max=3)
```

Wall-clock: 470 s for six policies.

### Outcomes

Three findings:

1. **Synchronous BR plateaus at k=1.** k=2, k=3, k=8 produce *identical*
   actions, matchings, and metrics to k=1. Synchronous BR hits a fixed
   point in one round on this geometry; subsequent rounds don't move
   anyone.

2. **Equilibrium (Gauss-Seidel) finds genuinely better fixed points.**
   2.4x more edges per epoch (16.46 vs 6.79), zero waste, slightly
   higher rho_mean. The "k -> infty" limit referenced in paper Sec IV.D
   Remark must be Gauss-Seidel, not synchronous.

3. **Adaptive sits halfway between predictive and equilibrium.** With
   only 13.83% of satellites escalating per epoch (1.28x compute vs
   k=1), adaptive realizes 10.67 edges/ep (57% more than k=1) and
   matches equilibrium's rho_cover exactly. The level-1 warm-start
   plus selective Gauss-Seidel-style updates of the ambiguous subset
   captures most of equilibrium's benefit cheaply.

### Decisions

- Re-run with a Gauss-Seidel k_step variant to get the proper
  k=1->infty smooth interpolation (synchronous mode doesn't deliver it).
- Reframe Plot 3 (currently scaling) as the k-ablation. Scaling moves
  to supplementary or gets cut for ACC.

### Artifacts

- `results/k_ablation/trace_medium_k{1,2,3,8}_seed42.npz`
- `results/k_ablation/trace_medium_equilibrium_seed42.npz`
- `results/k_ablation/trace_medium_adaptive_seed42.npz`

## 2026-05-20 — Request efficiency: greedy/predictive equivalence on lambda_2, divergence on waste

- **Branch/commit:** main / current
- **Ran by:** Shiva (local laptop)
- **Script:** ad-hoc diagnostic against `results/lambda2_traces/` traces

### Question

Headline run showed predictive and greedy produced identical realized
matchings (both rho_mean = 0.9855 on medium). Re-checked our optimization
to make sure we hadn't degraded predictive. Found the optimization is
correct; the equivalence is a structural property of the model.

### Diagnostic

Counted per-epoch requests (number of satellites with action != -1) and
realized edges, for each policy on medium seed 42.

```
policy        req/ep  edges/ep  waste/ep  waste%
predictive     38.11    6.79     24.53    64.4%
greedy         60.00    6.79     46.42    77.4%
random         60.00    4.20     51.61    86.0%
equilibrium    32.92   16.46      0.00     0.0%
```

### Outcomes

- **Predictive vs greedy on this geometry**: identical realized edges
  per epoch, identical rho_realized, identical union. Predictive wins
  by making 37% fewer requests (38 vs 60). The deferral mechanism
  eliminates the obviously-doomed requests but the mutual-choice rule
  was already eating those at the realization level.

- **Predictive's reciprocation prediction is wrong 64% of the time**
  (24.53 wasted requests / 38.11 total). The one-step indicator
  predicts mutual reciprocation, but two satellites both predicting
  each other can disagree (because each predicts the other as a
  value-maximizer, ignoring the other's own reciprocation logic). This
  is exactly Prop 8's statement: predictive is one round of BR, not a
  fixed point.

- **Equilibrium achieves zero waste**: at a NE every satellite's
  request is mutually consistent. Confirmed empirically.

- **Random's "waste" is highest** at 86%, predictably.

### Implication for the paper

The deferral mechanism's value is not in lambda_2 (where greedy ties
predictive). It is in operational cost: requests per epoch, energy
spent slewing the laser to partners who will not point back, command
bandwidth, mechanical wear. This is the right framing for Sec V.

### Artifacts

No new files; data is in `results/lambda2_traces/`.

## 2026-05-20 — Headline run: lambda2_traces, four policies on medium

- **Branch/commit:** main / current
- **Ran by:** Shiva (local laptop)
- **Script:** `scripts/run_lambda2_traces.py`

### Setup

Medium Walker (60/6/2), **2 orbital periods** (1148 epochs;
duration_periods cut from 3 to 2 to fit wall-clock budget), seeds
{42, 43, 44}, T = T_orb = 574 epochs, H = 10, c = 0.2, eps = 0.01.

Policies: predictive, greedy, random, equilibrium. Total 12 jobs.

### Results

```
policy        seeds  rho_max  rho_mean  rho_cover
predictive       3   1.0000   0.9855    0.9529
greedy           3   1.0000   0.9855    0.9529
random           3   0.7410   0.7165    0.9131
equilibrium      3   1.0000   0.9993    0.9843
```

Wall-clock: 636 s after the predictive-policy caching optimization
(see notes below). First (unoptimized) run was killed after >15 min on
predictive seed 1 alone.

### Outcomes

- **Predictive saturates the certificate**: rho_max = 1.0, rho_mean ~
  0.99. The certificate window is essentially full of feasibility-union
  edges by the time it fills.

- **Equilibrium edges out predictive slightly**: rho_mean 0.9993 vs
  0.9855. Both saturate at peak. Equilibrium's advantage is in the
  rolling-window steady state: it realizes denser matchings per epoch,
  so the rolling union decays less between feasibility refreshes.

- **Greedy ties predictive on every connectivity metric.** Same
  realized matchings. The deferral mechanism saves requests, not
  edges.

- **Random is clearly inferior** (rho_max 0.74); the no-strategy
  baseline does what it should.

### Note: predictive caching bug found and fixed

The initial run was unbearably slow because `_reciprocation_prob(i, j)`
was re-scoring satellite j's entire candidate set every time it was
called. Per epoch, that meant ~420 redundant re-computations.

Fixed by adding a per-epoch top-V-partner cache (`_compute_top_value_partners`)
that runs once per epoch and is read by `_reciprocation_prob` as an
O(1) lookup. Wall-clock dropped from "tens of minutes per policy seed"
to "~3 min per policy seed" on medium.

### Note: stale-trace bug found and fixed

After cutting duration_periods 3 -> 2, the second run loaded a stray
1722-epoch trace from before the kill, alongside 1148-epoch traces from
the new run. `plot_lambda2_traces` raised `ValueError: Inconsistent
n_epochs across seeds: [1722, 1148, 1148]`.

Fixed by adding `cached_trace_matches()` to `run_lambda2_traces.py`:
before treating a cached trace as valid, the manifest is checked for
n_epochs / T / H equal to the current config. Mismatch -> regenerate.

### Artifacts

- `results/lambda2_traces/trace_medium_{predictive,greedy,random,equilibrium}_T574_H10_seed{42,43,44}.npz`
- `figures/paper/fig1_lambda2_traces.pdf` (rendered separately)

## 2026-05-20 — Feasibility sanity check (medium + small)

- **Branch/commit:** main / current
- **Ran by:** Shiva (local laptop)
- **Script:** inline, manual

### Setup

Walker-Delta at 550 km, 53° inclination. Feasibility thresholds:
atm_buffer 80 km, range_max 8000 km, rate_max 1°/s, dt 10 s.

### Results

| Config | Sats | Mean feasible neighbors per sat | Saturation $\lambda_2(L^\cup)$ | Saturation $T_0$ |
|---|---|---|---|---|
| Small (24, 4, 1) | 24 | 1.82 | 0.922 | ~60 min |
| Medium (60, 6, 2) | 60 | 7.17 | 4.292 | ~60 min |

### Outcomes

- Both configs satisfy Assumption 5 with non-trivial $\alpha_0$.
- 71-74% of pairs are geometry-locked apart and never become feasible —
  this is the irreducible part that $\rho_{\mathrm{cover}}$ captures in Theorem 6.
- Picking $T_0 = 60$ min (360 epochs) gives the asymptotic $\alpha_0$.
  Certificate window $T$ should then be at least 60 min, ideally one
  full orbital period (~95 min) for smoothness.

### Decisions

- Use the values above as the empirical $(\alpha_0, T_0)$ in Plot 1 and
  the headline theorem comparison.
- The small config is sparse but connected; useful for the horizon
  ablation (Plot 2) where the trend matters more than the magnitude.

### Artifacts

- `data/processed/feas_walker_small_24_4_1_alt550_inc53_dt10_*.npz`
- `data/processed/feas_walker_medium_60_6_2_alt550_inc53_dt10_*.npz`

## 2026-05-20 — Theory revision: union-graph certificate, Kirchhoff value function

Three substantive theoretical changes, with a cascade of code edits.

### Changes

1. **Certificate moved from windowed sum to union graph.** The theorem
   now guarantees $\lambda_2(L^\cup_\mathcal{G}(t; T)) \geq \rho \alpha_0$,
   per the joint-connectivity framework of Jadbabaie-Lin-Morse (2003).
   The windowed Laplacian $\Phi(t)$ is retained as the policy's internal
   smoothing surrogate but is no longer the certified object.

2. **Value function switched from $\lambda_2$ marginal to Kirchhoff index
   marginal**, evaluated on $\Phi(t) + \varepsilon L^\cup_\mathcal{F}$.
   The $\lambda_2$ marginal is a step function in edge additions and
   gives zero gradient on disconnected graphs (which $\Phi$ is, at
   simulation start and intermittently afterward). The Kirchhoff
   marginal is smooth and gives a meaningful gradient throughout. The
   $\varepsilon L^\cup_\mathcal{F}$ term encodes common-knowledge
   orbital geometry as a soft prior; it does not appear in the
   certificate.

3. **All decision-rule quantities normalized to $[0, 1]$.** The value
   function uses per-satellite normalization $\tilde V_i = V_i /
   \max_k V_i(k;t)$. The switching cost uses fixed normalization
   $\tilde C^\text{switch}_{ij} = \angle(\cdot,\cdot)/\pi$. The
   switching-cost scale $c$ now in $[0, 1]$ with the interpretation
   "max fraction of normalized value a full slew can offset."

### Implementation

- `orbitmatch/graph/spectral.py`: added `kirchhoff_index`.
- `orbitmatch/policy/base.py`: removed `epsilon_union`, added
  `epsilon_geometric_prior`. `switching_cost_scale` validation tightened
  to $[0, 1]$, default $0.2$.
- `orbitmatch/policy/predictive.py`: rewrote `_value` to compute
  Kirchhoff drops. Rewrote `decide_for`, `_reciprocation_prob`,
  `_top_value_partner` to use per-satellite normalization. Removed
  cold-start `_NUMERICAL_FLOOR` and `_COLD_START_EPS` constants
  (subsumed by the geometric-prior regularization).

### Empirical: small config (24, 4, 1), full orbit, $T=60$, $c=0.2$, $\varepsilon=0.01$

| Metric                       | Value             |
|------------------------------|-------------------|
| Mean $\lambda_2(\Phi)$       | 1.74 (vs $\alpha_0 = 0.92$ geometric ceiling) |
| Mean edges per epoch         | 9.86 / 12 max     |
| Zero-$\lambda_2(\Phi)$ epochs | 0 / 513          |
| Total deferrals              | 3030              |
| Final-window components      | 1                 |
| Final-window isolated sats   | 0                 |

### Outcomes

- The Kirchhoff value function gives a smooth, structure-aware gradient
  that the previous $\lambda_2$-based formulation lacked.
- $\lambda_2(\Phi)$ now exceeds the geometric ceiling $\alpha_0$ in
  steady state, reflecting that the windowed sum accumulates multiple
  matchings (each matching's $\lambda_2$ alone is bounded by $\alpha_0$;
  the sum can be larger).
- The medium-config run was abandoned mid-flight under the old
  formulation. To be re-run under the new formulation as the first
  step of Phase 2.

### Artifacts

- `data/processed/feas_walker_small_24_4_1_alt550_inc53_dt10_*.npz` (unchanged)
- `data/processed/feas_walker_medium_60_6_2_alt550_inc53_dt10_*.npz` (unchanged)

## 2026-05-20 — Saturation result: $\rho = 1$ at $T = T_\text{orb}$

After the theory revision, ran both configs with the certificate window
set to a full orbital period. Both saturate exactly.

### Setup

| Setting                | Value                                |
|------------------------|--------------------------------------|
| $T = T_\text{orb}$     | 573 epochs (~95.5 min)               |
| $H$                    | 10 epochs (lookahead)                |
| $c$                    | 0.2 (switching-cost scale)           |
| $\varepsilon$          | 0.01 (geometric-prior weight)        |
| Policy                 | predictive (Kirchhoff value, BR=1)   |

### Results

| Config           | $n$ | $\alpha_0$ | feasibility edges | realized edges | $\rho_\text{realized}$ |
|------------------|----:|----------:|-----------------:|---------------:|----------------------:|
| Small (24/4/1)   |  24 | 0.9219    | 72               | 72             | **1.000**             |
| Medium (60/6/2)  |  60 | 4.2918    | 510              | 510            | **1.000**             |

Both configurations realize the full feasibility union of the orbit.
Realized-union degrees match feasibility-union degrees exactly: every
satellite linked at some point during the orbit with every other
satellite that was ever geometrically feasible.

### Diagnostics

- **Small config** (573 epochs, full orbit): 9.86 mean edges/epoch
  (steady-state), 3030 total deferrals, single connected component
  throughout warmup-and-beyond.
- **Medium config** (573 epochs, full orbit): 8.07 mean edges/epoch
  (steady-state) — much lower density of formed edges, because the
  policy correctly defers redundant edges once they're already in the
  union. 27016 total deferrals (47/epoch ≈ 78% of satellites deferring
  per epoch in steady state). Wall-clock: 891 s.

### Interpretation

The certificate window $T$ should match the geometric saturation
time. For Walker constellations, this is the orbital period: it takes
exactly one orbit for the feasibility union $\mathcal{F}^\cup(t; T)$
to stabilize. At $T < T_\text{orb}$, $\alpha_0(T)$ is smaller because
only a fraction of the orbital geometry has been exposed; at
$T \geq T_\text{orb}$, $\alpha_0$ saturates at its maximum and the
realized $\rho$ becomes a clean fraction.

The empirical observation $\rho_\text{realized} = 1$ at
$T = T_\text{orb}$ is the strongest possible certified
connectivity: the policy realizes every feasible edge over a full
orbit. The proof of Theorem 6 gives a lower bound
$\rho \geq \rho_\text{match} \cdot \rho_\text{cover}$, but the bound
is not tight for these constellations: both $\rho_\text{match}$
(Vizing-based) and $\rho_\text{cover}$ are below 1, yet their product
exceeds 1 in realized terms because the bound is conservative.

### Open questions

- $\rho$ vs. $T$ curve: how does $\rho$ degrade as $T$ shrinks below
  $T_\text{orb}$? Quick-look quantiles for small at $T_0 = 60$ epochs
  gave $\rho_\text{realized} \approx 0.38$ ($\lambda_2(L^\cup_\mathcal{G}) = 0.14$,
  $\alpha_0(T_0=60) = 0.36$). The full curve is pending.
- Baseline comparison at $T = T_\text{orb}$ pending. Hypothesis:
  Greedy realizes a smaller union (no reciprocation prediction → wasted
  attempts), Random realizes much less. Equilibrium achieves $\rho = 1$
  too but with more BR rounds.

### Files

- `data/processed/feas_walker_small_*.npz` (unchanged)
- `data/processed/feas_walker_medium_*.npz` (unchanged)
- No new result files saved yet; runs were exploratory.