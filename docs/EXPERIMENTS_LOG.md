# Experiments Log

Append a new entry every time a simulation is run. Newest first.

Format:
- Date, branch/commit, who ran it
- Script
- Config / parameters
- Outcome (one or two sentences)
- Artifacts (paths to results files and figures)

---

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