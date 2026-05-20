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


