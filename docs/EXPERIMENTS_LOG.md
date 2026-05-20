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

## 2026-05-20 — Theory revision: union-graph certificate, effective-resistance value

Two substantive changes to the paper and a cascade of code edits:

1. Certificate moved from windowed sum to union graph. The theorem
   guarantee is now on lambda_2(L_union_G(t; T)) >= rho * alpha_0,
   per the joint-connectivity framework of Jadbabaie-Lin-Morse (2003).
   The windowed Laplacian Phi(t) is retained as the policy's internal
   smoothing surrogate.

2. Value function switched from lambda_2 marginal to Kirchhoff-index
   (effective resistance) marginal, evaluated on Phi(t) + epsilon *
   L_union_F. The epsilon term encodes common-knowledge orbital
   geometry as a soft prior on the evaluation graph; it is *not* a
   regularization device, and it does not appear in the certificate.

3. Switching cost normalized to [0, 1] by dividing the angular slew
   by pi. Switching-cost scale c now in [0, 1] with the interpretation
   "max fraction of normalized value a full slew can offset."

Empirical (small config, full orbit, T=60, c=0.2, epsilon=0.01):
  mean lambda_2(Phi) = 1.74 (above alpha_0 = 0.92 geometric ceiling)
  zero-connectivity epochs = 0 / 513
  mean edges/epoch = 9.86 / 12 max
  total deferrals = 3030 over 573 epochs
