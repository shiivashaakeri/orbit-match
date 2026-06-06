# Risk-Sensitive Partner Selection — Design Sketch

Response to Prof. Ratliff's proposal feedback: *"think about risk sensitivity …
choose your partner in a way that hedges for selecting wrong, as the consequences
are likely pretty catastrophic."*

## 0. The key realization

Risk only exists when **reciprocation is uncertain at decision time**. Our current
paper resolves reciprocation deterministically with a **stable matching** (deferred
acceptance), so by construction no terminal is ever wasted — we *eliminated* the
risk by coordination rather than *hedging* it. That is a fine answer, but it is not
what the feedback asks for, and it quietly assumes a level of coordination a truly
decentralized constellation may not have.

The risk-sensitive variant therefore lives in the **one-shot predict-and-commit**
setting (closer to the original k=1 proposal): each satellite commits its terminal(s)
to predicted partners, and a link forms only where two satellites happened to choose
each other. Reciprocation is now a *random event* the satellite must hedge against.
The two mechanisms become a clean comparison axis:

| Mechanism | Reciprocation | Role in the paper |
|---|---|---|
| Stable matching (current) | certain (coordinated) | risk-free benchmark |
| Risk-sensitive one-shot (new) | uncertain (predicted) | realistic decentralized policy |
| Centralized greedy | certain (global) | optimization ceiling |

## 1. Reciprocation as a random variable

Fix satellite `i` and a candidate `j`. Let

- `Δ_ij` = the connectivity marginal of the link `(i,j)` — `log(1 + R_ij)`, exactly
  as today, estimated locally on the h-hop windowed-union view.
- `p_ij ∈ [0,1]` = `i`'s predicted probability that `j` reciprocates (points back).

The realized gain from pointing at `j` is the random variable

```
X_ij = Δ_ij · Bernoulli(p_ij)      (Δ_ij if the link forms, 0 if j chose elsewhere)
```

With `k` terminals, `i`'s total realized gain is `X_i = Σ_{j ∈ s_i} X_ij` (treat the
Bernoullis as independent to first order). **The current rule maximizes `E[X_i]`;
risk sensitivity replaces `E` with a risk measure.**

### Estimating `p_ij` locally

`i` already has the machinery to estimate marginals from a few-hop view; apply it
from `j`'s side. Let `σ_ji = Δ_j(i) − α·slew_j(i)` be `i`'s estimate of `j`'s score
for `i`, and `ĉ_j^{(k)}` `i`'s estimate of `j`'s k-th-best score (the cutoff to make
`j`'s shortlist). A simple, locally computable model:

```
p_ij = sigmoid( γ · ( σ_ji − ĉ_j^{(k)} ) )
```

`p_ij → 1` when `i` clearly clears `j`'s cutoff, `→ 0` when it clearly does not, and
`≈ 0.5` (maximum uncertainty) near the cutoff. `γ` is a sharpness/confidence knob.
(Equivalently, reuse the **log-linear / Gibbs response** the paper already invokes:
`j` selects `i` with probability `∝ e^{β σ_ji}`, giving `p_ij` directly — this ties
the risk model to a learning rule already in the writeup.)

## 2. Three risk-sensitive objectives

All three keep the **per-candidate additive structure** the whole paper relies on, so
they drop into the existing scoring loop. Listed simplest → most faithful.

**(A) Mean–variance** (one extra term, trivial to implement):
```
J_i(s_i) = Σ_{j∈s_i} [ p_ij Δ_ij − λ · p_ij(1−p_ij) Δ_ij² ] − α c_i(s_i)
```
`λ ≥ 0` is risk aversion; `p_ij(1−p_ij)Δ_ij²` is the per-link variance, maximized at
`p_ij = ½` (the genuinely uncertain partners), so the rule shies away from coin-flips.

**(B) Entropic / exponential risk** (recommended): the certainty equivalent
`U_θ(X) = −(1/θ) log E[e^{−θ X}]`. For independent Bernoulli gains it is additive:
```
J_i(s_i) = − (1/θ) Σ_{j∈s_i} log( 1 − p_ij + p_ij e^{−θ Δ_ij} )  − α c_i(s_i)
```
`θ → 0` recovers the risk-neutral expected value; `θ > 0` is risk-averse. Sanity
check: a *certain* link (`p=1`, value `v`) scores `v`; a *coin-flip* link (`p=½`,
value `2v`, same expected value) scores `≈ v − θv²/2 < v` — the policy correctly
prefers the reliable partner. Entropic risk is the standard risk-sensitive-control
objective and is dual to **distributional robustness**, which connects directly to
"dynamic games of incomplete information" from the proposal.

**(C) CVaR** (most literal match to "catastrophic"): maximize `CVaR_β(X_i)`, the
expected gain in the worst `β`-fraction of outcomes. For `k=1` the outcome is a
two-point distribution and CVaR is closed-form; for `k>1` it needs a small sample/LP,
so it is heavier but speaks exactly to tail risk.

**Recommendation:** prototype with **(A)** (one line of code), report with **(B)**
(clean theory, single parameter `θ`, robustness interpretation). Mention (C) as the
literal tail-risk reading.

## 3. How it slots into the code

Minimal, surgical changes — the geometry, windowed union, and `log τ` marginal all stay.

- **`satgame/game.py`** — today the per-candidate score is `marginal(i,j) − α·slew`.
  Add a `risk` mode where the score becomes the risk-adjusted contribution above.
  This needs `p_ij`, which means computing the marginal **from `j`'s side too**
  (`Δ_j(i)`) — the same `marginal()` helper, swapping the ego center. Add a
  `reciprocation_prob(i, j)` helper.
- **Mechanism switch** — to let risk actually bite, add a `--realize {matching,oneshot}`
  flag. `oneshot`: each satellite commits its top-k by risk-adjusted score; links form
  only on mutual commits (no deferred acceptance). `matching` stays the risk-free
  benchmark.
- **`config.py`** — new fields `risk_measure ∈ {none, meanvar, entropic}`, `risk_param`
  (`λ` or `θ`), `recip_sharpness` (`γ`), `realize`.
- **`run.py` / metrics** — add two metrics that make risk visible:
  - **reciprocation-failure rate** = fraction of committed pointings that formed no
    link (risk-aversion should drive this down);
  - **downside connectivity** = CVaR / worst-epoch `log τ` across seeds (risk-aversion
    should lift the floor).
- **Baselines/plots** — add `game-risk` alongside `game`, `furthest`, `centralized`.

## 4. Game-theoretic caveat (an honest open item)

Risk-sensitive utilities are nonlinear transforms of the additive objective, so the
**exact potential structure likely does not survive** — we lose the clean
Nash-existence-via-potential and finite-improvement convergence. Two defensible routes:

1. **Small-`θ` / `λ` perturbation:** the risk term is an `O(θ)` perturbation of the
   exact potential, giving an **ε-potential game** — approximate equilibria exist and
   best response approximately converges. State the bound in terms of `θ` and the
   marginal range.
2. **Incomplete-information game:** treat `p_ij` as beliefs and lean on the
   partial-information NE-seeking result already cited (Salehisadaghiani–Pavel), which
   needs no potential.

This is a feature, not a bug — it is exactly the kind of analysis the course rewards.

## 5. Experiment that answers the feedback directly

Sweep the risk parameter (`θ` or `λ`) from risk-neutral to risk-averse and plot the
**risk–return frontier**: mean `log τ` (x) vs downside `log τ` / reciprocation-success
(y). The expected story — and the one-line answer to Prof. Ratliff — is that a little
risk aversion **sacrifices a small amount of average connectivity for a large
improvement in the worst case**, i.e. it hedges against the catastrophic
mis-selection she flagged. Overlay the matching policy (risk-free ceiling on
reciprocation) and centralized (connectivity ceiling) for context.

## 6. One-paragraph reply you could send her

> Thanks — this is exactly right under our original one-terminal model, where a
> mis-prediction wastes the only terminal and can isolate a satellite. We've since
> generalized to k terminals and added a matching mechanism that removes
> reciprocation failure by coordination, but we agree the more realistic and
> interesting decentralized policy keeps reciprocation uncertain. We're adding a
> risk-sensitive selection rule (entropic / CVaR over the reciprocation-uncertain
> connectivity gain) so satellites prefer partners that are likely to reciprocate,
> and we'll report the risk–return frontier against the matching and centralized
> benchmarks. Would love to chat.
