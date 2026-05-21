# Findings

Empirical conclusions from the simulation work that affect the paper.
Unlike `EXPERIMENTS_LOG.md` (which is dated per-run), this document is
a running list of *what we believe to be true* about the model and the
policy, supported by the data so far. Update when new findings change
the picture; do not delete (mark superseded if needed).

Last reviewed: 2026-05-20

---

## F1. The connectivity certificate saturates at $T = T_{\text{orb}}$

**Claim.** When the certificate window equals one orbital period, the
predictive policy realizes every edge that the feasibility geometry
admits over that window. Empirically, $\rho_{\text{realized}} = 1$ at
$T = T_{\text{orb}}$ on both small (24/4/1) and medium (60/6/2) Walker
constellations.

**Implication for the paper.** Theorem 6's bound
$\rho_{\text{realized}} \geq \rho \cdot \alpha_0$ is conservative for
these constellations: $\rho_{\text{match}} \cdot \rho_{\text{cover}}$
multiplied gives a lower bound, but the realized value saturates well
above it. The paper should report the saturation result as the
headline, the lower bound as the guarantee.

**Caveat.** Saturation holds *at* $T = T_{\text{orb}}$. After the
window rolls forward, the realized union sheds edges that the policy
correctly deferred from re-realizing (the deferral mechanism doing
exactly what it should). At $t > T_{\text{orb}}$ the realized union
covers ~88% of the feasibility union; the missing edges are recovered
once the geometry rolls them back into the next window. See F3.

**Source.** EXPERIMENTS_LOG 2026-05-20 saturation entry.

---

## F2. Predictive and greedy realize identical matchings on Walker constellations

**Claim.** With $p_{ij}$ the one-step BR indicator (eq. 17), the
predictive policy and the greedy policy ($p_{ij} \equiv 1$) produce
the *same realized matchings on every epoch*. Both achieve
$\rho_{\text{realized}} = 1.0$ peak and $\rho_{\text{mean}} = 0.9855$
on medium Walker.

**Why.** Under the mutual-choice rule, edge $(i, j)$ forms iff both
$i$ requests $j$ AND $j$ requests $i$. Greedy: $i$ requests its top-V
partner $j^\star$, and $j^\star$ requests its own top-V partner
$k^\star$. The edge forms iff $k^\star = i$, i.e., the two are mutual
top-V choices. Predictive: $p_{ij^\star} = 1$ iff $j^\star$'s top-V
is $i$, so $i$ requests $j^\star$ only when this mutual top-V
condition holds. The set of mutual-top-V pairs is the same in both
cases, so the realized matchings agree.

**Implication for the paper.** The deferral mechanism's value is
*not* in $\lambda_2$. Greedy reaches the same certificate. The
deferral mechanism's value is in **request efficiency** (F4), which
should be the framing in Section V.

**Source.** EXPERIMENTS_LOG 2026-05-20 request-efficiency entry.

---

## F3. The rolling-window union decays after $T_{\text{orb}}$

**Claim.** $\rho_{\text{cover}}$ drops from 1.0 at $t = T_{\text{orb}}$
to ~0.88 by $t = 1.5 T_{\text{orb}}$ on small Walker. The policy
correctly chooses not to re-request already-saturated edges in the
window.

**Why.** Once the rolling certificate window contains the full
feasibility union, the policy has no marginal-value incentive to
re-realize those edges (the value function gives zero drop for
already-present edges). The deferral mechanism takes over. As epochs
roll forward, edges from the *earliest* part of the window age out,
and the union shrinks even though feasibility per-epoch has not
changed.

**Implication for the paper.** This is the deferral mechanism working
as designed, not a regression. Worth noting in Section V text: the
"saturation" claim is at $t = T_{\text{orb}}$, not at all future $t$.
The certificate theorem still holds (every rolling window includes a
full feasibility cycle), just with $\rho_{\text{cover}} < 1$ in
steady state.

**Source.** `scripts/check_diagnostics.py` test 4; EXPERIMENTS_LOG
2026-05-20 saturation entry.

---

## F4. Request efficiency is where the deferral mechanism pays off

**Claim.** Per epoch on medium Walker:

| policy       | requests | edges | waste | waste% |
|---|---:|---:|---:|---:|
| predictive   | 38.11    | 6.79  | 24.53 | 64.4%  |
| greedy       | 60.00    | 6.79  | 46.42 | 77.4%  |
| random       | 60.00    | 4.20  | 51.61 | 86.0%  |
| equilibrium  | 32.92    | 16.46 |  0.00 |  0.0%  |

Predictive makes ~37% fewer requests than greedy for the same realized
matching. Equilibrium makes ~45% fewer requests than greedy *and*
realizes 2.4× more edges per epoch, with zero waste.

**Why.** Greedy requests every top-V partner regardless of
reciprocation. ~46 of those per epoch fail. Predictive only requests
when its one-step indicator says the partner will reciprocate. Many
satellites correctly self-suppress. But the one-step prediction is
not perfect (F5), so 64% of predictive's requests still fail; the
mutual-choice rule eats those.

**Implication for the paper.** Section V should report request and
waste counts alongside connectivity metrics. The paper's headline
claim should not be "predictive beats greedy on $\lambda_2$" (it
doesn't — F2) but "predictive achieves the certificate with 37%
fewer requests."

**Source.** EXPERIMENTS_LOG 2026-05-20 request-efficiency entry.

---

## F5. One-step reciprocation prediction is right ~36% of the time

**Claim.** Predictive's $p_{ij}$ is the one-step BR indicator: $i$
predicts $j$ will reciprocate iff $j$'s top-V choice (ignoring
reciprocation) is $i$. Empirically, when predictive sends a request
under $p_{ij} = 1$, it's right 35.6% of the time (14 successes per
38 requests, on medium).

**Why.** $i$ models $j$ as a *value-maximizer*, i.e., as if $j$ ran
without its own reciprocation logic. But $j$'s actual decision does
include reciprocation, so $j$'s actual top choice may differ from
$j$'s top-V choice. When $i$ and $j$ disagree on each other's
behavior, neither's prediction matches the other's actual action.

**Implication for the paper.** This is Proposition 8's empirical
content: predictive is one round of BR, not a fixed point. It does
not converge to NE of $\Gamma_t$ at each epoch; the gap is real and
quantifiable. Section V should report this 36% number as a witness
that the one-step truncation costs something operationally.

**Source.** EXPERIMENTS_LOG 2026-05-20 request-efficiency entry.

---

## F6. Synchronous BR is a poor coordination protocol; Gauss-Seidel BR is better

**Claim.** When iterating the level-$k$ family from a level-0
initialization on $\Gamma_t$, synchronous BR (Jacobi) hits a fixed
point in one round. Subsequent rounds (k = 2, 3, 8) produce identical
matchings, identical metrics. Gauss-Seidel BR finds genuinely better
fixed points (denser matchings, lower waste).

**Why.** Both modes are valid BR dynamics, but they have different
basins of attraction. Synchronous BR with all-at-once updates lands
on a *symmetric* fixed point that's typically shallow (close to
level-0). Gauss-Seidel breaks ties through the update ordering, which
amounts to a small extra signal that lets the iteration walk farther
into the potential's interior.

**Implication for the paper.** The paper's Sec IV.D Remark ("k-step
predictors implement k rounds, in the limit converging to NE") should
be qualified: it's true for Gauss-Seidel BR, false for synchronous BR.
Worth a sentence in §IV.D and a paragraph in §V around Figure 3 (the
k-ablation).

**Source.** EXPERIMENTS_LOG 2026-05-20 k-ablation entry.

---

## F7. Adaptive depth: most decisions are unambiguous

**Claim.** On medium Walker with $\delta = 0.1$, only 13.83% of
satellites per epoch face an "ambiguous" decision (gap < $\delta$
between top two scores). The other 86% face an unambiguous decision
and can safely commit at $k = 1$.

**Implication for the paper.** The level-$k$ family has a *structural
sparsity*: most decisions are clear-cut, and only a small fraction
benefit from deeper reasoning. An adaptive policy can exploit this to
get near-equilibrium quality at near-$k=1$ cost.

**Source.** EXPERIMENTS_LOG 2026-05-20 k-ablation entry.

---

## F8. The adaptive policy captures most of equilibrium's benefit at 28% extra cost

**Claim.** Adaptive ($\delta = 0.1$, $k_{\max} = 3$):

- 36.51 requests/epoch (vs 38.11 for $k=1$)
- 10.67 edges/epoch (vs 6.79 for $k=1$, 16.46 for equilibrium — 57% more than k=1, 65% of equilibrium)
- 41.6% waste (vs 64.4% for $k=1$, 0% for equilibrium)
- $\rho_{\text{cover}}$ = 0.9843 (matches equilibrium exactly)
- $\rho_{\text{mean}}$ = 0.9980 (very close to equilibrium's 0.9993)
- Cost: 1.28× $k=1$

**Why.** The adaptive policy doesn't just selectively recurse; it also
**warm-starts from the level-1 profile** before iterating the
ambiguous subset. This warm-start is closer to a good fixed point
than level-0, which is what makes the small recursion budget so
effective.

**Implication for the paper.** This is the operationally most useful
variant in the paper's family. Even if equilibrium is the
theoretical ceiling, adaptive is the deployable middle ground. Could
be the §VI conclusion's forward-looking statement.

**Source.** EXPERIMENTS_LOG 2026-05-20 k-ablation entry.

---

## F9. The Vizing bound on $\rho_{\text{match}}$ is vacuous in our regime

**Claim.** The paper bounds $\rho_{\text{match}} \leq T / \chi'$
where $\chi'$ is the edge chromatic number of the feasibility union,
and Vizing's theorem says $\chi' \in \{\Delta, \Delta + 1\}$. For our
constellations $\Delta = 6$ (small) or higher, and $T = T_{\text{orb}}
\geq 60$. So the bound $T / \Delta \geq 10 \to$ clipped to 1.0.

**Implication for the paper.** The matching constraint is not the
binding constraint on $\rho$ at $T = T_{\text{orb}}$. All of the
empirical $\rho < 1$ comes from $\rho_{\text{cover}}$, not from the
matching bottleneck. Worth mentioning in the §V text discussing
Theorem 6's decomposition: in this regime, $\rho_{\text{match}}$ is
"free" and the policy's task is purely cover.

**Source.** `orbitmatch/experiments/diagnostics.py`,
`scripts/check_diagnostics.py`.

---

## F10. The potential $W_t$ is monotone within an epoch but not across

**Claim.** Sanity check S4 verified: BR iterations within a single
epoch produce a non-decreasing trace of $W_t(a)$. This is Corollary 3
of the paper, numerically validated.

The same is **not** true *across* epochs. $\lambda_2(L^\cup_\mathcal{G}(t; T))$
can fall as $t$ increases because the rolling window discards old
edges (F3). The empirical $W_t$ from equilibrium tells us nothing
about whether the same equilibrium choice was best for $W_{t+1}$;
there is no across-epoch optimization happening.

**Implication for the paper.** Don't claim across-epoch monotonicity
in §IV. The closed-loop trajectory is *not* a gradient flow on $W_t$
— each $W_t$ is a different function, and even the equilibrium policy
makes choices that decrease the *next* epoch's potential.

**Source.** `scripts/run_sanity_checks.py` S4; corrected
`scripts/check_equilibrium.py` test 5.

---

## Notes

- F1, F4, F5, F8 are paper-bound; they would each appear as numbered
  observations in §V text or a paragraph of §V interpretation.
- F2, F6, F10 are *clarifying* observations: they explain why a naive
  reading of the policy comparison is wrong.
- F3 is a footnote-level remark that prevents reviewer confusion.
- F7 motivates the adaptive variant (F8).
- F9 is methodological: tells us *which* of the two factors in
  $\rho = \rho_{\text{match}} \cdot \rho_{\text{cover}}$ matters.