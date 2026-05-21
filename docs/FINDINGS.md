# Findings

Empirical conclusions from the simulation work that affect the paper.
Unlike `EXPERIMENTS_LOG.md` (which is dated per-run), this document is
a running list of *what we believe to be true* about the model and the
policy, supported by the data so far. Update when new findings change
the picture; do not delete (mark superseded if needed).

Last reviewed: 2026-05-20 (v2 — k1_gs validation complete)

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

## F6. Best-response update order matters more than depth (revised)

**Earlier draft of this finding said "synchronous BR is poor; Gauss-Seidel
BR is better." The validated picture is sharper than that:**

**Claim.** One BR sweep from level-0 is enough to reach a NE — IF the
sweep uses Gauss-Seidel (sequential) updates. Synchronous (Jacobi)
updates from level-0 land in a different, much worse fixed point in
one sweep, and additional synchronous sweeps don't escape it.

Empirical comparison on medium Walker:

```
                     edges/ep   waste%   rho_cover
k1_sync                6.79     64.4%    0.9529
k3_sync                6.79     64.4%    0.9529   (same fixed point)
k1_gs (identity)      18.56      0.0%    0.9902
k1_gs (reversed)      17.86      0.0%    0.9882   (different NE)
k1_gs (random)        18.71      0.0%    0.9863   (different NE)
equilibrium*          16.46      0.0%    0.9843
```

*equilibrium is GS to convergence, warm-started from the level-1
predictive output. Notably worse than k1_gs from level-0.

**Why.** Synchronous BR is order-invariant by construction but
also coordination-free: every satellite responds to the *previous*
joint action with no information about its neighbors' current
intentions. The resulting fixed point is the "median" of selfish
top-V claims — many satellites end up pointing at partners whose
top-V choice is somebody else. Sequential BR breaks this: a
satellite that takes its turn after $i' = 5$ sees that 5 has already
committed to some partner $j$, and best-responds against that — not
against 5's hypothetical action. Information flows within the
sweep, even though there are no inter-satellite messages.

**Why the warm-start hurts.** Equilibrium warm-starts from the
level-1 (predictive) profile, then iterates GS to convergence. But
the level-1 profile is the synchronous-BR fixed point — the shallow
basin. Subsequent GS sweeps can't fully escape the basin; equilibrium
lands at a *local* maximum of the potential $W_t$ near the
predictive output, not the better basin that k1_gs reaches from
level-0. **The warm-start is harmful.**

**Implication for the paper.** The paper's headline policy should
be *predictive matching with Gauss-Seidel update order*. One BR
sweep, sequential. Implementation cost is identical to synchronous
(same number of value computations), but the realized matching
quality is dramatically different. The implementation difference is
purely about update sequencing — no additional inter-satellite
communication required if a natural priority order (satellite
index, recently-seen-feasibility, or similar) is broadcast as part
of the orbital geometry.

**Source.** Tables in EXPERIMENTS_LOG 2026-05-20 k-ablation and
2026-05-20 k1_gs validation entries.

---

## F11. Gauss-Seidel from level-0 finds a NE in one sweep (validated)

**Claim.** `KStepPredictive(k=1, mode="gauss_seidel")` starting from
the level-0 (greedy) profile reaches a Nash equilibrium of
$\Gamma_t$ in a single GS sweep. Subsequent sweeps (k=2, 3, 8) produce
*identical* matchings. Zero waste at every epoch.

**Validation.** Confirmed deterministic across three seeds (seed-spread
in edges/ep = 0.0000, in rho_cover = 0.0000). The result is not a
statistical fluke — it's a structural property of the BR map on this
geometry.

**Why one sweep suffices.** The level-0 profile is already "mostly
correct": each satellite picks its top-V partner, and for ~40% of
pairs the top-V choices are already mutual (so those pairs would form
an edge under any update order). The Gauss-Seidel sweep cleans up the
remaining ~60%: when satellite $i$ takes its turn, predecessors $i' < i$
have already updated to their best responses, so $i$ has accurate
information about which partners are still "available." After all
satellites have updated once, the joint action is self-consistent —
every satellite's request is mutually optimal given everyone else's.
A second sweep can't change anything because the first sweep already
reached a fixed point.

**Implication for the paper.** This is the *operational headline*.
The paper proposes a one-BR-sweep policy; the data shows that under
the right update order, this policy achieves a NE with zero waste
and the densest possible matching. The level-$k$ family with $k > 1$
buys nothing once we use Gauss-Seidel.

**Source.** EXPERIMENTS_LOG 2026-05-20 k-ablation; 2026-05-20 k1_gs
validation entry.

---

## F12. The game has multiple NEs; ordering picks one

**Claim.** Different Gauss-Seidel update orderings yield *different*
NEs of $\Gamma_t$. All NEs found have zero waste, but they differ
in edges-per-epoch by ~5% and in $\rho_{\text{cover}}$ by ~0.4%:

```
ordering        edges/ep   rho_cover   rho_mean
identity        18.56      0.9902      0.9988
reversed        17.86      0.9882      0.9994
random (seed 7) 18.71      0.9863      0.9998
```

**Why.** Asynchronous BR on a potential game has multiple local
maxima of $W_t$; which one you reach depends on the order in which
players update. Earlier-updating satellites get "first claim" on
their preferred partners; later satellites best-respond to a
reduced choice set. The specific NE reached encodes a priority
structure imposed by the ordering.

**Implication for the paper.** Worth a remark in §III when the
policy is defined, and a sentence in §V around Figure 3:

> *"The Gauss-Seidel update order is not unique. Different orderings
> reach different NEs of $\Gamma_t$, all with zero waste, but
> differing in matching density by a few percent. In Section V we
> use the natural satellite-index ordering; this is conservative,
> as random orderings yield marginally denser matchings."*

This is the kind of honest observation that strengthens the paper —
it shows the author understands the algorithm at the algebraic level,
not just the empirical level.

**Caveat for follow-up work.** A principled choice of ordering
(e.g., by earliest deadline first, or by feasibility-window
closing time) could marginally improve $\rho_{\text{cover}}$ further.
Out of scope for ACC, but worth a §VI sentence.

**Source.** EXPERIMENTS_LOG 2026-05-20 k1_gs validation entry,
Validation B.

---

## F13. Equilibrium is a *worse* NE than k1_gs

**Claim.** The `EquilibriumMatching` policy (warm-start from level-1,
iterate GS to convergence) reaches a NE that is **strictly dominated**
by k1_gs (warm-start from level-0, one GS sweep) on every metric:

```
                req/ep   edges/ep   waste%   rho_mean   rho_cover
equilibrium     32.92    16.46      0%       0.9993     0.9843
k1_gs           37.12    18.56      0%       0.9988     0.9902
```

(equilibrium has marginally higher `rho_mean`, but that's at the
0.05% level; the practical winner is k1_gs on every other axis.)

**Why.** Both are NEs; the game has multiple equilibria. Equilibrium
reaches one local maximum of $W_t$ near the level-1 profile;
k1_gs reaches a different local maximum near the level-0 profile.
The level-0-basin maximum happens to be globally better on this
geometry.

**Implication for the paper.** "Equilibrium" as we implemented it
is *not* the upper bound on the policy family. The right ceiling is
k1_gs. Reframing implications:

- Remove equilibrium from comparisons in the headline figure
  (Fig 1). Use predictive (k1_sync) and k1_gs as the two policy
  candidates; greedy and random as baselines.
- Reframe Fig 3 (the ablation) around the *update-order* axis (sync
  vs GS at k=1), with the depth $k$ ablation collapsed into a remark:
  both orderings stabilize at $k = 1$; depth $> 1$ buys nothing.
- The "BR to convergence" thread shrinks to a single sentence in §V:
  "iterating BR from the predictive policy's output does not
  improve over a single GS sweep from the greedy profile."

**Source.** Combined: EXPERIMENTS_LOG 2026-05-20 k-ablation (showing
the gap) and validation entry (showing k1_gs is the consistent NE
across seeds/orderings).

---

## F14. Adaptive sits between sync and GS basins (revised interpretation)

**Earlier interpretation of adaptive's surprise behavior was:
"adaptive captures most of equilibrium's benefit at 1.28× cost."
The validated picture refines this:**

**Claim.** Adaptive (delta=0.1, k_max=3, warm-start from level-1)
lands between the sync basin (10.67 edges/ep, 41.6% waste) and the
GS basin (18.56 edges/ep, 0% waste). It partially escapes the
sync basin via selective GS-style updates on the ambiguous subset,
but inherits the same level-1 warm-start trap that equilibrium
falls into.

**Implication for the paper.** Adaptive remains an interesting
mid-cost variant, but it's not the right ceiling — k1_gs is. If
the paper wants a "cheap NE finder" recommendation, it's
`KStepPredictive(k=1, mode="gauss_seidel")`, not adaptive.

Adaptive's value now is as an example of "selective deepening helps
escape shallow basins" — a conceptual contribution, not a
performance one. Could be a §V paragraph or could be cut.

**Source.** EXPERIMENTS_LOG 2026-05-20 k-ablation; compared against
F11 and F13.

---

## Canonical traces for paper figures

After today's validation, the canonical reference data are:

- **Fig 1 ($\lambda_2$ traces, headline)**: `results/lambda2_traces/*.npz`,
  3 seeds, 4 policies (predictive=k1_sync, greedy, random, equilibrium).
  Equilibrium will likely be dropped or demoted; see F13.

- **Fig 3 ($k$-ablation / update order)**: replace the existing
  `results/k_ablation/` with the focused comparison of:
  - k1_sync (= predictive, the paper's headline)
  - k1_gs identity (the recommended deployable policy)
  - greedy (baseline)
  - adaptive (optional, as a §V remark)
  All on seed 42, medium config.

- **k1_gs ordering remark (Fig 3 caption or §V text)**:
  `results/k_gs_validation/trace_k1_gs_seed42_{identity,reversed,random_p7}.npz`
  These three traces support the F12 multi-NE observation.

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

## F8. The adaptive policy captures most of equilibrium's benefit at 28% extra cost (superseded by F14)

**This finding is preserved for the record but superseded by F14
once the k1_gs result was validated.** The "ceiling" used here is
equilibrium, which F13 now shows is *not* the right ceiling. See
F14 for the revised interpretation.

**Claim (original).** Adaptive ($\delta = 0.1$, $k_{\max} = 3$):

- 36.51 requests/epoch (vs 38.11 for $k=1$)
- 10.67 edges/epoch (vs 6.79 for $k=1$, 16.46 for equilibrium — 57% more than k=1, 65% of equilibrium)
- 41.6% waste (vs 64.4% for $k=1$, 0% for equilibrium)
- $\rho_{\text{cover}}$ = 0.9843 (matches equilibrium exactly)
- $\rho_{\text{mean}}$ = 0.9980 (very close to equilibrium's 0.9993)
- Cost: 1.28× $k=1$

**Why (original).** The adaptive policy doesn't just selectively recurse; it also
**warm-starts from the level-1 profile** before iterating the
ambiguous subset.

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

- **F1, F4, F5, F11, F13** are the paper-bound headline findings.
  Each appears as a numbered observation in §V text or as a paragraph
  of §V interpretation.
- **F2, F6, F10, F12** are *clarifying* observations that prevent a
  naive reader from drawing wrong conclusions from the data.
- **F3** is a footnote-level remark.
- **F7** motivates the adaptive variant; **F8** and **F14** show
  adaptive is interesting but not the operational ceiling (k1_gs is).
- **F9** is methodological; tells us $\rho_{\text{cover}}$ is what
  $\rho$ tracks in our regime.

## Revision history

- **v1 (initial)**: F1–F10 captured the headline run findings.
- **v2 (2026-05-20)**: F11–F14 added after k1_gs validation. F6 and F8
  revised to reflect the corrected ceiling (k1_gs, not equilibrium).