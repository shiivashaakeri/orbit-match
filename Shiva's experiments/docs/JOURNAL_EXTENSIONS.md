# Journal Extensions

A catalog of experiments that would strengthen the theory beyond what
the ACC paper presents. None of these are in the ACC submission; all
are reserved for a journal extension (e.g., IEEE T-AC, T-CST) or
follow-up paper.

Last revised: 2026-05-21

---

## Why this document exists

The ACC paper is tight: 6 pages, 3 policies, 2 figures. The full
empirical surface of the theory is much larger. Today's exploration
(documented in `EXPERIMENTS_LOG.md` 2026-05-21 entries) demonstrated
that the policy is at a Pareto frontier on Walker geometry, but it
didn't exhaust the experimental space. This document lists what we'd
do with more pages and more time.

The list is grouped by purpose. Each entry includes a one-line
description, an effort estimate, and the paper-value of the result.

---

## 1. Negative controls

Show that the certificate fails when the theorem's assumptions are
violated. Strengthens the theorem by showing the assumptions are
*necessary*, not just sufficient.

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 1a | Sparse constellation where some pairs never become mutually feasible. Show $\alpha_0 \to 0$ and consensus stalls. | Low | High — directly demonstrates persistent feasibility matters |
| 1b | Take $T < T_0$ (certificate window shorter than the persistent-feasibility window). Show $\rho_{\text{realized}}$ drops below 1. We have some preliminary data on this for small Walker. | Low | Medium — establishes that $T \geq T_0$ is tight, not slack |
| 1c | Compare to a non-matching policy that violates the mutual-choice rule (e.g., allow each satellite to point at multiple partners). Show predictive's certificate property is special to the matching structure. | Medium | Medium — shows matching is not arbitrary; the value function and the constraint cooperate |

## 2. Theorem tightness

Probe how loose the bound $\rho \geq \rho_{\text{match}} \cdot \rho_{\text{cover}}$ is in practice.

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 2a | Plot $\rho_{\text{realized}}$ versus the lower bound across many geometries. Quantify the gap. We expect the bound to be loose; this experiment shows by how much. | Medium | High — could lead to a tighter bound |
| 2b | Construct a regime where Vizing-bound on $\rho_{\text{match}}$ is tight (small $T$, dense feasibility) and show the policy's $\rho_{\text{match}}$ tracks Vizing prediction. | Medium | Medium — validates one of the bound's two factors |

## 3. Scaling

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 3a | $\rho_{\text{realized}}$ versus $n$ (Walker-24, -60, -120, -240). | Medium-high | High — establishes the result at scale |
| 3b | Different orbital regimes: polar, Walker-star, highly elliptical. | High | Medium-high — broadens claim beyond Walker-Delta |
| 3c | Wall-clock per epoch versus $n$. Confirms distributed scaling. | Low | Medium — addresses computational practicality |

## 4. Robustness

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 4a | Satellite dropout: remove one satellite per orbital period; show $\rho_{\text{realized}}$ recovers within $T$. | Low | High — self-healing property |
| 4b | Geometry perturbations: add ~km-scale noise to positions; show graceful degradation. | Low | Medium — robustness to model uncertainty |
| 4c | Subset of satellites has slower slew rate (hardware failure). | Medium | Medium — heterogeneous capability case |

## 5. Downstream task validation

The corollary to Theorem 6 (consensus convergence) is one example of a
downstream task. The corollary is more general; it covers any
algorithm that depends on joint connectivity.

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 5a | Distributed estimation: each satellite has a noisy observation of a global quantity; track variance reduction over time. | Medium | High |
| 5b | Distributed time synchronization: clocks with drift; show clocks synchronize at the predicted rate. | Medium | High — motivates the formation control / GNSS application |
| 5c | Distributed Kalman filtering for a target. | High | High |

## 6. Comparison with state-of-the-art

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 6a | Compare predictive to scheduled contact plans (centralized, precomputed). | Medium-high | High — shows we match scheduling without the planning |
| 6b | Compare to other distributed link-formation algorithms (Roth stable matching, auction-based matching, gossip). | High | High — positioning vs literature |

## 7. Sensitivity analysis

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 7a | Lookahead horizon $H$ sweep across constellation sizes. | Low-medium | Medium |
| 7b | Switching-cost scale $c$ sweep. | Low | Low |
| 7c | Geometric-prior weight $\varepsilon$ sweep. | Low | Low |

## 8. Model extensions

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 8a | Time-varying switching cost: slew cost depends on orbital velocity. | Medium | Low-medium |
| 8b | Asymmetric value: some satellites are more important. Modify $V$ to be asymmetric. | Medium | Medium |
| 8c | Multi-laser satellites: each satellite has $k > 1$ lasers (degree constraint, not matching). | High | High — significant extension |

## 9. Coordination-augmented policies (the journal version of today's exploration)

The ACC paper's §V.E notes that no purely-distributed modification
Pareto-dominates baseline (F17). The natural extension: introduce
small amounts of coordination (broadcast bandwidth) and characterize
the resulting trade-off curve.

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 9a | Sequential predictive with broadcast (k1_gs): formally introduce the broadcast assumption. Show zero waste, NE in one sweep. Already implemented; ready to write up. | Low | High |
| 9b | The Pareto frontier we discovered today: plot (waste, coverage) across the (H, $\beta$, $\gamma$) grid as a quantitative frontier. | Low | Medium |
| 9c | Coordination-minimal protocols: relate broadcast bandwidth to position on the frontier. How much coordination buys you how much. | Medium-high | High |

## 10. Theoretical-empirical bridges

| ID | Experiment | Effort | Paper value |
|----|------------|--------|-------------|
| 10a | Olshevsky-Tsitsiklis envelope: compute the explicit consensus decay rate from the cited paper for our $\rho \alpha_0$, $T$, $\mu$. Overlay on Fig 2. | Low-medium | Medium — quantitative validation of the corollary |
| 10b | Empirical decomposition of $\rho$ into $\rho_{\text{match}}$ and $\rho_{\text{cover}}$ across geometries. | Medium | Medium |

---

## Notes

- Effort estimates are rough engineer-days: "Low" = 1 day, "Medium" =
  3-5 days, "High" = 1-2 weeks, "High+" = multi-week effort.
- Paper-value is subjective; "High" means strengthens the headline
  claim, "Medium" means strengthens supporting claims, "Low" means
  nice-to-have.
- Several experiments could be combined into a single figure or table
  (e.g., 1a + 1b in one negative-controls table, 5a-5c as a single
  "downstream tasks" subsection).
- For a 12-page journal version, the highest-value combinations are
  probably: 1a + 9a + 5a-5c (theory completeness + practical
  coordination + applications).
