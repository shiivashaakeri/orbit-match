# Notation

Symbol glossary for the project, kept in sync with `paper/predictive_matching_acc.tex`
and the code. The paper introduces most of these in §II–§IV; this doc
adds entries for symbols that came up in implementation or empirical
analysis but don't all appear in the paper.

Last revised: 2026-05-20

---

## Sets and graphs

| Symbol | Meaning |
|---|---|
| $\mathcal{N}$ | The set of satellites, $\{1, \ldots, n\}$. |
| $n$ | Number of satellites. Small Walker: 24. Medium: 60. |
| $\mathcal{F}(t)$ | Feasibility graph at epoch $t$. Edge $(i,j) \in \mathcal{F}(t)$ iff the pair is geometrically reachable, in range, and within laser slew rate. |
| $\mathcal{F}_i(t)$ | Feasible-neighbor set of satellite $i$ at epoch $t$. |
| $\mathcal{G}(t; a)$ | Realized communication graph at epoch $t$ under joint action $a$. Edge $(i,j) \in \mathcal{G}$ iff $a_i = j$ and $a_j = i$. |
| $\mathcal{F}^\cup(t; T_0)$ | Feasibility-union graph: $(\mathcal{N}, \bigcup_{s = t}^{t+T_0-1} \mathcal{F}(s))$. |
| $\mathcal{G}^\cup(t; T)$ | Realized-union graph: $(\mathcal{N}, \bigcup_{s = t}^{t+T-1} \mathcal{G}(s))$. |
| $L^\cup_\mathcal{F}(t; T_0)$ | Laplacian of $\mathcal{F}^\cup(t; T_0)$. |
| $L^\cup_\mathcal{G}(t; T)$ | Laplacian of $\mathcal{G}^\cup(t; T)$. The certificate is on this Laplacian's $\lambda_2$. |
| $\Phi(t)$ | Windowed Laplacian: rolling sum of $T$ recent realized-matching Laplacians. The policy's internal smoothing surrogate; NOT the certified object. |
| $\Phi_\varepsilon(t)$ | $\Phi(t) + \varepsilon L^\cup_\mathcal{F}$. Evaluation graph for the Kirchhoff value function; encodes the geometric prior. |

---

## Actions and policies

| Symbol | Meaning |
|---|---|
| $a_i(t)$ | Action of satellite $i$ at epoch $t$, in $\mathcal{F}_i(t) \cup \{\varnothing\}$. |
| $\varnothing$ | The "do not request" action. Wins ties in the argmax. |
| $a_{-i}$ | The joint action of all satellites except $i$. |
| $a^{(r)}$ | The joint action profile at BR round $r$ within a single epoch. $a^{(0)}$ is the level-0 profile. |
| $\mathcal{I}_i(t)$ | Information state of satellite $i$ at epoch $t$. Common-knowledge geometry plus $i$'s past actions and link outcomes. |
| $\pi_i: \mathcal{I}_i \to \mathcal{F}_i(t) \cup \{\varnothing\}$ | A distributed policy for satellite $i$. |
| $\Gamma_t$ | The epoch-$t$ matching game: $(\mathcal{N}, \{\mathcal{A}_i(t)\}, \{u_i^t\})$. |
| $\mathcal{A}_i(t)$ | $i$'s action set in $\Gamma_t$. Equals $\mathcal{F}_i(t) \cup \{\varnothing\}$. |
| $u_i^t(a)$ | $i$'s utility in $\Gamma_t$: $V_i(a_i; t, a_{-i}) - C^\text{switch}_{i, a_i}(t)$. |
| $W_t(a)$ | Exact potential function of $\Gamma_t$ (Lemma 2). Sum of $\lambda_2(\Phi)$ over the lookahead minus separable switching cost. |
| $\text{BR}_i(a_{-i})$ | $i$'s best-response operator: $\arg\max_j u_i^t(j, a_{-i})$. |

---

## Decision rule and reciprocation

| Symbol | Meaning |
|---|---|
| $V_i(j; t, a_{-i})$ | Raw value of edge $(i, j)$ at epoch $t$: lookahead-summed Kirchhoff drop in $\Phi_\varepsilon$ from adding the edge. |
| $\tilde V_i(j; t)$ | Per-satellite normalized value: $V_i(j; t) / \max_k V_i(k; t)$. In $[0, 1]$. |
| $C^\text{switch}_{ij}(t)$ | Switching cost from $i$'s current pointing direction to the bearing toward $j$ at epoch $t$. Angular, in radians. |
| $\tilde C^\text{switch}_{ij}(t)$ | Normalized switching cost: $C^\text{switch}_{ij}(t) / \pi$. In $[0, 1]$. |
| $c$ | Switching-cost scale. Currently $c = 0.2$. Range $[0, 1]$. |
| $p_{ij}(t)$ | Reciprocation prediction. In principle in $[0, 1]$; in our implementation the one-step BR indicator (eq. 17), so $\in \{0, 1\}$. |
| $H$ | Lookahead horizon (in epochs) for the value function. Currently $H = 10$. |
| $T$ | Certificate window (in epochs). Currently $T = T_\text{orb}$. |
| $T_0$ | Persistent-feasibility window (Assumption 5). The minimum window over which the feasibility-union must be connected. |
| $\varepsilon$ | Geometric-prior weight in $\Phi_\varepsilon = \Phi + \varepsilon L^\cup_\mathcal{F}$. Currently $\varepsilon = 0.01$. |

---

## Certificate quantities

| Symbol | Meaning |
|---|---|
| $\alpha_0$ | Persistent-feasibility constant (Assumption 5): $\min_t \lambda_2(L^\cup_\mathcal{F}(t; T_0))$. Small: 0.92. Medium: 4.29. |
| $\rho$ | Efficiency ratio (Theorem 6): $\rho = \rho_\text{match} \cdot \rho_\text{cover}$. |
| $\rho_\text{match}$ | Matching-constraint efficiency: realized union edges per epoch vs the maximum matching size in $\mathcal{F}(t)$. Bounded by $T / (\Delta + 1)$ (lower) and $T / \Delta$ (upper) via Vizing. **Vacuous in our regime** ($T \gg \Delta$ → both bounds clip to 1; F9). |
| $\rho_\text{cover}$ | Coverage efficiency: realized-union edges / feasibility-union edges. The improvable factor. |
| $\rho_\text{realized}$ | Empirical observed ratio $\lambda_2(L^\cup_\mathcal{G}(t; T)) / \alpha_0$. Reported as final, max, and mean over post-warmup epochs. |
| $\Delta$ | Maximum vertex degree of $\mathcal{F}^\cup(t; T_0)$. Currently $\Delta \in \{6, 7\}$. |

---

## Recursion depth (level-$k$ family)

These symbols are *new* — they don't appear in the paper as written.
The paper's §IV.D Remark mentions "$k$-step predictors" without naming
the parameter. We use $k$ explicitly because we ablate over it.

| Symbol | Meaning |
|---|---|
| $k$ | **Recursion depth in strategy.** Number of rounds of BR dynamics applied within a single epoch. NOT the lookahead horizon $H$. |
| $k = 0$ | Level-0: every satellite picks its top-V partner ignoring reciprocation. Equivalent to `greedy`. |
| $k = 1$ | One BR round against the level-0 profile. The paper's `predictive` policy. |
| $k = \infty$ | BR iterated to a fixed point. Under Gauss-Seidel update order, this is a NE of $\Gamma_t$. |
| $\delta$ | Gap threshold for the adaptive policy. Each satellite escalates to $k_\max$ if its top two normalized scores differ by less than $\delta$. Currently $\delta = 0.1$. |
| $k_\max$ | Maximum recursion depth in the adaptive policy. Currently $k_\max = 3$. |

---

## Two horizons (don't conflate)

This deserves its own callout because in conversation it's easy to mix up.

| Symbol | What it controls | Where it appears |
|---|---|---|
| $H$ | **Time horizon** of the value function lookahead. How many future epochs $i$ sums Kirchhoff drops over. | §III.B of paper, `PolicyParams.H` in code. |
| $k$ | **Strategic recursion depth** at the current epoch. How many rounds of "I think you think I think..." | §IV.D Remark of paper, `KStepPredictive.k` in code. |
| $T$ | **Certificate window length.** Window over which the realized-union Laplacian is evaluated. | §II.D of paper. Distinct from both $H$ and $k$. |
| $T_0$ | **Persistent-feasibility window length.** Window over which the feasibility-union must be connected. | Assumption 5 of paper. Typically $T_0 \leq T$. |

All four are in epochs. $H$ and $k$ are policy hyperparameters; $T$
and $T_0$ are theorem parameters.

---

## Empirical metrics (reported in `DiagnosticsReport`)

| Symbol | Code field | Meaning |
|---|---|---|
| $\rho_\text{realized}^\text{final}$ | `rho_realized_final` | $\lambda_2(L^\cup_\mathcal{G}(T_\text{end}; T)) / \alpha_0$ |
| $\rho_\text{realized}^\text{max}$ | `rho_realized_max` | $\max_{t \geq T} \rho_\text{realized}(t)$ |
| $\bar\rho_\text{realized}$ | `rho_realized_mean` | $\frac{1}{n_\text{post-warmup}} \sum_{t \geq T} \rho_\text{realized}(t)$ |
| $\rho_\text{cover}^\text{final}$ | `rho_cover_final` | edges($\mathcal{G}^\cup$) / edges($\mathcal{F}^\cup$) at $t = T_\text{end}$ |
| — | `requests_per_epoch` | $(a \neq \varnothing)$.sum() / $n_\text{epochs}$ |
| — | `edges_per_epoch` | mean of $\|\mathcal{G}(t)\|$ |
| — | `waste_per_epoch` | requests minus $2 \times$ edges (each edge $=$ two satisfied requests) |
| — | `waste_pct` | 100 $\cdot$ waste / requests |
| — | `ambiguous_frac` (adaptive only) | fraction of satellites escalating beyond $k=1$ at each epoch |
| — | `br_rounds` (equilibrium only) | number of Gauss-Seidel sweeps to convergence at each epoch |
| — | `deferrals` (predictive only) | number of satellites that played $\varnothing$ at each epoch |

---

## Tools

| Symbol | Code class | Meaning |
|---|---|---|
| $\Phi$ | `WindowedLaplacian` | Rolling-sum Laplacian over the last $T$ matchings. |
| $\mathcal{G}^\cup$ | `WindowedUnion` | Rolling-union adjacency over the last $T$ matchings. |
| — | `PredictiveMatching` | Implements $k = 1$, the paper's policy. |
| — | `EquilibriumMatching` | Gauss-Seidel BR to convergence, warm-start from level-1. |
| — | `KStepPredictive` | Family parametrized by $(k, \text{mode})$. Boundary cases match `greedy`, `predictive`. |
| — | `AdaptivePredictive` | Selective escalation policy, parametrized by $(\delta, k_\max)$. |