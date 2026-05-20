# Notation

This document lists the canonical symbols used in the paper and the code names
they map to. Keep them aligned — if you change one, change the other.

## Sets and indices

| Paper                       | Code                  | Meaning                                       |
|-----------------------------|-----------------------|-----------------------------------------------|
| $\mathcal{N}$               | `nodes`               | Set of satellites, indexed $1..n$             |
| $n$                         | `n`                   | Number of satellites                          |
| $t$                         | `t`                   | Discrete epoch                                |
| $\Delta t$                  | `dt_s`                | Epoch length in seconds                       |
| $\mathcal{F}(t)$            | `F[t]`                | Feasibility set at epoch $t$                  |
| $\mathcal{F}_i(t)$           | `F_i[t, i]`           | Feasible neighbors of $i$ at epoch $t$        |
| $\mathcal{F}^\cup(t; T_0)$  | `feasibility_union(F, t, T_0)` | Feasibility-union graph over window  |
| $L^\cup_\mathcal{F}(t; T_0)$ | `union_F_laplacian`   | Laplacian of feasibility union                |
| $a_i(t)$                    | `actions[t, i]`       | Action of $i$ at epoch $t$                    |
| $e_{ij}(t)$                 | `edges[t, i, j]`      | 1 if link $(i,j)$ realized at epoch $t$       |
| $\mathcal{G}(t)$            | `G[t]`                | Realized communication graph                  |
| $L(t)$                      | `L[t]`                | Graph Laplacian of $\mathcal{G}(t)$           |
| $\mathcal{G}^\cup(t; T)$    | `realized_union_graph` | Realized union graph over the last $T$ epochs |
| $L^\cup_\mathcal{G}(t; T)$  | `union_G_laplacian`   | Laplacian of realized union                   |
| $\Phi(t)$                   | `Phi[t]`              | Windowed Laplacian (sum of recent $L(s)$)     |
| $\lambda_2(\cdot)$          | `lambda_2(...)`       | Algebraic connectivity                        |
| $\Omega(\cdot)$             | `kirchhoff_index(...)` | Kirchhoff index (effective resistance sum)   |

## Policy

| Paper                          | Code                  | Meaning                                          |
|--------------------------------|-----------------------|--------------------------------------------------|
| $V_i(j; t, a_{-i})$            | `_value(i, j, t)`     | Raw value: marginal decrease in $\Omega(\Phi_\varepsilon)$ |
| $\tilde V_i(j; t)$             | computed inline       | Normalized value, in $[0, 1]$, per-satellite max |
| $C^\mathrm{switch}_{ij}(t)$    | `_switching_cost(i, j, t)` | Raw switching cost (angular slew in rad)    |
| $\tilde C^\mathrm{switch}_{ij}(t)$ | computed inline    | Normalized switching cost, in $[0, 1]$ (raw/$\pi$) |
| $p_{ij}(t)$                    | `_reciprocation_prob(i, j, t)` | Reciprocation prediction in $\{0, 1\}$  |
| $H$                            | `params.H`            | Lookahead horizon (epochs)                       |
| $T$                            | `params.T`            | Certificate window length (epochs)               |
| $T_0$                          | `T_0`                 | Feasibility window length (epochs)               |
| $c$                            | `params.switching_cost_scale` | Switching-cost scale, in $[0, 1]$         |
| $\varepsilon$                  | `params.epsilon_geometric_prior` | Geometric-prior weight (default 0.01) |

## Evaluation graph

The policy ranks candidates against an *internal evaluation graph*:
$$\Phi_\varepsilon(t) := \Phi(t) + \varepsilon \cdot L^\cup_\mathcal{F}.$$
The $\varepsilon L^\cup_\mathcal{F}$ term encodes common-knowledge orbital
geometry as a soft prior. The certificate (Theorem 6) is on the realized
union graph $\mathcal{G}^\cup$, which does *not* include $\varepsilon$.

## Decision rule

| Paper symbol | Code expression |
|--------------|------------------|
| $a_i(t) = \arg\max_j p_{ij}(t) \big[\tilde V_i(j;t) - c \tilde C^\mathrm{switch}_{ij}(t)\big]$ | `decide_for(i, t)` |

Per-satellite normalization is applied at the decision boundary:
$\tilde V_i(j;t) = V_i(j;t) / \max_{k \in \mathcal{F}_i(t)} V_i(k;t)$ and
$\tilde C^\mathrm{switch}_{ij}(t) = \angle(\theta_i, \hat\theta_{ij}) / \pi$.

## Analysis

| Paper                       | Code                  | Meaning                                       |
|-----------------------------|-----------------------|-----------------------------------------------|
| $\Gamma_t$                  | `gamma_t`             | Per-epoch matching game                       |
| $W_t(a)$                    | `potential(t, a)`     | Exact potential function                      |
| $a^\star_t$                 | `a_star[t]`           | Equilibrium of $\Gamma_t$                     |
| $\alpha_0$                  | `alpha_0`             | Persistent feasibility constant               |
| $\rho_\mathrm{match}$       | `rho_match`           | Matching-constraint efficiency                |
| $\rho_\mathrm{cover}$       | `rho_cover`           | Feasibility-coverage efficiency               |
| $\rho$                      | `rho`                 | $\rho_\mathrm{match} \cdot \rho_\mathrm{cover}$ |

## Certificate

The closed-loop guarantee (Theorem 6) is

$$\lambda_2\!\big(L^\cup_\mathcal{G}(t; T)\big) \geq \rho \cdot \alpha_0 \quad \text{for } t \geq T - 1.$$

Note: the certificate is on $\lambda_2$ of the **realized union graph**, not
on $\lambda_2(\Phi)$ or on $\Omega(\Phi)$. $\Phi$ is the policy's internal
smoothing object; $\Omega$ is the value-function metric. The certificate
itself is in terms of the realized matchings' joint-connectivity
$\lambda_2$, per Jadbabaie-Lin-Morse (2003).