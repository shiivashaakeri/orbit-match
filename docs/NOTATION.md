# Notation

This document lists the canonical symbols used in the paper and the code names
they map to. Keep them aligned — if you change one, change the other.

## Sets and indices

| Paper                 | Code                  | Meaning                                  |
|-----------------------|-----------------------|------------------------------------------|
| $\mathcal{N}$         | `nodes`               | Set of satellites, indexed $1..n$        |
| $n$                   | `n`                   | Number of satellites                     |
| $t$                   | `t`                   | Discrete epoch                           |
| $\Delta t$            | `dt_s`                | Epoch length in seconds                  |
| $\mathcal{F}(t)$      | `F[t]`                | Feasibility set at epoch $t$             |
| $\mathcal{F}_i(t)$    | `F_i[t, i]`           | Feasible neighbors of $i$ at epoch $t$   |
| $\mathcal{F}^\cup(t; T_0)$ | `F_union`        | Feasibility-union graph over window      |
| $a_i(t)$              | `actions[t, i]`       | Action of $i$ at epoch $t$               |
| $e_{ij}(t)$           | `edges[t, i, j]`      | 1 if link $(i,j)$ realized at epoch $t$  |
| $\mathcal{G}(t)$      | `G[t]`                | Realized communication graph             |
| $L(t)$                | `L[t]`                | Graph Laplacian of $\mathcal{G}(t)$      |
| $\Phi(t)$             | `Phi[t]`              | Windowed Laplacian                       |
| $\lambda_2(\cdot)$    | `lambda2(...)`        | Algebraic connectivity                   |

## Policy

| Paper                  | Code                  | Meaning                                  |
|------------------------|-----------------------|------------------------------------------|
| $V_i(j; t, a_{-i})$    | `value(i, j, t, ...)` | Value of requesting $j$                  |
| $p_{ij}(t)$            | `recip_prob(i, j, t)` | Reciprocation prediction                 |
| $C^\text{switch}_{ij}(t)$ | `switch_cost(i, j, t)` | Switching cost                       |
| $H$                    | `H`                   | Lookahead horizon                        |
| $T$                    | `T`                   | Certificate window length                |
| $T_0$                  | `T_0`                 | Feasibility window length                |
| $c$                    | `c`                   | Switching cost scale                     |

## Analysis

| Paper                   | Code                  | Meaning                                  |
|-------------------------|-----------------------|------------------------------------------|
| $\Gamma_t$              | `gamma_t`             | Per-epoch matching game                  |
| $W_t(a)$                | `potential(t, a)`     | Exact potential function                 |
| $a^\star_t$             | `a_star[t]`           | Equilibrium of $\Gamma_t$                |
| $\alpha_0$              | `alpha_0`             | Persistent feasibility constant          |
| $\rho_\text{match}$     | `rho_match`           | Per-epoch matching efficiency            |
| $\rho_\text{cover}$     | `rho_cover`           | Feasibility coverage efficiency          |
| $\rho$                  | `rho`                 | $\rho_\text{match} \cdot \rho_\text{cover}$ |
