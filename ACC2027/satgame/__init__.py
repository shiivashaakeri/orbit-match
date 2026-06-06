"""satgame: decentralized inter-satellite-link topology formation as a game.

Submodules
----------
constellation : orbital model + SGP4 propagation
geometry      : relative bearings and field-of-view feasibility
graph         : empty-network construction + windowed-union / union-series helpers
baselines     : greedy / furthest / MDMD link-selection baselines
game          : the potential-game best-response policy (game_theory_formation)
metrics       : connectivity metrics (algebraic connectivity, effective
                resistance, log tau -- the optimized objective)
simulate      : trajectory precompute + epoch-by-epoch simulation driver
"""

__version__ = "0.1.0"
