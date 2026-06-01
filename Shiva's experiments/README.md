# orbitmatch

Predictive matching for inter-satellite link formation in LEO constellations.

Companion code for the ACC submission *Predictive Inter-Satellite Link Formation as a Matching Game*.

## Setup

```bash
# Clone, then from the repo root:
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -e ".[dev]"
```

The `-e` flag installs the package in editable mode — code changes are picked up
without reinstalling. All scripts below should be run from the repo root with
`python -m`.

## Running

Sanity checks (do these first, after any change to the feasibility or graph modules):

```bash
python -m scripts.run_sanity_checks
```

Paper experiments (each produces a file in `results/` and a figure in `figures/paper/`):

```bash
python -m scripts.run_lambda2_traces      # Plot 1: lambda_2 traces vs. baselines
python -m scripts.run_horizon_ablation    # Plot 2: ablation on horizon H
python -m scripts.run_scaling             # Plot 3: scaling with n
python -m scripts.run_robustness          # Plot 4 (optional): satellite-failure recovery
python -m scripts.render_paper_figures    # Re-renders all four plots from saved data
python -m scripts.render_tables           # Builds LaTeX tables in tables/
```

Tests:

```bash
pytest
```

## Project layout

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan, conventions, and rationale.

## Conventions

- All times are integer epoch indices unless suffixed `_s` (seconds) or `_min` (minutes).
- All positions are ECI in km, shape `(n_epochs, n, 3)`.
- All Laplacians are dense `numpy` arrays for `n <= 60`, sparse for larger.
- Results are saved as `.npz` (numpy) or `.parquet` (pandas) — never `.pkl`.
- Figures are saved as PDF for the paper, PNG for diagnostics.
- Color palette is fixed in `orbitmatch/plotting/theme.py` — do not hardcode colors elsewhere.
