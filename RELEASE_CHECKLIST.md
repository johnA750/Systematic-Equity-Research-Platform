# Release Checklist

This checklist is the quick public-release guide for reviewers, recruiters, and future maintenance.

## Install

```powershell
cd Trading-Backtester
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project downloads public market data and factor data at runtime, so internet access is required.

## Recommended First Scripts

1. `python run_production_baseline.py`
   Runs the 50-stock production baseline with the default production parameters.

2. `python research\run_sp500_research.py`
   Runs the current S&P 500 research mode with liquidity, execution, factor, and capacity reporting.

3. `python research\run_walk_forward.py`
   Runs selected-parameter versus fixed-production-parameter walk-forward validation.

## Run Modes

```powershell
# Production baseline
python run_production_baseline.py

# Backwards-compatible main runner
python run_backtest.py

# Current S&P 500 research
python research\run_sp500_research.py

# Walk-forward validation
python research\run_walk_forward.py

# Historical point-in-time S&P 500 research
$env:HISTORICAL_CONSTITUENTS_FILE = ".\data\historical_sp500_constituents.csv"
python research\run_historical_sp500.py
```

Historical S&P 500 mode requires a user-supplied point-in-time constituent file. That dataset is intentionally ignored by git because it may be private or licensed.

## Outputs

Reports are written to:

```text
reports/<mode>/<timestamp>/
```

Common outputs include:

- `metrics_summary.csv`
- `factor_summary.csv`
- `rolling_factor_exposures.csv`
- `execution_costs.csv`
- `capacity_summary.csv`
- `comparison_dashboard.png`
- `execution_dashboard.png`
- `capacity_curves.png`
- `impact_comparison.png`

Curated example visuals for GitHub live in `docs/assets/`.

## GitHub Pages Showcase

The public showcase site lives in `docs/` and requires no build step.

Recommended GitHub Pages settings:

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`

Expected project URL:

```text
https://johnA750.github.io/Trading-Backtester/
```

Detailed deployment instructions are in `docs/GITHUB_PAGES.md`.

## Current Limitations

- Yahoo Finance data is suitable for demonstration and research prototyping, not production trading.
- Current S&P 500 mode is not survivorship-bias-free.
- Historical survivorship-aware mode requires external point-in-time membership and delisted-security data.
- Market impact and capacity estimates are simplified institutional-style approximations, not broker-calibrated execution models.
- Results depend on data availability, download quality, and the selected backtest window.

## Future Roadmap

- Add a small CI smoke-test workflow.
- Add an optional offline sample dataset for deterministic demos.
- Add cleaner command-line flags for mode/config overrides.
- Integrate higher-quality institutional data sources for point-in-time membership, delistings, and corporate actions.
- Add packaging metadata if the project becomes a reusable library rather than a research repository.

## Public Release Gate

Before publishing:

- Confirm `git status` contains only intentional source, docs, and metadata changes.
- Confirm `.venv/`, `reports/`, caches, generated CSVs, and private datasets are ignored.
- Run `python -m py_compile` across project Python files.
- Run `python -m pip check`.
- Open `README.md` on GitHub and confirm diagrams and dashboard images render.
