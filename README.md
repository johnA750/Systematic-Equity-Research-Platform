# Trading Strategy Backtesting Engine

A modular Python backtesting engine for evaluating rule-based equity trading strategies on historical U.S. stock data.

## Overview

This project simulates how a portfolio would have performed when following systematic trading rules over time. It is designed to support clean experimentation with simple strategies, transparent assumptions, and reproducible results.

The engine supports a fixed research universe or an optional current S&P 500
universe, momentum and pullback sleeves, realistic execution assumptions, and
robustness analytics. A current-constituent S&P 500 universe is not a
point-in-time historical membership dataset and should not be described as
survivorship-bias-free.

## Current Scope

- Historical daily U.S. equity data
- Momentum and pullback-based mean reversion strategies
- Inverse-vol sleeve combination and volatility targeting
- Transaction costs plus fixed or optional adaptive slippage
- Turnover, gross exposure, drawdown, CAPM, and regime analysis
- Optional current S&P 500 universe, liquidity eligibility, and sector exposure reporting
- Optional research-only SciPy constrained sleeve optimizer

## Configuration Highlights

- Set `USE_SP500_UNIVERSE = True` in `config/config.py` to use the current
  S&P 500 constituent list; leave it `False` for the original 50-stock baseline.
- Set `ENABLE_LIQUIDITY_FILTER = True` to apply lagged rolling average-dollar-
  volume eligibility using `MIN_AVERAGE_DOLLAR_VOLUME` and `ADV_LOOKBACK_WINDOW`.
- Set `MAX_POSITION_WEIGHT` to a decimal cap such as `0.10`; leave it `None`
  to preserve original weighting behavior.
- Set `SLIPPAGE_MODEL` to `"fixed"`, `"volatility_adjusted"`, or
  `"turnover_adjusted"`; `"fixed"` preserves the original return path.
- Set `ENABLE_CONSTRAINED_OPTIMIZER = True` to print optional research-only
  minimum-variance sleeve weights; inverse-vol remains the main construction.

## Experiment Modes

The project exposes three explicit modes. Each run creates a timestamped
artifact directory under `reports/<mode>/` containing result tables, plots,
and `experiment_metadata.json` with the effective universe, configuration,
and execution assumptions.

```powershell
# Production 50-stock fixed-parameter baseline
.\.venv\Scripts\python.exe run_production_baseline.py

# Current S&P 500 scalability and realism research
.\.venv\Scripts\python.exe research\run_sp500_research.py

# Rolling/expanding out-of-sample validation
.\.venv\Scripts\python.exe research\run_walk_forward.py
```

`run_backtest.py` remains backwards-compatible and defaults to the production
baseline profile. Set `VERBOSE_ANALYTICS = True` in `config/config.py` to
restore detailed console sections; summary tables and detailed plots are
saved regardless when `SAVE_REPORTS = True`.

Full-sample modes save one large `comparison_dashboard.png` by default, with
cumulative value, drawdown, gross exposure, and headline metric/CAPM panels
shown together for easier comparison. Set `SAVE_INDIVIDUAL_PLOTS = True` to
also retain the separate legacy charts.

## Walk-Forward Research

Configure rolling or expanding evaluation through the `WALK_FORWARD_*`
settings in `config/config.py`. The runner selects parameter combinations
using training-period Sharpe only, applies train-estimated inverse-vol sleeve
weights to the following test period, and evaluates the fixed production
parameter pair on the identical OOS windows. It reports a selected-versus-
fixed comparison, train/test metrics and CAPM results, calculates parameter
stability, plots rolling OOS Sharpe, volatility, beta, and alpha, and exports
structured CSV tables including `selected_results.csv` and
`fixed_parameter_results.csv`.

## Project Structure

```text
docs/
data/
strategies/
portfolio/
execution/
engine/
analytics/
reporting/
config/

run_backtest.py
