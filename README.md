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
