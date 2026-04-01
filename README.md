# Trading Strategy Backtesting Engine

A modular Python backtesting engine for evaluating rule-based equity trading strategies on historical U.S. stock data.

## Overview

This project simulates how a portfolio would have performed when following systematic trading rules over time. It is designed to support clean experimentation with simple strategies, transparent assumptions, and reproducible results.

The initial version focuses on a momentum-based equity strategy using daily data, with planned extensions for additional strategies, transaction cost modeling, and robustness analysis.

## Current Scope

- Historical daily U.S. equity data
- Rule-based momentum strategy
- Long-only portfolio construction
- Equal-weight allocation
- Core performance metrics
- Equity curve reporting

## Planned Extensions

- Mean reversion strategy
- Benchmark comparison
- Transaction cost modeling
- Drawdown and rolling performance analysis
- Parameter sensitivity testing
- Expanded stock universe

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