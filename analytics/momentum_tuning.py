import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import itertools
import pandas as pd

from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from analytics.slippage import apply_slippage_to_returns

from strategies.momentum import generate_momentum_weights
from engine.backtest import run_backtest

from config.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    TRANSACTION_COST,
)


# =========================
# SETTINGS
# =========================
SLIPPAGE_BPS = 5.0

LOOKBACKS = [60, 90, 120, 180]
TOP_NS = [5, 10, 15, 20]
REBALANCE_FREQS = [5, 10, 21]
VOL_LOOKBACKS = [10, 20, 40]


# =========================
# Load data once
# =========================
prices = load_price_data(TICKERS, START_DATE, END_DATE)
prices = clean_price_data(prices)

returns = compute_daily_returns(prices)


# =========================
# Run grid search
# =========================
results = []

param_grid = list(itertools.product(
    LOOKBACKS,
    TOP_NS,
    REBALANCE_FREQS,
    VOL_LOOKBACKS
))

print(f"Running {len(param_grid)} momentum configurations...\n")

for idx, (lookback, top_n, rebalance_freq, vol_lookback) in enumerate(param_grid):

    try:
        # Generate weights
        weights = generate_momentum_weights(
            prices,
            lookback=lookback,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
            vol_lookback=vol_lookback,
        )

        # Backtest (gross)
        gross_returns, turnover = run_backtest(
            weights,
            returns,
            transaction_cost=TRANSACTION_COST,
            return_turnover=True,
        )

        # Apply slippage
        net_returns = apply_slippage_to_returns(
            gross_returns,
            turnover,
            slippage_bps=SLIPPAGE_BPS,
        )

        # Metrics
        metrics = compute_performance_metrics(net_returns)
        turnover_metrics = compute_turnover_metrics(turnover)

        results.append({
            "lookback": lookback,
            "top_n": top_n,
            "rebalance_freq": rebalance_freq,
            "vol_lookback": vol_lookback,

            "return": metrics["Annualized Return"],
            "sharpe": metrics["Sharpe Ratio"],
            "vol": metrics["Volatility"],
            "max_dd": metrics["Max Drawdown"],

            "turnover": turnover_metrics["Annualized Turnover"],
        })

        print(f"[{idx+1}/{len(param_grid)}] Done: L={lookback}, N={top_n}, R={rebalance_freq}, V={vol_lookback}")

    except Exception as e:
        print(f"Failed: L={lookback}, N={top_n}, R={rebalance_freq}, V={vol_lookback}")
        print(e)


# =========================
# Results dataframe
# =========================
results_df = pd.DataFrame(results)

# Rank by Sharpe
results_df = results_df.sort_values(by="sharpe", ascending=False)

print("\n================ BEST CONFIGS ==================\n")
print(results_df.head(10))


# =========================
# Save results
# =========================
results_df.to_csv("momentum_tuning_results.csv", index=False)

print("\nSaved results to momentum_tuning_results.csv")