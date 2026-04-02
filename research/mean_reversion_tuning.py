import itertools
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from analytics.slippage import apply_slippage_to_returns
from strategies.mean_reversion import generate_mean_reversion_weights
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

LOOKBACKS = [3, 5, 7, 10, 15]
TOP_NS = [5, 10, 15, 20]
REBALANCE_FREQS = [2, 5, 10, 21]
TREND_WINDOWS = [20, 50, 100]

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
    TREND_WINDOWS,
))

print(f"Running {len(param_grid)} mean reversion configurations...\n")

for idx, (lookback, top_n, rebalance_freq, trend_window) in enumerate(param_grid):
    try:
        weights = generate_mean_reversion_weights(
            prices,
            lookback=lookback,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
            trend_window=trend_window,
        )

        gross_returns, turnover = run_backtest(
            weights,
            returns,
            transaction_cost=TRANSACTION_COST,
            return_turnover=True,
        )

        net_returns = apply_slippage_to_returns(
            gross_returns,
            turnover,
            slippage_bps=SLIPPAGE_BPS,
        )

        metrics = compute_performance_metrics(net_returns)
        turnover_metrics = compute_turnover_metrics(turnover)

        results.append({
            "lookback": lookback,
            "top_n": top_n,
            "rebalance_freq": rebalance_freq,
            "trend_window": trend_window,
            "return": metrics["Annualized Return"],
            "sharpe": metrics["Sharpe Ratio"],
            "vol": metrics["Volatility"],
            "max_dd": metrics["Max Drawdown"],
            "turnover": turnover_metrics["Annualized Turnover"],
        })

        print(
            f"[{idx+1}/{len(param_grid)}] Done: "
            f"L={lookback}, N={top_n}, R={rebalance_freq}, T={trend_window}"
        )

    except Exception as e:
        print(
            f"Failed: L={lookback}, N={top_n}, "
            f"R={rebalance_freq}, T={trend_window}"
        )
        print(e)

# =========================
# Results dataframe
# =========================
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="sharpe", ascending=False)

print("\n================ BEST CONFIGS ==================\n")
print(results_df.head(15))

# =========================
# Save results
# =========================
results_df.to_csv("mean_reversion_tuning_results.csv", index=False)

print("\nSaved results to mean_reversion_tuning_results.csv")