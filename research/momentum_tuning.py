import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import itertools
import pandas as pd

from data.loader import load_market_data, load_price_data, clean_price_data
from data.liquidity import build_liquidity_eligibility_mask
from data.sp500 import get_sp500_tickers
from analytics.returns import compute_daily_returns
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from execution.slippage import apply_execution_slippage

from strategies.momentum import generate_momentum_weights
from engine.backtest import run_backtest

from config.config import (
    TICKERS,
    USE_SP500_UNIVERSE,
    START_DATE,
    END_DATE,
    TRANSACTION_COST,
    MAX_MISSING_DATA_FRACTION,
    MIN_STOCK_PRICE,
    DATA_DOWNLOAD_BATCH_SIZE,
    ENABLE_LIQUIDITY_FILTER,
    MIN_AVERAGE_DOLLAR_VOLUME,
    ADV_LOOKBACK_WINDOW,
    MAX_POSITION_WEIGHT,
    SLIPPAGE_MODEL,
    SLIPPAGE_BPS,
    SLIPPAGE_VOLATILITY_WINDOW,
    SLIPPAGE_VOLATILITY_REFERENCE,
    SLIPPAGE_TURNOVER_MULTIPLIER,
)


# =========================
# SETTINGS
# =========================
LOOKBACKS = [60, 90, 120, 180]
TOP_NS = [5, 10, 15, 20]
REBALANCE_FREQS = [5, 10, 21]
VOL_LOOKBACKS = [10, 20, 40]


# =========================
# Load data once
# =========================
universe_tickers = get_sp500_tickers() if USE_SP500_UNIVERSE else TICKERS
if ENABLE_LIQUIDITY_FILTER:
    prices, volumes = load_market_data(
        universe_tickers,
        START_DATE,
        END_DATE,
        batch_size=DATA_DOWNLOAD_BATCH_SIZE,
    )
else:
    prices = load_price_data(
        universe_tickers,
        START_DATE,
        END_DATE,
        batch_size=DATA_DOWNLOAD_BATCH_SIZE,
    )
    volumes = pd.DataFrame()
prices = clean_price_data(
    prices,
    max_missing_fraction=MAX_MISSING_DATA_FRACTION,
    min_price=MIN_STOCK_PRICE,
)
liquidity_eligibility = (
    build_liquidity_eligibility_mask(
        prices,
        volumes.reindex(index=prices.index, columns=prices.columns),
        min_average_dollar_volume=MIN_AVERAGE_DOLLAR_VOLUME,
        lookback_window=ADV_LOOKBACK_WINDOW,
    )
    if ENABLE_LIQUIDITY_FILTER else None
)

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
            max_position_weight=MAX_POSITION_WEIGHT,
            eligibility_mask=liquidity_eligibility,
        )

        # Backtest (gross)
        gross_returns, turnover = run_backtest(
            weights,
            returns,
            transaction_cost=TRANSACTION_COST,
            return_turnover=True,
        )

        # Apply slippage
        net_returns = apply_execution_slippage(
            gross_returns,
            turnover,
            slippage_bps=SLIPPAGE_BPS,
            model=SLIPPAGE_MODEL,
            volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
            volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
            turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
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
