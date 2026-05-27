import os
import sys
import random
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_market_data, load_price_data, clean_price_data
from data.liquidity import build_liquidity_eligibility_mask
from data.sp500 import get_sp500_tickers
from analytics.returns import compute_daily_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from execution.slippage import apply_execution_slippage
from strategies.mean_reversion import generate_mean_reversion_weights
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

N_RUNS = 2000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

LOOKBACKS = [1, 2, 3, 4, 5, 7, 10]
TOP_NS = [5, 8, 10, 12, 15, 20, 25, 30]
REBALANCE_FREQS = [1, 2, 3, 5, 7, 10, 15]
TREND_WINDOWS = [20, 30, 50, 75, 100, 150, 200]
MIN_PULLBACKS = [-0.20, -0.15, -0.12, -0.10, -0.08, -0.06, -0.05, -0.04, -0.03]
MAX_PULLBACKS = [-0.06, -0.05, -0.04, -0.03, -0.02, -0.015, -0.01]
WEIGHTING_SCHEMES = ["equal", "shock"]


def sample_params():
    while True:
        lookback = random.choice(LOOKBACKS)
        top_n = random.choice(TOP_NS)
        rebalance_freq = random.choice(REBALANCE_FREQS)
        trend_window = random.choice(TREND_WINDOWS)
        min_pullback = random.choice(MIN_PULLBACKS)
        max_pullback = random.choice(MAX_PULLBACKS)
        weighting_scheme = random.choice(WEIGHTING_SCHEMES)

        if min_pullback < max_pullback and trend_window > lookback * 3:
            return {
                "lookback": lookback,
                "top_n": top_n,
                "rebalance_freq": rebalance_freq,
                "trend_window": trend_window,
                "min_pullback": min_pullback,
                "max_pullback": max_pullback,
                "weighting_scheme": weighting_scheme,
            }


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

results = []
seen = set()

print(f"Running {N_RUNS} mean reversion configurations...\n")

while len(results) < N_RUNS:
    params = sample_params()
    key = tuple(params.items())

    if key in seen:
        continue
    seen.add(key)

    try:
        weights = generate_mean_reversion_weights(
            prices,
            lookback=params["lookback"],
            top_n=params["top_n"],
            rebalance_freq=params["rebalance_freq"],
            trend_window=params["trend_window"],
            min_pullback=params["min_pullback"],
            max_pullback=params["max_pullback"],
            weighting_scheme=params["weighting_scheme"],
            max_position_weight=MAX_POSITION_WEIGHT,
            eligibility_mask=liquidity_eligibility,
        )

        gross_returns, turnover = run_backtest(
            weights,
            returns,
            transaction_cost=TRANSACTION_COST,
            return_turnover=True,
        )

        net_returns = apply_execution_slippage(
            gross_returns,
            turnover,
            slippage_bps=SLIPPAGE_BPS,
            model=SLIPPAGE_MODEL,
            volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
            volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
            turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
        )

        metrics = compute_performance_metrics(net_returns)
        turnover_metrics = compute_turnover_metrics(turnover)

        results.append({
            "lookback": params["lookback"],
            "top_n": params["top_n"],
            "rebalance_freq": params["rebalance_freq"],
            "trend_window": params["trend_window"],
            "min_pullback": params["min_pullback"],
            "max_pullback": params["max_pullback"],
            "weighting_scheme": params["weighting_scheme"],
            "return": metrics["Annualized Return"],
            "sharpe": metrics["Sharpe Ratio"],
            "vol": metrics["Volatility"],
            "max_dd": metrics["Max Drawdown"],
            "turnover": turnover_metrics["Annualized Turnover"],
        })

        if len(results) % 50 == 0 or len(results) == 1:
            print(
                f"[{len(results)}/{N_RUNS}] "
                f"L={params['lookback']}, "
                f"N={params['top_n']}, "
                f"R={params['rebalance_freq']}, "
                f"TW={params['trend_window']}, "
                f"PB=({params['min_pullback']},{params['max_pullback']}), "
                f"W={params['weighting_scheme']}"
            )

    except Exception as e:
        print(f"Failed on params: {params}")
        print(e)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="sharpe", ascending=False)

print("\n================ BEST CONFIGS ==================\n")
print(results_df.head(20).to_string(index=False))

results_df.to_csv("mean_reversion_tuning_results.csv", index=False)
print("\nSaved results to mean_reversion_tuning_results.csv")
