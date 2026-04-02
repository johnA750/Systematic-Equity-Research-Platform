from config.config import (
    TICKERS,
    BENCHMARK_TICKER,
    START_DATE,
    END_DATE,
    MOMENTUM_LOOKBACK,
    TOP_N,
    REBALANCE_FREQ,
    TRANSACTION_COST,
)

from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from strategies.momentum import generate_momentum_weights
from engine.backtest import run_backtest
from reporting.plots import plot_strategy_vs_benchmark


# Load and clean price data
prices = load_price_data(TICKERS, START_DATE, END_DATE)
prices = clean_price_data(prices)

# Compute asset returns
returns = compute_daily_returns(prices)

# Generate momentum weights
momentum_weights = generate_momentum_weights(
    prices,
    lookback=MOMENTUM_LOOKBACK,
    top_n=TOP_N,
    rebalance_freq=REBALANCE_FREQ
)

# Run momentum backtest
momentum_returns, momentum_turnover = run_backtest(
    momentum_weights,
    returns,
    transaction_cost=TRANSACTION_COST,
    return_turnover=True
)

momentum_cumulative = compute_cumulative_returns(momentum_returns)
momentum_metrics = compute_performance_metrics(momentum_returns)
turnover_metrics = compute_turnover_metrics(momentum_turnover)

# Load benchmark
benchmark_prices = load_price_data(BENCHMARK_TICKER, START_DATE, END_DATE)
benchmark_prices = clean_price_data(benchmark_prices)

benchmark_returns_df = compute_daily_returns(benchmark_prices)
benchmark_returns = benchmark_returns_df.iloc[:, 0]

benchmark_cumulative = compute_cumulative_returns(benchmark_returns)
benchmark_metrics = compute_performance_metrics(benchmark_returns)

# Print metrics
print("Momentum Strategy Metrics:")
for key, value in momentum_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMomentum Turnover Metrics:")
for key, value in turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

# Plot performance
plot_strategy_vs_benchmark(
    momentum_cumulative,
    benchmark_cumulative,
    strategy_label="Momentum",
    benchmark_label="SPY"
)