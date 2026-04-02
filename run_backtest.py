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
from analytics.drawdown import compute_drawdown_series
from strategies.momentum import generate_momentum_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from engine.backtest import run_backtest
from portfolio.regime import compute_regime_multiplier
from reporting.plots import plot_strategy_vs_benchmark, plot_drawdown


# Load and clean price data
prices = load_price_data(TICKERS, START_DATE, END_DATE)
prices = clean_price_data(prices)

# Compute asset returns
returns = compute_daily_returns(prices)

# Load benchmark
benchmark_prices_df = load_price_data(BENCHMARK_TICKER, START_DATE, END_DATE)
benchmark_prices_df = clean_price_data(benchmark_prices_df)
benchmark_price_series = benchmark_prices_df.iloc[:, 0]

benchmark_returns_df = compute_daily_returns(benchmark_prices_df)
benchmark_returns = benchmark_returns_df.iloc[:, 0]

benchmark_cumulative = compute_cumulative_returns(benchmark_returns)
benchmark_metrics = compute_performance_metrics(benchmark_returns)
benchmark_drawdown = compute_drawdown_series(benchmark_cumulative)

# Compute regime multiplier from benchmark
regime_multiplier = compute_regime_multiplier(
    benchmark_price_series,
    ma_window=200,
    sensitivity=2.0
)

# -------- Momentum Strategy --------
momentum_weights = generate_momentum_weights(
    prices,
    lookback=MOMENTUM_LOOKBACK,
    top_n=TOP_N,
    rebalance_freq=REBALANCE_FREQ,
    vol_lookback=20
)

regime_multiplier_aligned = regime_multiplier.reindex(momentum_weights.index).fillna(0.0)
momentum_weights = momentum_weights.mul(regime_multiplier_aligned, axis=0)

momentum_returns, momentum_turnover = run_backtest(
    momentum_weights,
    returns,
    transaction_cost=TRANSACTION_COST,
    return_turnover=True
)

momentum_cumulative = compute_cumulative_returns(momentum_returns)
momentum_metrics = compute_performance_metrics(momentum_returns)
momentum_turnover_metrics = compute_turnover_metrics(momentum_turnover)
momentum_drawdown = compute_drawdown_series(momentum_cumulative)

# -------- Mean Reversion Strategy --------
meanrev_weights = generate_mean_reversion_weights(
    prices,
    lookback=20,
    top_n=TOP_N,
    rebalance_freq=REBALANCE_FREQ
)

regime_multiplier_aligned = regime_multiplier.reindex(meanrev_weights.index).fillna(0.0)
meanrev_weights = meanrev_weights.mul(regime_multiplier_aligned, axis=0)

meanrev_returns, meanrev_turnover = run_backtest(
    meanrev_weights,
    returns,
    transaction_cost=TRANSACTION_COST,
    return_turnover=True
)

meanrev_cumulative = compute_cumulative_returns(meanrev_returns)
meanrev_metrics = compute_performance_metrics(meanrev_returns)
meanrev_turnover_metrics = compute_turnover_metrics(meanrev_turnover)
meanrev_drawdown = compute_drawdown_series(meanrev_cumulative)

# Print metrics
print("Momentum Strategy Metrics:")
for key, value in momentum_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMomentum Turnover Metrics:")
for key, value in momentum_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMean Reversion Strategy Metrics:")
for key, value in meanrev_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMean Reversion Turnover Metrics:")
for key, value in meanrev_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"Mean Reversion: {meanrev_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

# Plots
plot_strategy_vs_benchmark(
    momentum_cumulative,
    benchmark_cumulative,
    strategy_label="Momentum",
    benchmark_label="SPY"
)

plot_drawdown(momentum_drawdown, title="Momentum Drawdown")
plot_drawdown(benchmark_drawdown, title="SPY Drawdown")

plot_strategy_vs_benchmark(
    meanrev_cumulative,
    benchmark_cumulative,
    strategy_label="Mean Reversion",
    benchmark_label="SPY"
)

plot_drawdown(meanrev_drawdown, title="Mean Reversion Drawdown")