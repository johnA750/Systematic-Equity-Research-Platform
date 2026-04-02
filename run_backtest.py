import pandas as pd

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
from analytics.factor_regression import run_capm_regression, summarize_capm_results
from analytics.regime_analysis import (
    classify_ma_regime,
    classify_vol_regime,
    compute_regime_metrics,
    print_regime_metrics,
)
from strategies.momentum import generate_momentum_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from engine.backtest import run_backtest
from portfolio.regime import compute_regime_multiplier
from portfolio.regime_weighting import compute_regime_weights
from reporting.plots import plot_strategy_vs_benchmark, plot_drawdown


# =========================
# Load and clean price data
# =========================
prices = load_price_data(TICKERS, START_DATE, END_DATE)
prices = clean_price_data(prices)

# Compute asset returns
returns = compute_daily_returns(prices)

# =========================
# Load benchmark
# =========================
benchmark_prices_df = load_price_data(BENCHMARK_TICKER, START_DATE, END_DATE)
benchmark_prices_df = clean_price_data(benchmark_prices_df)
benchmark_price_series = benchmark_prices_df.iloc[:, 0]

benchmark_returns_df = compute_daily_returns(benchmark_prices_df)
benchmark_returns = benchmark_returns_df.iloc[:, 0]

benchmark_cumulative = compute_cumulative_returns(benchmark_returns)
benchmark_metrics = compute_performance_metrics(benchmark_returns)
benchmark_drawdown = compute_drawdown_series(benchmark_cumulative)

# =========================
# Compute regime multiplier
# =========================
regime_multiplier = compute_regime_multiplier(
    benchmark_price_series,
    ma_window=200,
    sensitivity=2.0
)

# =========================
# Momentum Strategy
# =========================
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

# =========================
# Mean Reversion Strategy
# =========================
meanrev_weights = generate_mean_reversion_weights(
    prices,
    lookback=7,
    top_n=TOP_N,
    rebalance_freq=10
)

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

# =========================
# Combined Portfolio - Regime Weighted
# =========================
regime_weights = compute_regime_weights(
    ma_regime,
    bull_momentum_weight=0.7,
    bear_momentum_weight=0.3,
)

regime_weights = regime_weights.reindex(momentum_returns.index).ffill()

combined_returns = (
    regime_weights["momentum"] * momentum_returns +
    regime_weights["mean_reversion"] * meanrev_returns
)
combined_cumulative = compute_cumulative_returns(combined_returns)
combined_metrics = compute_performance_metrics(combined_returns)
combined_drawdown = compute_drawdown_series(combined_cumulative)

combined_turnover = (
    regime_weights["momentum"] * momentum_turnover.reindex(combined_returns.index).fillna(0.0)
    + regime_weights["mean_reversion"] * meanrev_turnover.reindex(combined_returns.index).fillna(0.0)
)
combined_turnover_metrics = compute_turnover_metrics(combined_turnover)

# =========================
# CAPM Regressions
# =========================
momentum_capm = run_capm_regression(momentum_returns, benchmark_returns)
meanrev_capm = run_capm_regression(meanrev_returns, benchmark_returns)
combined_capm = run_capm_regression(combined_returns, benchmark_returns)

momentum_capm_summary = summarize_capm_results(momentum_capm)
meanrev_capm_summary = summarize_capm_results(meanrev_capm)
combined_capm_summary = summarize_capm_results(combined_capm)

# =========================
# Regime Analysis
# =========================
ma_regime = classify_ma_regime(benchmark_price_series, ma_window=200)
vol_regime = classify_vol_regime(benchmark_returns, vol_window=20, threshold_quantile=0.5)

momentum_ma_regime = compute_regime_metrics(
    momentum_returns,
    ma_regime,
    momentum_turnover,
)

meanrev_ma_regime = compute_regime_metrics(
    meanrev_returns,
    ma_regime,
    meanrev_turnover,
)

combined_ma_regime = compute_regime_metrics(
    combined_returns,
    ma_regime,
    combined_turnover,
)

momentum_vol_regime = compute_regime_metrics(
    momentum_returns,
    vol_regime,
    momentum_turnover,
)

meanrev_vol_regime = compute_regime_metrics(
    meanrev_returns,
    vol_regime,
    meanrev_turnover,
)

combined_vol_regime = compute_regime_metrics(
    combined_returns,
    vol_regime,
    combined_turnover,
)

# =========================
# Print metrics
# =========================
print("Momentum Strategy Metrics:")
for key, value in momentum_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMomentum Turnover Metrics:")
for key, value in momentum_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMomentum CAPM Regression:")
for key, value in momentum_capm_summary.items():
    print(f"{key}: {value:.6f}")

print("\nMean Reversion Strategy Metrics:")
for key, value in meanrev_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMean Reversion Turnover Metrics:")
for key, value in meanrev_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMean Reversion CAPM Regression:")
for key, value in meanrev_capm_summary.items():
    print(f"{key}: {value:.6f}")

print("\nLast Regime Weights:")
print(regime_weights.iloc[-1])

print("\nRegime-Weighted Combined Strategy Metrics:")
for key, value in combined_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nRegime-Weighted Combined Turnover Metrics:")
for key, value in combined_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nCombined Strategy CAPM Regression:")
for key, value in combined_capm_summary.items():
    print(f"{key}: {value:.6f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"Mean Reversion: {meanrev_cumulative.iloc[-1]:.4f}")
print(f"Regime-Weighted Combined: {combined_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

print("\nCombined CAPM Full Summary:")
print(combined_capm["model_summary"])

print_regime_metrics("Momentum MA Regime Analysis", momentum_ma_regime)
print_regime_metrics("Mean Reversion MA Regime Analysis", meanrev_ma_regime)
print_regime_metrics("Combined MA Regime Analysis", combined_ma_regime)

print_regime_metrics("Momentum Volatility Regime Analysis", momentum_vol_regime)
print_regime_metrics("Mean Reversion Volatility Regime Analysis", meanrev_vol_regime)
print_regime_metrics("Combined Volatility Regime Analysis", combined_vol_regime)

# =========================
# Plots
# =========================
plot_strategy_vs_benchmark(
    momentum_cumulative,
    benchmark_cumulative.reindex(momentum_cumulative.index),
    strategy_label="Momentum",
    benchmark_label="SPY"
)

plot_drawdown(momentum_drawdown, title="Momentum Drawdown")

plot_strategy_vs_benchmark(
    meanrev_cumulative,
    benchmark_cumulative.reindex(meanrev_cumulative.index),
    strategy_label="Mean Reversion",
    benchmark_label="SPY"
)

plot_drawdown(meanrev_drawdown, title="Mean Reversion Drawdown")

plot_strategy_vs_benchmark(
    combined_cumulative,
    benchmark_cumulative.reindex(combined_cumulative.index),
    strategy_label="Combined Inverse-Vol",
    benchmark_label="SPY"
)

plot_drawdown(combined_drawdown, title="Combined Inverse-Vol Drawdown")