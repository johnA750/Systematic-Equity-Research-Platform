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
    OPTIMIZATION_CONFIG,
)

from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from analytics.drawdown import compute_drawdown_series
from analytics.optimization import optimize_strategy_mix
from strategies.momentum import generate_momentum_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from engine.backtest import run_backtest
from portfolio.regime import compute_regime_multiplier
from reporting.plots import plot_strategy_vs_benchmark, plot_drawdown


def split_train_test(df, train_fraction=0.7):
    split_idx = int(len(df) * train_fraction)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


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

# Optional regime scaling for mean reversion if you want to test later:
# regime_multiplier_aligned = regime_multiplier.reindex(meanrev_weights.index).fillna(0.0)
# meanrev_weights = meanrev_weights.mul(regime_multiplier_aligned, axis=0)

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
# Combined Portfolio - Inverse Vol
# =========================
momentum_vol = momentum_returns.std()
meanrev_vol = meanrev_returns.std()

inv_vol_mom = 1 / momentum_vol if momentum_vol != 0 else 0
inv_vol_mr = 1 / meanrev_vol if meanrev_vol != 0 else 0

weight_sum = inv_vol_mom + inv_vol_mr
mom_weight = inv_vol_mom / weight_sum if weight_sum != 0 else 0.5
mr_weight = inv_vol_mr / weight_sum if weight_sum != 0 else 0.5

combined_returns = mom_weight * momentum_returns + mr_weight * meanrev_returns
combined_cumulative = compute_cumulative_returns(combined_returns)
combined_metrics = compute_performance_metrics(combined_returns)
combined_drawdown = compute_drawdown_series(combined_cumulative)

# =========================
# Combined Portfolio - SciPy Optimized
# =========================
strategy_returns = pd.concat(
    [
        momentum_returns.rename("momentum"),
        meanrev_returns.rename("mean_reversion"),
    ],
    axis=1,
).dropna()

strategy_turnovers = pd.concat(
    [
        momentum_turnover.rename("momentum"),
        meanrev_turnover.rename("mean_reversion"),
    ],
    axis=1,
).reindex(strategy_returns.index)

train_returns, test_returns = split_train_test(
    strategy_returns,
    train_fraction=OPTIMIZATION_CONFIG["train_fraction"],
)

train_turnovers, test_turnovers = split_train_test(
    strategy_turnovers,
    train_fraction=OPTIMIZATION_CONFIG["train_fraction"],
)

bounds = [
    OPTIMIZATION_CONFIG["weight_bounds"]["momentum"],
    OPTIMIZATION_CONFIG["weight_bounds"]["mean_reversion"],
]

opt_result = optimize_strategy_mix(
    strategy_returns=train_returns,
    bounds=bounds,
    penalty_strength=OPTIMIZATION_CONFIG["turnover_penalty_strength"],
    strategy_turnovers=train_turnovers,
)

optimized_weights = opt_result["weights"]

optimized_train_returns = (train_returns @ optimized_weights.values).rename("optimized_train")
optimized_test_returns = (test_returns @ optimized_weights.values).rename("optimized_test")
optimized_full_returns = (strategy_returns @ optimized_weights.values).rename("optimized_combined")

optimized_train_cumulative = compute_cumulative_returns(optimized_train_returns)
optimized_test_cumulative = compute_cumulative_returns(optimized_test_returns)
optimized_full_cumulative = compute_cumulative_returns(optimized_full_returns)

optimized_train_metrics = compute_performance_metrics(optimized_train_returns)
optimized_test_metrics = compute_performance_metrics(optimized_test_returns)
optimized_full_metrics = compute_performance_metrics(optimized_full_returns)

optimized_full_drawdown = compute_drawdown_series(optimized_full_cumulative)

# Approx portfolio turnover using weighted sleeve turnover
optimized_portfolio_turnover = (
    strategy_turnovers["momentum"] * abs(optimized_weights["momentum"])
    + strategy_turnovers["mean_reversion"] * abs(optimized_weights["mean_reversion"])
)

optimized_turnover_metrics = compute_turnover_metrics(optimized_portfolio_turnover)

# =========================
# Print metrics
# =========================
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

print("\nInverse-Vol Combined Weights:")
print(f"Momentum Weight: {mom_weight:.4f}")
print(f"Mean Reversion Weight: {mr_weight:.4f}")

print("\nInverse-Vol Combined Strategy Metrics:")
for key, value in combined_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nOptimization Success:")
print(opt_result["success"])
print(opt_result["message"])

print("\nOptimized Combined Weights:")
for key, value in optimized_weights.items():
    print(f"{key}: {value:.4f}")

print("\nOptimized Combined Metrics - Train:")
for key, value in optimized_train_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nOptimized Combined Metrics - Test:")
for key, value in optimized_test_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nOptimized Combined Metrics - Full Sample:")
for key, value in optimized_full_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nOptimized Combined Turnover Metrics:")
for key, value in optimized_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"Mean Reversion: {meanrev_cumulative.iloc[-1]:.4f}")
print(f"Inverse-Vol Combined: {combined_cumulative.iloc[-1]:.4f}")
print(f"Optimized Combined: {optimized_full_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

# =========================
# Plots
# =========================
plot_strategy_vs_benchmark(
    momentum_cumulative,
    benchmark_cumulative,
    strategy_label="Momentum",
    benchmark_label="SPY"
)

plot_drawdown(momentum_drawdown, title="Momentum Drawdown")

plot_strategy_vs_benchmark(
    meanrev_cumulative,
    benchmark_cumulative,
    strategy_label="Mean Reversion",
    benchmark_label="SPY"
)

plot_drawdown(meanrev_drawdown, title="Mean Reversion Drawdown")

plot_strategy_vs_benchmark(
    combined_cumulative,
    benchmark_cumulative,
    strategy_label="Combined Inverse-Vol",
    benchmark_label="SPY"
)

plot_drawdown(combined_drawdown, title="Combined Inverse-Vol Drawdown")

plot_strategy_vs_benchmark(
    optimized_full_cumulative,
    benchmark_cumulative,
    strategy_label="Combined Optimized",
    benchmark_label="SPY"
)

plot_drawdown(optimized_full_drawdown, title="Combined Optimized Drawdown")