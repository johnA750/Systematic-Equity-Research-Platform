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
from analytics.slippage import apply_slippage_to_returns, compute_slippage_cost_series
from strategies.momentum import generate_momentum_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from engine.backtest import run_backtest
from portfolio.regime import compute_regime_multiplier
from reporting.plots import plot_strategy_vs_benchmark, plot_drawdown


# =========================
# User-set research assumptions
# =========================
SLIPPAGE_BPS = 5.0
CAPM_USE_HAC = True
CAPM_HAC_LAGS = 5

# =========================
# Load and clean price data
# =========================
prices = load_price_data(TICKERS, START_DATE, END_DATE)
prices = clean_price_data(prices)

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
# Regime classification
# =========================
ma_regime = classify_ma_regime(benchmark_price_series, ma_window=200)
vol_regime = classify_vol_regime(benchmark_returns, vol_window=20, threshold_quantile=0.5)

# =========================
# Regime multiplier for momentum sleeve
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

momentum_returns_gross, momentum_turnover = run_backtest(
    momentum_weights,
    returns,
    transaction_cost=TRANSACTION_COST,
    return_turnover=True
)

momentum_returns = apply_slippage_to_returns(
    momentum_returns_gross,
    momentum_turnover,
    slippage_bps=SLIPPAGE_BPS,
)

momentum_slippage_cost = compute_slippage_cost_series(
    momentum_turnover,
    slippage_bps=SLIPPAGE_BPS,
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
    lookback=5,
    top_n=15,
    rebalance_freq=10,
    trend_window=40,
    min_pullback=-0.10,
    max_pullback=-0.03,
    weighting_scheme="equal",
)

meanrev_returns_gross, meanrev_turnover = run_backtest(
    meanrev_weights,
    returns,
    transaction_cost=TRANSACTION_COST,
    return_turnover=True
)

meanrev_returns = apply_slippage_to_returns(
    meanrev_returns_gross,
    meanrev_turnover,
    slippage_bps=SLIPPAGE_BPS,
)

meanrev_slippage_cost = compute_slippage_cost_series(
    meanrev_turnover,
    slippage_bps=SLIPPAGE_BPS,
)

meanrev_cumulative = compute_cumulative_returns(meanrev_returns)
meanrev_metrics = compute_performance_metrics(meanrev_returns)
meanrev_turnover_metrics = compute_turnover_metrics(meanrev_turnover)
meanrev_drawdown = compute_drawdown_series(meanrev_cumulative)

# =========================
# Combined Portfolio - Inverse Vol Baseline
# =========================
momentum_vol = momentum_returns.std()
meanrev_vol = meanrev_returns.std()

inv_vol_mom = 1 / momentum_vol if momentum_vol != 0 else 0
inv_vol_mr = 1 / meanrev_vol if meanrev_vol != 0 else 0

weight_sum = inv_vol_mom + inv_vol_mr
base_mom_weight = inv_vol_mom / weight_sum if weight_sum != 0 else 0.5
base_mr_weight = inv_vol_mr / weight_sum if weight_sum != 0 else 0.5

combined_returns = base_mom_weight * momentum_returns + base_mr_weight * meanrev_returns
combined_cumulative = compute_cumulative_returns(combined_returns)
combined_metrics = compute_performance_metrics(combined_returns)
combined_drawdown = compute_drawdown_series(combined_cumulative)

combined_turnover = (
    base_mom_weight * momentum_turnover.reindex(combined_returns.index).fillna(0.0)
    + base_mr_weight * meanrev_turnover.reindex(combined_returns.index).fillna(0.0)
)
combined_turnover_metrics = compute_turnover_metrics(combined_turnover)

combined_slippage_cost = compute_slippage_cost_series(
    combined_turnover,
    slippage_bps=SLIPPAGE_BPS,
)

# =========================
# Combined Portfolio - Smooth Volatility Weighting (Experimental)
# =========================
short_vol = benchmark_returns.rolling(20).std()
long_vol = benchmark_returns.rolling(100).std()

vol_ratio = short_vol / long_vol
vol_ratio = vol_ratio.replace([float("inf"), -float("inf")], pd.NA)
vol_ratio = vol_ratio.bfill().clip(lower=0.5, upper=1.5)

momentum_weight_series = 0.60 - 0.15 * (vol_ratio - 1.0)
momentum_weight_series = momentum_weight_series.clip(lower=0.45, upper=0.65)
meanrev_weight_series = 1.0 - momentum_weight_series

momentum_weight_series = momentum_weight_series.reindex(momentum_returns.index).ffill().fillna(0.60)
meanrev_weight_series = meanrev_weight_series.reindex(meanrev_returns.index).ffill().fillna(0.40)

tilt_strength = 0.25

tilted_mom_weight = (1 - tilt_strength) * base_mom_weight + tilt_strength * momentum_weight_series
tilted_mr_weight = (1 - tilt_strength) * base_mr_weight + tilt_strength * meanrev_weight_series

experimental_combined_returns = (
    tilted_mom_weight * momentum_returns +
    tilted_mr_weight * meanrev_returns
)

experimental_combined_cumulative = compute_cumulative_returns(experimental_combined_returns)
experimental_combined_metrics = compute_performance_metrics(experimental_combined_returns)
experimental_combined_drawdown = compute_drawdown_series(experimental_combined_cumulative)

experimental_combined_turnover = (
    tilted_mom_weight * momentum_turnover.reindex(experimental_combined_returns.index).fillna(0.0)
    + tilted_mr_weight * meanrev_turnover.reindex(experimental_combined_returns.index).fillna(0.0)
)
experimental_combined_turnover_metrics = compute_turnover_metrics(experimental_combined_turnover)

experimental_combined_slippage_cost = compute_slippage_cost_series(
    experimental_combined_turnover,
    slippage_bps=SLIPPAGE_BPS,
)

# =========================
# CAPM regression
# =========================
momentum_capm = run_capm_regression(
    momentum_returns,
    benchmark_returns,
    use_hac=CAPM_USE_HAC,
    hac_lags=CAPM_HAC_LAGS,
)

meanrev_capm = run_capm_regression(
    meanrev_returns,
    benchmark_returns,
    use_hac=CAPM_USE_HAC,
    hac_lags=CAPM_HAC_LAGS,
)

combined_capm = run_capm_regression(
    combined_returns,
    benchmark_returns,
    use_hac=CAPM_USE_HAC,
    hac_lags=CAPM_HAC_LAGS,
)

experimental_combined_capm = run_capm_regression(
    experimental_combined_returns,
    benchmark_returns,
    use_hac=CAPM_USE_HAC,
    hac_lags=CAPM_HAC_LAGS,
)

momentum_capm_summary = summarize_capm_results(momentum_capm)
meanrev_capm_summary = summarize_capm_results(meanrev_capm)
combined_capm_summary = summarize_capm_results(combined_capm)
experimental_combined_capm_summary = summarize_capm_results(experimental_combined_capm)

# =========================
# Regime analysis
# =========================
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

experimental_combined_ma_regime = compute_regime_metrics(
    experimental_combined_returns,
    ma_regime,
    experimental_combined_turnover,
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

experimental_combined_vol_regime = compute_regime_metrics(
    experimental_combined_returns,
    vol_regime,
    experimental_combined_turnover,
)

# =========================
# Print metrics
# =========================
print(f"Assumed Slippage (bps): {SLIPPAGE_BPS:.2f}")
print(f"CAPM Covariance Type: {'HAC/Newey-West' if CAPM_USE_HAC else 'Nonrobust'}")

print("\nMomentum Strategy Metrics:")
for key, value in momentum_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMomentum Turnover Metrics:")
for key, value in momentum_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMomentum Average Daily Slippage Cost:")
print(f"{momentum_slippage_cost.mean():.6f}")

print("\nMomentum CAPM Regression:")
for key, value in momentum_capm_summary.items():
    if isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.6f}")

print("\nMean Reversion Strategy Metrics:")
for key, value in meanrev_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMean Reversion Turnover Metrics:")
for key, value in meanrev_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMean Reversion Average Daily Slippage Cost:")
print(f"{meanrev_slippage_cost.mean():.6f}")

print("\nMean Reversion CAPM Regression:")
for key, value in meanrev_capm_summary.items():
    if isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.6f}")

print("\nInverse-Vol Combined Weights:")
print(f"Momentum Weight: {base_mom_weight:.4f}")
print(f"Mean Reversion Weight: {base_mr_weight:.4f}")

print("\nInverse-Vol Combined Strategy Metrics:")
for key, value in combined_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nInverse-Vol Combined Turnover Metrics:")
for key, value in combined_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nInverse-Vol Combined Average Daily Slippage Cost:")
print(f"{combined_slippage_cost.mean():.6f}")

print("\nCombined Strategy CAPM Regression:")
for key, value in combined_capm_summary.items():
    if isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.6f}")

print("\nExperimental Smooth Vol Weights (Last):")
print(f"Momentum Weight: {tilted_mom_weight.iloc[-1]:.4f}")
print(f"Mean Reversion Weight: {tilted_mr_weight.iloc[-1]:.4f}")

print("\nExperimental Smooth Combined Strategy Metrics:")
for key, value in experimental_combined_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nExperimental Smooth Combined Turnover Metrics:")
for key, value in experimental_combined_turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nExperimental Smooth Combined Average Daily Slippage Cost:")
print(f"{experimental_combined_slippage_cost.mean():.6f}")

print("\nExperimental Smooth CAPM Regression:")
for key, value in experimental_combined_capm_summary.items():
    if isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.6f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"Mean Reversion: {meanrev_cumulative.iloc[-1]:.4f}")
print(f"Inverse-Vol Combined: {combined_cumulative.iloc[-1]:.4f}")
print(f"Experimental Smooth Combined: {experimental_combined_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

print("\nCombined CAPM Full Summary:")
print(combined_capm["model_summary"])

print_regime_metrics("Momentum MA Regime Analysis", momentum_ma_regime)
print_regime_metrics("Mean Reversion MA Regime Analysis", meanrev_ma_regime)
print_regime_metrics("Combined MA Regime Analysis", combined_ma_regime)
print_regime_metrics("Experimental Combined MA Regime Analysis", experimental_combined_ma_regime)

print_regime_metrics("Momentum Volatility Regime Analysis", momentum_vol_regime)
print_regime_metrics("Mean Reversion Volatility Regime Analysis", meanrev_vol_regime)
print_regime_metrics("Combined Volatility Regime Analysis", combined_vol_regime)
print_regime_metrics("Experimental Combined Volatility Regime Analysis", experimental_combined_vol_regime)

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

plot_strategy_vs_benchmark(
    experimental_combined_cumulative,
    benchmark_cumulative.reindex(experimental_combined_cumulative.index),
    strategy_label="Experimental Smooth Vol",
    benchmark_label="SPY"
)

plot_drawdown(experimental_combined_drawdown, title="Experimental Smooth Vol Drawdown")