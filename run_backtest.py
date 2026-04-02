import pandas as pd
import matplotlib.pyplot as plt

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
from analytics.slippage import apply_slippage_to_returns
from strategies.momentum import generate_momentum_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from engine.backtest import run_backtest
from portfolio.regime import compute_regime_multiplier
from portfolio.vol_targeting import apply_vol_targeting
from reporting.plots import plot_strategy_vs_benchmark, plot_drawdown


# =========================
# User-set research assumptions
# =========================
SLIPPAGE_BPS = 5.0
CAPM_USE_HAC = True
CAPM_HAC_LAGS = 5

VOL_TARGET_ENABLED = True
TARGET_ANNUAL_VOL = 0.12
VOL_TARGET_WINDOW = 20
VOL_TARGET_MAX_LEVERAGE = 1.5

VERBOSE = False

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

momentum_cumulative = compute_cumulative_returns(momentum_returns)
momentum_metrics = compute_performance_metrics(momentum_returns)
momentum_turnover_metrics = compute_turnover_metrics(momentum_turnover)
momentum_drawdown = compute_drawdown_series(momentum_cumulative)

# =========================
# Mean Reversion / Pullback Strategy
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
combined_returns.name = "combined_inverse_vol"

combined_cumulative = compute_cumulative_returns(combined_returns)
combined_metrics = compute_performance_metrics(combined_returns)
combined_drawdown = compute_drawdown_series(combined_cumulative)

combined_turnover = (
    base_mom_weight * momentum_turnover.reindex(combined_returns.index).fillna(0.0)
    + base_mr_weight * meanrev_turnover.reindex(combined_returns.index).fillna(0.0)
)
combined_turnover_metrics = compute_turnover_metrics(combined_turnover)

# =========================
# Combined Portfolio - Vol Targeted
# =========================
if VOL_TARGET_ENABLED:
    vol_targeted_returns, leverage_series = apply_vol_targeting(
        combined_returns,
        target_annual_vol=TARGET_ANNUAL_VOL,
        rolling_window=VOL_TARGET_WINDOW,
        max_leverage=VOL_TARGET_MAX_LEVERAGE,
    )
else:
    vol_targeted_returns = combined_returns.copy()
    leverage_series = pd.Series(1.0, index=combined_returns.index, name="leverage")

vol_targeted_returns.name = "combined_vol_targeted"
vol_targeted_cumulative = compute_cumulative_returns(vol_targeted_returns)
vol_targeted_metrics = compute_performance_metrics(vol_targeted_returns)
vol_targeted_drawdown = compute_drawdown_series(vol_targeted_cumulative)

# =========================
# CAPM regression
# =========================
combined_capm = run_capm_regression(
    combined_returns,
    benchmark_returns,
    use_hac=CAPM_USE_HAC,
    hac_lags=CAPM_HAC_LAGS,
)

vol_targeted_capm = run_capm_regression(
    vol_targeted_returns,
    benchmark_returns,
    use_hac=CAPM_USE_HAC,
    hac_lags=CAPM_HAC_LAGS,
)

combined_capm_summary = summarize_capm_results(combined_capm)
vol_targeted_capm_summary = summarize_capm_results(vol_targeted_capm)

# =========================
# Regime analysis
# =========================
combined_ma_regime = compute_regime_metrics(
    combined_returns,
    ma_regime,
    combined_turnover,
)

combined_vol_regime = compute_regime_metrics(
    combined_returns,
    vol_regime,
    combined_turnover,
)

# =========================
# Print metrics
# =========================
print(f"Assumed Slippage (bps): {SLIPPAGE_BPS:.2f}")

print("\nMomentum Strategy Metrics:")
for key, value in momentum_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMean Reversion Strategy Metrics:")
for key, value in meanrev_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nInverse-Vol Combined Weights:")
print(f"Momentum Weight: {base_mom_weight:.4f}")
print(f"Mean Reversion Weight: {base_mr_weight:.4f}")

print("\nInverse-Vol Combined Strategy Metrics:")
for key, value in combined_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nVol-Targeted Combined Strategy Metrics:")
for key, value in vol_targeted_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"Mean Reversion: {meanrev_cumulative.iloc[-1]:.4f}")
print(f"Inverse-Vol Combined: {combined_cumulative.iloc[-1]:.4f}")
print(f"Vol-Targeted Combined: {vol_targeted_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

print("\nCombined Strategy CAPM Regression:")
for key, value in combined_capm_summary.items():
    if isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.6f}")

print("\nVol-Targeted Combined CAPM Regression:")
for key, value in vol_targeted_capm_summary.items():
    if isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.6f}")

if VERBOSE:
    print("\nMomentum Turnover Metrics:")
    for key, value in momentum_turnover_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nMean Reversion Turnover Metrics:")
    for key, value in meanrev_turnover_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nInverse-Vol Combined Turnover Metrics:")
    for key, value in combined_turnover_metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\nAverage Vol-Target Leverage: {leverage_series.mean():.4f}")
    print(f"Max Vol-Target Leverage: {leverage_series.max():.4f}")

    print_regime_metrics("Combined MA Regime Analysis", combined_ma_regime)
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

plot_strategy_vs_benchmark(
    vol_targeted_cumulative,
    benchmark_cumulative.reindex(vol_targeted_cumulative.index),
    strategy_label="Vol-Targeted Combined",
    benchmark_label="SPY"
)

plot_drawdown(vol_targeted_drawdown, title="Vol-Targeted Combined Drawdown")

# Show all plots at once
plt.show()