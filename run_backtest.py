import os
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config.modes import get_experiment_mode
from config.config import (
    TICKERS,
    BENCHMARK_TICKER,
    UNIVERSE_MODE as CONFIG_UNIVERSE_MODE,
    USE_SP500_UNIVERSE,
    HISTORICAL_CONSTITUENTS_FILE,
    START_DATE,
    END_DATE,
    MOMENTUM_LOOKBACK,
    TOP_N,
    REBALANCE_FREQ,
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
    ENABLE_MARKET_IMPACT,
    MARKET_IMPACT_MODEL,
    MAX_ADV_PARTICIPATION,
    IMPACT_COEFFICIENT,
    CAPACITY_TARGET_DOLLARS,
    ENABLE_CONSTRAINED_OPTIMIZER,
    OPTIMIZER_MAX_WEIGHT,
    REPORT_ROOT,
    SAVE_REPORTS,
    SAVE_PLOTS,
    SAVE_INDIVIDUAL_PLOTS,
    VERBOSE_ANALYTICS,
    ENABLE_FACTOR_ANALYSIS,
    FACTOR_FREQUENCY,
    ENABLE_FF5_ANALYSIS,
    FACTOR_ROLLING_MODEL,
    FACTOR_ROLLING_WINDOW,
    FACTOR_USE_HAC,
    FACTOR_HAC_LAGS,
)

from data.loader import load_market_data, load_price_data, clean_price_data
from data.liquidity import build_liquidity_eligibility_mask
from data.sp500 import get_sp500_constituents
from data.constituents import (
    CURRENT_SP500,
    HISTORICAL_SP500,
    build_membership_mask,
    combine_eligibility_masks,
    compute_annual_membership_changes,
    compute_universe_turnover,
    load_constituent_history,
    report_missing_historical_prices,
)
from analytics.returns import compute_daily_returns
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from analytics.drawdown import compute_drawdown_series
from analytics.factor_regression import run_capm_regression, summarize_capm_results
from analytics.factors import run_factor_model_suite, compute_rolling_factor_exposures
from analytics.exposure import (
    compute_gross_exposure,
    summarize_gross_exposure,
    summarize_sector_exposures,
)
from analytics.regime_analysis import (
    classify_ma_regime,
    classify_vol_regime,
    compute_regime_metrics,
    print_regime_metrics,
)
from execution.slippage import apply_execution_slippage, compute_slippage_costs
from execution.market_impact import (
    build_execution_cost_report,
    build_impact_diagnostic_tables,
    compute_capacity_sensitivity,
    estimate_market_impact,
)
from strategies.momentum import generate_momentum_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from engine.backtest import run_backtest
from portfolio.regime import compute_regime_multiplier
from portfolio.vol_targeting import apply_vol_targeting
from portfolio.optimization import optimize_long_only_weights
from reporting.plots import (
    plot_backtest_dashboard,
    plot_strategy_vs_benchmark,
    plot_drawdown,
    plot_gross_exposure,
    plot_rolling_factor_exposures,
    plot_execution_dashboard,
    plot_capacity_curves,
    plot_impact_comparison,
    plot_survivorship_bias_analysis,
)
from reporting.experiments import (
    create_experiment_directory,
    save_figures,
    save_metadata,
    save_tables,
)


# =========================
# User-set research assumptions
# =========================
CAPM_USE_HAC = True
CAPM_HAC_LAGS = 5

VOL_TARGET_ENABLED = True
TARGET_ANNUAL_VOL = 0.12
VOL_TARGET_WINDOW = 20
VOL_TARGET_MAX_LEVERAGE = 1.5

configured_default_mode = {
    CURRENT_SP500: "sp500_research",
    HISTORICAL_SP500: "historical_sp500_research",
}.get(CONFIG_UNIVERSE_MODE, "production_baseline")
ACTIVE_MODE = get_experiment_mode(os.environ.get("BACKTEST_MODE", configured_default_mode))
UNIVERSE_MODE = ACTIVE_MODE.universe_mode
HISTORICAL_CONSTITUENTS_FILE = os.environ.get(
    "HISTORICAL_CONSTITUENTS_FILE",
    HISTORICAL_CONSTITUENTS_FILE,
)
USE_SP500_UNIVERSE = ACTIVE_MODE.use_sp500_universe
ENABLE_LIQUIDITY_FILTER = ACTIVE_MODE.enable_liquidity_filter
MAX_POSITION_WEIGHT = ACTIVE_MODE.max_position_weight
SLIPPAGE_MODEL = ACTIVE_MODE.slippage_model
ENABLE_CONSTRAINED_OPTIMIZER = ACTIVE_MODE.enable_constrained_optimizer
VERBOSE = VERBOSE_ANALYTICS

# =========================
# Load and clean price data
# =========================
constituent_history = pd.DataFrame()
membership_mask = None
universe_turnover = pd.DataFrame()
annual_membership_changes = pd.DataFrame()
unavailable_historical_prices = pd.DataFrame()
if UNIVERSE_MODE == CURRENT_SP500:
    constituents = get_sp500_constituents()
    universe_tickers = [record["ticker"] for record in constituents]
    sector_map = {
        record["ticker"]: record["sector"]
        for record in constituents
        if record["sector"]
    }
elif UNIVERSE_MODE == HISTORICAL_SP500:
    constituent_history = load_constituent_history(HISTORICAL_CONSTITUENTS_FILE)
    universe_tickers = sorted(constituent_history["ticker"].unique())
    sector_records = constituent_history[
        constituent_history["sector"].astype(str).str.len() > 0
    ]
    sector_map = (
        sector_records.drop_duplicates("ticker", keep="last")
        .set_index("ticker")["sector"]
        .to_dict()
    )
else:
    universe_tickers = TICKERS
    sector_map = {}

prices, volumes = load_market_data(
    universe_tickers,
    START_DATE,
    END_DATE,
    batch_size=DATA_DOWNLOAD_BATCH_SIZE,
)
if UNIVERSE_MODE == HISTORICAL_SP500:
    unavailable_historical_prices = report_missing_historical_prices(
        constituent_history,
        prices,
    )

prices = clean_price_data(
    prices,
    max_missing_fraction=None if UNIVERSE_MODE == HISTORICAL_SP500 else MAX_MISSING_DATA_FRACTION,
    min_price=MIN_STOCK_PRICE,
)

if prices.empty:
    raise ValueError("No tradable equities remained after universe data filtering.")

if UNIVERSE_MODE == HISTORICAL_SP500:
    membership_mask = build_membership_mask(
        prices.index,
        constituent_history,
        tickers=list(prices.columns),
    )
    universe_turnover = compute_universe_turnover(membership_mask)
    annual_membership_changes = compute_annual_membership_changes(universe_turnover)

liquidity_eligibility = None
if ENABLE_LIQUIDITY_FILTER:
    liquidity_eligibility = build_liquidity_eligibility_mask(
        prices,
        volumes.reindex(index=prices.index, columns=prices.columns),
        min_average_dollar_volume=MIN_AVERAGE_DOLLAR_VOLUME,
        lookback_window=ADV_LOOKBACK_WINDOW,
    )
strategy_eligibility = combine_eligibility_masks(membership_mask, liquidity_eligibility)

print(f"Mode: {ACTIVE_MODE.name}")
universe_description = {
    CURRENT_SP500: "Current S&P 500",
    HISTORICAL_SP500: "Point-in-time historical S&P 500",
}.get(UNIVERSE_MODE, "Configured ticker list")
print(
    f"Universe: {universe_description} "
    f"({len(prices.columns)} retained of {len(universe_tickers)} requested)"
)
if UNIVERSE_MODE == HISTORICAL_SP500:
    print(
        f"Membership Changes: {int(annual_membership_changes['additions'].sum())} additions; "
        f"{int(annual_membership_changes['removals'].sum())} removals; "
        f"{len(unavailable_historical_prices)} unavailable price histories"
    )
if ENABLE_LIQUIDITY_FILTER and liquidity_eligibility is not None:
    latest_eligible = int(liquidity_eligibility.iloc[-1].sum())
    print(
        f"Liquidity Filter: enabled (ADV >= ${MIN_AVERAGE_DOLLAR_VOLUME:,.0f}, "
        f"{ADV_LOOKBACK_WINDOW}-day lagged window; "
        f"{latest_eligible} eligible on final date)"
    )
elif ENABLE_LIQUIDITY_FILTER:
    print("Liquidity Filter: volume unavailable; continuing with price-only behavior")

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
    vol_lookback=20,
    max_position_weight=MAX_POSITION_WEIGHT,
    eligibility_mask=strategy_eligibility,
)

regime_multiplier_aligned = regime_multiplier.reindex(momentum_weights.index).fillna(0.0)
momentum_weights = momentum_weights.mul(regime_multiplier_aligned, axis=0)
if membership_mask is not None:
    momentum_weights = momentum_weights.where(membership_mask, 0.0)

momentum_returns_gross, momentum_turnover = run_backtest(
    momentum_weights,
    returns,
    transaction_cost=TRANSACTION_COST,
    return_turnover=True
)

momentum_returns = apply_execution_slippage(
    momentum_returns_gross,
    momentum_turnover,
    slippage_bps=SLIPPAGE_BPS,
    model=SLIPPAGE_MODEL,
    volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
    turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
)
momentum_slippage_costs = compute_slippage_costs(
    momentum_returns_gross,
    momentum_turnover,
    slippage_bps=SLIPPAGE_BPS,
    model=SLIPPAGE_MODEL,
    volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
    turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
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
    max_position_weight=MAX_POSITION_WEIGHT,
    eligibility_mask=strategy_eligibility,
)

if membership_mask is not None:
    meanrev_weights = meanrev_weights.where(membership_mask, 0.0)

meanrev_returns_gross, meanrev_turnover = run_backtest(
    meanrev_weights,
    returns,
    transaction_cost=TRANSACTION_COST,
    return_turnover=True
)

meanrev_returns = apply_execution_slippage(
    meanrev_returns_gross,
    meanrev_turnover,
    slippage_bps=SLIPPAGE_BPS,
    model=SLIPPAGE_MODEL,
    volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
    turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
)
meanrev_slippage_costs = compute_slippage_costs(
    meanrev_returns_gross,
    meanrev_turnover,
    slippage_bps=SLIPPAGE_BPS,
    model=SLIPPAGE_MODEL,
    volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
    turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
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

combined_turnover = (
    base_mom_weight * momentum_turnover.reindex(returns.index).fillna(0.0)
    + base_mr_weight * meanrev_turnover.reindex(returns.index).fillna(0.0)
)
combined_weights = (
    base_mom_weight * momentum_weights
    + base_mr_weight * meanrev_weights
)
combined_pre_impact_returns = base_mom_weight * momentum_returns + base_mr_weight * meanrev_returns
combined_transaction_costs = TRANSACTION_COST * combined_turnover
combined_slippage_costs = (
    base_mom_weight * momentum_slippage_costs.reindex(combined_pre_impact_returns.index).fillna(0.0)
    + base_mr_weight * meanrev_slippage_costs.reindex(combined_pre_impact_returns.index).fillna(0.0)
)
market_impact_result = estimate_market_impact(
    combined_weights,
    prices,
    volumes.reindex(index=prices.index, columns=prices.columns),
    portfolio_name="combined_inverse_vol",
    target_dollars=CAPACITY_TARGET_DOLLARS,
    impact_enabled=ENABLE_MARKET_IMPACT,
    model=MARKET_IMPACT_MODEL,
    max_adv_participation=MAX_ADV_PARTICIPATION,
    impact_coefficient=IMPACT_COEFFICIENT,
    adv_lookback_window=ADV_LOOKBACK_WINDOW,
    volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
)
combined_returns = combined_pre_impact_returns - market_impact_result.applied_costs.reindex(
    combined_pre_impact_returns.index
).fillna(0.0)
combined_returns.name = "combined_inverse_vol"
execution_costs = build_execution_cost_report(
    combined_turnover,
    combined_transaction_costs,
    combined_slippage_costs,
    market_impact_result,
    portfolio_name="combined_inverse_vol",
)
capacity_summary = market_impact_result.capacity_summary
market_impact_analysis = market_impact_result.daily_analysis
impact_enabled_returns = combined_pre_impact_returns - market_impact_result.daily_analysis[
    "modeled_market_impact_cost"
].reindex(combined_pre_impact_returns.index).fillna(0.0)
impact_enabled_returns.name = "combined_impact_enabled"
impact_diagnostic_tables = build_impact_diagnostic_tables(
    market_impact_result,
    portfolio_name="combined_inverse_vol",
)
capacity_sensitivity = pd.DataFrame()
impact_cost_contribution = pd.DataFrame()
impact_comparison = pd.DataFrame()
if ACTIVE_MODE.name == "sp500_research":
    momentum_impact_attribution = estimate_market_impact(
        base_mom_weight * momentum_weights,
        prices,
        volumes.reindex(index=prices.index, columns=prices.columns),
        portfolio_name="momentum_sleeve",
        target_dollars=CAPACITY_TARGET_DOLLARS,
        impact_enabled=False,
        model=MARKET_IMPACT_MODEL,
        max_adv_participation=MAX_ADV_PARTICIPATION,
        impact_coefficient=IMPACT_COEFFICIENT,
        adv_lookback_window=ADV_LOOKBACK_WINDOW,
        volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    )
    meanrev_impact_attribution = estimate_market_impact(
        base_mr_weight * meanrev_weights,
        prices,
        volumes.reindex(index=prices.index, columns=prices.columns),
        portfolio_name="mean_reversion_sleeve",
        target_dollars=CAPACITY_TARGET_DOLLARS,
        impact_enabled=False,
        model=MARKET_IMPACT_MODEL,
        max_adv_participation=MAX_ADV_PARTICIPATION,
        impact_coefficient=IMPACT_COEFFICIENT,
        adv_lookback_window=ADV_LOOKBACK_WINDOW,
        volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    )
    impact_cost_contribution = pd.DataFrame(
        [
            {
                "component": "Momentum Sleeve",
                "modeled_market_impact_cost": momentum_impact_attribution.daily_analysis[
                    "modeled_market_impact_cost"
                ].sum(),
            },
            {
                "component": "Mean Reversion Sleeve",
                "modeled_market_impact_cost": meanrev_impact_attribution.daily_analysis[
                    "modeled_market_impact_cost"
                ].sum(),
            },
            {
                "component": "Combined Net Trades",
                "modeled_market_impact_cost": market_impact_analysis[
                    "modeled_market_impact_cost"
                ].sum(),
            },
        ]
    )
    capacity_sensitivity = compute_capacity_sensitivity(
        combined_weights,
        prices,
        volumes.reindex(index=prices.index, columns=prices.columns),
        combined_pre_impact_returns,
        portfolio_name="sp500_combined_inverse_vol",
        target_dollars=CAPACITY_TARGET_DOLLARS,
        model=MARKET_IMPACT_MODEL,
        max_adv_participation=MAX_ADV_PARTICIPATION,
        impact_coefficient=IMPACT_COEFFICIENT,
        adv_lookback_window=ADV_LOOKBACK_WINDOW,
        volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    )

combined_cumulative = compute_cumulative_returns(combined_returns)
combined_metrics = compute_performance_metrics(combined_returns)
combined_drawdown = compute_drawdown_series(combined_cumulative)
combined_turnover_metrics = compute_turnover_metrics(combined_turnover)
combined_exposure_metrics = summarize_gross_exposure(combined_weights)
combined_gross_exposure = compute_gross_exposure(combined_weights)
combined_sector_exposure = summarize_sector_exposures(combined_weights, sector_map)

# =========================
# Historical universe survivorship comparison
# =========================
survivorship_comparison = pd.DataFrame()
survivor_only_returns = pd.Series(dtype=float)
survivor_only_cumulative = pd.Series(dtype=float)
if UNIVERSE_MODE == HISTORICAL_SP500:
    final_members = membership_mask.iloc[-1]
    survivor_mask = pd.DataFrame(
        [final_members.to_numpy()] * len(prices.index),
        index=prices.index,
        columns=prices.columns,
    )
    survivor_eligibility = combine_eligibility_masks(survivor_mask, liquidity_eligibility)
    survivor_momentum_weights = generate_momentum_weights(
        prices,
        lookback=MOMENTUM_LOOKBACK,
        top_n=TOP_N,
        rebalance_freq=REBALANCE_FREQ,
        vol_lookback=20,
        max_position_weight=MAX_POSITION_WEIGHT,
        eligibility_mask=survivor_eligibility,
    )
    survivor_momentum_weights = survivor_momentum_weights.mul(
        regime_multiplier_aligned,
        axis=0,
    ).where(survivor_mask, 0.0)
    survivor_momentum_gross, survivor_momentum_turnover = run_backtest(
        survivor_momentum_weights,
        returns,
        transaction_cost=TRANSACTION_COST,
        return_turnover=True,
    )
    survivor_momentum_returns = apply_execution_slippage(
        survivor_momentum_gross,
        survivor_momentum_turnover,
        slippage_bps=SLIPPAGE_BPS,
        model=SLIPPAGE_MODEL,
        volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
        volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
        turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
    )
    survivor_meanrev_weights = generate_mean_reversion_weights(
        prices,
        lookback=5,
        top_n=15,
        rebalance_freq=10,
        trend_window=40,
        min_pullback=-0.10,
        max_pullback=-0.03,
        weighting_scheme="equal",
        max_position_weight=MAX_POSITION_WEIGHT,
        eligibility_mask=survivor_eligibility,
    ).where(survivor_mask, 0.0)
    survivor_meanrev_gross, survivor_meanrev_turnover = run_backtest(
        survivor_meanrev_weights,
        returns,
        transaction_cost=TRANSACTION_COST,
        return_turnover=True,
    )
    survivor_meanrev_returns = apply_execution_slippage(
        survivor_meanrev_gross,
        survivor_meanrev_turnover,
        slippage_bps=SLIPPAGE_BPS,
        model=SLIPPAGE_MODEL,
        volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
        volatility_reference=SLIPPAGE_VOLATILITY_REFERENCE,
        turnover_multiplier=SLIPPAGE_TURNOVER_MULTIPLIER,
    )
    survivor_momentum_vol = survivor_momentum_returns.std()
    survivor_meanrev_vol = survivor_meanrev_returns.std()
    survivor_inv_momentum = 1 / survivor_momentum_vol if survivor_momentum_vol != 0 else 0
    survivor_inv_meanrev = 1 / survivor_meanrev_vol if survivor_meanrev_vol != 0 else 0
    survivor_weight_sum = survivor_inv_momentum + survivor_inv_meanrev
    survivor_momentum_allocation = (
        survivor_inv_momentum / survivor_weight_sum if survivor_weight_sum else 0.5
    )
    survivor_meanrev_allocation = (
        survivor_inv_meanrev / survivor_weight_sum if survivor_weight_sum else 0.5
    )
    survivor_pre_impact_returns = (
        survivor_momentum_allocation * survivor_momentum_returns
        + survivor_meanrev_allocation * survivor_meanrev_returns
    )
    survivor_combined_weights = (
        survivor_momentum_allocation * survivor_momentum_weights
        + survivor_meanrev_allocation * survivor_meanrev_weights
    )
    survivor_market_impact = estimate_market_impact(
        survivor_combined_weights,
        prices,
        volumes.reindex(index=prices.index, columns=prices.columns),
        portfolio_name="survivor_only_inverse_vol",
        target_dollars=CAPACITY_TARGET_DOLLARS,
        impact_enabled=ENABLE_MARKET_IMPACT,
        model=MARKET_IMPACT_MODEL,
        max_adv_participation=MAX_ADV_PARTICIPATION,
        impact_coefficient=IMPACT_COEFFICIENT,
        adv_lookback_window=ADV_LOOKBACK_WINDOW,
        volatility_window=SLIPPAGE_VOLATILITY_WINDOW,
    )
    survivor_only_returns = survivor_pre_impact_returns - survivor_market_impact.applied_costs
    survivor_only_returns.name = "survivor_only_inverse_vol"
    survivor_only_cumulative = compute_cumulative_returns(survivor_only_returns)
    point_in_time_metrics = compute_performance_metrics(combined_returns)
    survivor_only_metrics = compute_performance_metrics(survivor_only_returns)
    survivorship_comparison = pd.DataFrame(
        [
            {
                "universe": "point_in_time_membership",
                **point_in_time_metrics,
            },
            {
                "universe": "end_date_survivors_only",
                **survivor_only_metrics,
            },
        ]
    )
    survivorship_comparison["annualized_return_bias_vs_point_in_time"] = (
        survivorship_comparison["Annualized Return"]
        - point_in_time_metrics["Annualized Return"]
    )
    survivorship_comparison["sharpe_bias_vs_point_in_time"] = (
        survivorship_comparison["Sharpe Ratio"]
        - point_in_time_metrics["Sharpe Ratio"]
    )

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
vol_targeted_weights = combined_weights.mul(
    leverage_series.reindex(combined_weights.index).fillna(1.0),
    axis=0,
)
vol_targeted_exposure_metrics = summarize_gross_exposure(vol_targeted_weights)
vol_targeted_gross_exposure = compute_gross_exposure(vol_targeted_weights)
vol_targeted_sector_exposure = summarize_sector_exposures(vol_targeted_weights, sector_map)

# =========================
# Optional research-only constrained optimization
# =========================
optimized_sleeve_weights = None
if ENABLE_CONSTRAINED_OPTIMIZER:
    sleeve_returns = pd.concat(
        [
            momentum_returns.rename("momentum"),
            meanrev_returns.rename("mean_reversion"),
        ],
        axis=1,
    )
    optimized_sleeve_weights = optimize_long_only_weights(
        sleeve_returns,
        max_weight=OPTIMIZER_MAX_WEIGHT,
    )

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
# Optional Ken French factor decomposition
# =========================
factor_summary = pd.DataFrame()
rolling_factor_exposures = pd.DataFrame()
factor_analysis_error = None
if ENABLE_FACTOR_ANALYSIS:
    try:
        factor_summary, factor_data = run_factor_model_suite(
            {
                "combined_inverse_vol": combined_returns,
                "combined_vol_targeted": vol_targeted_returns,
            },
            frequency=FACTOR_FREQUENCY,
            include_ff5=ENABLE_FF5_ANALYSIS,
            use_hac=FACTOR_USE_HAC,
            hac_lags=FACTOR_HAC_LAGS,
            start_date=START_DATE,
            end_date=END_DATE,
        )
        rolling_dataset = "ff5" if FACTOR_ROLLING_MODEL == "ff5" else "ff3"
        rolling_factor_exposures = compute_rolling_factor_exposures(
            combined_returns,
            factor_data[rolling_dataset],
            model=FACTOR_ROLLING_MODEL,
            frequency=FACTOR_FREQUENCY,
            window=FACTOR_ROLLING_WINDOW,
            use_hac=FACTOR_USE_HAC,
            hac_lags=FACTOR_HAC_LAGS,
        )
    except Exception as exc:
        factor_analysis_error = str(exc)

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
# Execution realism comparison diagnostics
# =========================
if ACTIVE_MODE.name == "sp500_research":
    impact_enabled_metrics = compute_performance_metrics(impact_enabled_returns)
    slippage_only_metrics = compute_performance_metrics(combined_pre_impact_returns)
    impact_comparison_rows = [
        {
            "scenario": "S&P Slippage Only",
            "annualized_return": slippage_only_metrics["Annualized Return"],
            "sharpe": slippage_only_metrics["Sharpe Ratio"],
            "max_drawdown": slippage_only_metrics["Max Drawdown"],
        },
        {
            "scenario": "S&P Impact Enabled",
            "annualized_return": impact_enabled_metrics["Annualized Return"],
            "sharpe": impact_enabled_metrics["Sharpe Ratio"],
            "max_drawdown": impact_enabled_metrics["Max Drawdown"],
        },
    ]
    baseline_root = Path(REPORT_ROOT) / "production_baseline"
    for baseline_directory in sorted(baseline_root.glob("*"), reverse=True):
        metadata_path = baseline_directory / "experiment_metadata.json"
        metrics_path = baseline_directory / "summary_metrics.csv"
        if not metadata_path.exists() or not metrics_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        execution_assumptions = metadata.get("execution_assumptions", {})
        if execution_assumptions.get("market_impact_enabled", False):
            continue
        baseline_table = pd.read_csv(metrics_path)
        baseline_row = baseline_table[baseline_table["portfolio"] == "combined_inverse_vol"]
        if baseline_row.empty:
            continue
        baseline_row = baseline_row.iloc[0]
        impact_comparison_rows.append(
            {
                "scenario": "Baseline Latest Impact-Off",
                "annualized_return": baseline_row["Annualized Return"],
                "sharpe": baseline_row["Sharpe Ratio"],
                "max_drawdown": baseline_row["Max Drawdown"],
            }
        )
        break
    impact_comparison = pd.DataFrame(impact_comparison_rows)

# =========================
# Print metrics
# =========================
print(f"Execution: {SLIPPAGE_MODEL} slippage ({SLIPPAGE_BPS:.2f} base bps)")
capacity_row = capacity_summary.iloc[0]
impact_status = "enabled" if ENABLE_MARKET_IMPACT else "disabled"
if capacity_row["liquidity_data_available"]:
    print(
        f"Market Impact: {impact_status} ({MARKET_IMPACT_MODEL}; "
        f"${CAPACITY_TARGET_DOLLARS:,.0f} target; "
        f"estimated capacity ${capacity_row['estimated_capacity_dollars']:,.0f}; "
        f"ADV breaches {capacity_row['participation_breach_fraction']:.2%})"
    )
else:
    print(f"Market Impact: {impact_status}; volume unavailable for capacity diagnostics")
if ACTIVE_MODE.name == "sp500_research" and not market_impact_result.trade_details.empty:
    summary_row = impact_diagnostic_tables["execution_dashboard_summary"].iloc[0]
    expensive_trade = impact_diagnostic_tables["most_expensive_trades"].iloc[0]
    print(
        f"Execution Diagnostics: avg participation {summary_row['average_participation_rate']:.2%}; "
        f"largest impact trade {expensive_trade['ticker']} on "
        f"{expensive_trade['date'].date()} ({expensive_trade['modeled_market_impact_cost']:.4%})"
    )
    print(
        f"Liquidity Bottleneck: {summary_row['binding_liquidity_ticker']} at "
        f"{summary_row['binding_participation_rate']:.2%} ADV participation"
    )
if UNIVERSE_MODE == HISTORICAL_SP500 and not survivorship_comparison.empty:
    survivor_row = survivorship_comparison[
        survivorship_comparison["universe"] == "end_date_survivors_only"
    ].iloc[0]
    print(
        f"Survivorship Comparison: survivor-only annual return bias "
        f"{survivor_row['annualized_return_bias_vs_point_in_time']:.2%}; "
        f"Sharpe bias {survivor_row['sharpe_bias_vs_point_in_time']:.4f}"
    )

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

print("\nCore CAPM Summary:")
print(f"Inverse-Vol Alpha Annualized: {combined_capm_summary['Alpha Annualized']:.6f}")
print(f"Inverse-Vol Beta: {combined_capm_summary['Beta']:.6f}")
print(f"Inverse-Vol Alpha p-value: {combined_capm_summary['Alpha p-value']:.6f}")

if not factor_summary.empty:
    ff3_primary = factor_summary[
        (factor_summary["portfolio"] == "combined_inverse_vol")
        & (factor_summary["model"] == "FF3")
    ].iloc[0]
    print("\nCore Fama-French 3-Factor Summary:")
    print(f"Alpha Annualized: {ff3_primary['alpha_annualized']:.6f}")
    print(f"Market Beta: {ff3_primary['market_beta']:.6f}")
    print(f"SMB Exposure: {ff3_primary['smb']:.6f}")
    print(f"HML Exposure: {ff3_primary['hml']:.6f}")
elif factor_analysis_error:
    print(f"\nFactor analysis unavailable: {factor_analysis_error}")

if VERBOSE:
    print("\nMomentum Strategy Metrics:")
    for key, value in momentum_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nMean Reversion Strategy Metrics:")
    for key, value in meanrev_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nInverse-Vol Combined Weights:")
    print(f"Momentum Weight: {base_mom_weight:.4f}")
    print(f"Mean Reversion Weight: {base_mr_weight:.4f}")

    print("\nGross Exposure Summary:")
    for key, value in combined_exposure_metrics.items():
        print(f"Inverse-Vol Combined {key}: {value:.4f}")
    for key, value in vol_targeted_exposure_metrics.items():
        print(f"Vol-Targeted Combined {key}: {value:.4f}")

    if not combined_sector_exposure.empty:
        print("\nInverse-Vol Combined Sector Exposure:")
        print(combined_sector_exposure.to_string(float_format=lambda value: f"{value:.4f}"))
        print("\nVol-Targeted Combined Sector Exposure:")
        print(vol_targeted_sector_exposure.to_string(float_format=lambda value: f"{value:.4f}"))

    if optimized_sleeve_weights is not None:
        print("\nResearch-Only Constrained Minimum-Variance Sleeve Weights:")
        for key, value in optimized_sleeve_weights.items():
            print(f"{key}: {value:.4f}")

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

    if not factor_summary.empty:
        print("\nKen French Factor Decomposition:")
        print(factor_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\nMomentum Turnover Metrics:")
    for key, value in momentum_turnover_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nMean Reversion Turnover Metrics:")
    for key, value in meanrev_turnover_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nInverse-Vol Combined Turnover Metrics:")
    for key, value in combined_turnover_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nMarket Impact And Capacity Diagnostics:")
    print(capacity_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(
        f"Turnover-Adjusted Total Execution Cost: "
        f"{execution_costs['total_execution_cost'].sum() / combined_turnover.sum():.6f}"
    )

    print(f"\nAverage Vol-Target Leverage: {leverage_series.mean():.4f}")
    print(f"Max Vol-Target Leverage: {leverage_series.max():.4f}")

    if UNIVERSE_MODE == HISTORICAL_SP500:
        print("\nAnnual Historical Membership Changes:")
        print(annual_membership_changes.to_string(index=False))
        if not unavailable_historical_prices.empty:
            print("\nHistorical Members Without Vendor Price History:")
            print(unavailable_historical_prices.to_string(index=False))

    print_regime_metrics("Combined MA Regime Analysis", combined_ma_regime)
    print_regime_metrics("Combined Volatility Regime Analysis", combined_vol_regime)

# =========================
# Plots
# =========================
figures = {
    "comparison_dashboard": plot_backtest_dashboard(
        cumulative_returns={
            "Momentum": momentum_cumulative,
            "Mean Reversion": meanrev_cumulative,
            "Combined Inverse-Vol": combined_cumulative,
            "Vol-Targeted Combined": vol_targeted_cumulative,
            "SPY": benchmark_cumulative,
        },
        drawdowns={
            "Momentum": momentum_drawdown,
            "Mean Reversion": meanrev_drawdown,
            "Combined Inverse-Vol": combined_drawdown,
            "Vol-Targeted Combined": vol_targeted_drawdown,
            "SPY": benchmark_drawdown,
        },
        gross_exposures={
            "Combined Inverse-Vol": combined_gross_exposure,
            "Vol-Targeted Combined": vol_targeted_gross_exposure,
        },
        metrics_summary={
            "Momentum": momentum_metrics,
            "Mean Rev.": meanrev_metrics,
            "Combined": combined_metrics,
            "Vol Target": vol_targeted_metrics,
            "SPY": benchmark_metrics,
        },
        capm_summary=combined_capm_summary,
        title=f"{ACTIVE_MODE.name.replace('_', ' ').title()} Dashboard",
    )
}

if SAVE_INDIVIDUAL_PLOTS:
    figures["momentum_vs_benchmark"] = plot_strategy_vs_benchmark(
        momentum_cumulative,
        benchmark_cumulative.reindex(momentum_cumulative.index),
        strategy_label="Momentum",
        benchmark_label="SPY",
    )
    figures["momentum_drawdown"] = plot_drawdown(momentum_drawdown, title="Momentum Drawdown")
    figures["mean_reversion_vs_benchmark"] = plot_strategy_vs_benchmark(
        meanrev_cumulative,
        benchmark_cumulative.reindex(meanrev_cumulative.index),
        strategy_label="Mean Reversion",
        benchmark_label="SPY",
    )
    figures["mean_reversion_drawdown"] = plot_drawdown(meanrev_drawdown, title="Mean Reversion Drawdown")
    figures["combined_inverse_vol_vs_benchmark"] = plot_strategy_vs_benchmark(
        combined_cumulative,
        benchmark_cumulative.reindex(combined_cumulative.index),
        strategy_label="Combined Inverse-Vol",
        benchmark_label="SPY",
    )
    figures["combined_inverse_vol_drawdown"] = plot_drawdown(
        combined_drawdown, title="Combined Inverse-Vol Drawdown"
    )
    figures["vol_targeted_combined_vs_benchmark"] = plot_strategy_vs_benchmark(
        vol_targeted_cumulative,
        benchmark_cumulative.reindex(vol_targeted_cumulative.index),
        strategy_label="Vol-Targeted Combined",
        benchmark_label="SPY",
    )
    figures["vol_targeted_combined_drawdown"] = plot_drawdown(
        vol_targeted_drawdown, title="Vol-Targeted Combined Drawdown"
    )
    figures["combined_inverse_vol_gross_exposure"] = plot_gross_exposure(
        combined_gross_exposure,
        title="Combined Inverse-Vol Gross Exposure",
        label="Combined Inverse-Vol",
    )
    figures["vol_targeted_combined_gross_exposure"] = plot_gross_exposure(
        vol_targeted_gross_exposure,
        title="Vol-Targeted Combined Gross Exposure",
        label="Vol-Targeted Combined",
    )

if not rolling_factor_exposures.empty:
    figures["rolling_factor_exposures"] = plot_rolling_factor_exposures(
        rolling_factor_exposures,
        title_prefix="Combined Inverse-Vol",
    )

if ACTIVE_MODE.name == "sp500_research" and not capacity_sensitivity.empty:
    figures["execution_dashboard"] = plot_execution_dashboard(
        execution_costs.reset_index(),
        market_impact_analysis.reset_index(),
        impact_cost_contribution,
        capacity_summary,
        title="S&P 500 Research Execution Dashboard",
    )
    figures["capacity_curves"] = plot_capacity_curves(
        capacity_sensitivity,
        title="S&P 500 Research Capacity Sensitivity",
    )
    figures["impact_comparison"] = plot_impact_comparison(
        impact_comparison,
        cumulative_paths={
            "S&P Slippage Only": compute_cumulative_returns(combined_pre_impact_returns),
            "S&P Impact Enabled": compute_cumulative_returns(impact_enabled_returns),
        },
        title="S&P Execution And Universe Comparison",
    )

if UNIVERSE_MODE == HISTORICAL_SP500 and not survivorship_comparison.empty:
    figures["survivorship_bias_analysis"] = plot_survivorship_bias_analysis(
        combined_cumulative,
        survivor_only_cumulative,
        universe_turnover,
        survivorship_comparison,
        title="Point-In-Time S&P 500 Survivorship Bias Analysis",
    )

if SAVE_REPORTS:
    report_directory, timestamp = create_experiment_directory(REPORT_ROOT, ACTIVE_MODE.name)
    metric_rows = []
    for name, metrics in (
        ("momentum", momentum_metrics),
        ("mean_reversion", meanrev_metrics),
        ("combined_inverse_vol", combined_metrics),
        ("combined_vol_targeted", vol_targeted_metrics),
        ("benchmark_spy", benchmark_metrics),
    ):
        metric_rows.append({"portfolio": name, **metrics})
    capm_table = pd.DataFrame(
        [
            {"portfolio": "combined_inverse_vol", **combined_capm_summary},
            {"portfolio": "combined_vol_targeted", **vol_targeted_capm_summary},
        ]
    )
    tables = {
        "summary_metrics": pd.DataFrame(metric_rows),
        "capm_summary": capm_table,
        "gross_exposure_summary": pd.DataFrame(
            [
                {"portfolio": "combined_inverse_vol", **combined_exposure_metrics},
                {"portfolio": "combined_vol_targeted", **vol_targeted_exposure_metrics},
            ]
        ),
        "daily_returns": pd.concat(
            [
                momentum_returns.rename("momentum"),
                meanrev_returns.rename("mean_reversion"),
                combined_returns.rename("combined_inverse_vol"),
                vol_targeted_returns.rename("combined_vol_targeted"),
                benchmark_returns.rename("benchmark_spy"),
            ],
            axis=1,
        ).reset_index(names="date"),
        "execution_costs": execution_costs.reset_index(),
        "capacity_summary": capacity_summary,
        "market_impact_analysis": market_impact_analysis.reset_index(),
    }
    if not combined_sector_exposure.empty:
        tables["combined_inverse_vol_sector_exposure"] = combined_sector_exposure.reset_index(names="sector")
        tables["combined_vol_targeted_sector_exposure"] = vol_targeted_sector_exposure.reset_index(names="sector")
    if not factor_summary.empty:
        tables["factor_summary"] = factor_summary
        tables["rolling_factor_exposures"] = rolling_factor_exposures
    if ACTIVE_MODE.name == "sp500_research" and not capacity_sensitivity.empty:
        tables["capacity_sensitivity"] = capacity_sensitivity
        tables["impact_cost_contribution"] = impact_cost_contribution
        tables["impact_comparison"] = impact_comparison
        tables.update(impact_diagnostic_tables)
    if UNIVERSE_MODE == HISTORICAL_SP500:
        tables["constituent_history"] = constituent_history
        tables["universe_turnover"] = universe_turnover
        tables["annual_membership_changes"] = annual_membership_changes
        tables["survivorship_comparison"] = survivorship_comparison
        tables["unavailable_historical_prices"] = unavailable_historical_prices
    save_tables(report_directory, tables)
    if SAVE_PLOTS:
        save_figures(report_directory, figures)
    save_metadata(
        report_directory,
        timestamp,
        ACTIVE_MODE,
        {
            "description": universe_description,
            "universe_mode": UNIVERSE_MODE,
            "requested_count": len(universe_tickers),
            "retained_count": len(prices.columns),
            "benchmark": BENCHMARK_TICKER[0],
            "start_date": START_DATE,
            "end_date": END_DATE,
            "constituent_history_file": (
                HISTORICAL_CONSTITUENTS_FILE if UNIVERSE_MODE == HISTORICAL_SP500 else None
            ),
            "unavailable_historical_price_count": len(unavailable_historical_prices),
        },
        {
            "transaction_cost": TRANSACTION_COST,
            "slippage_model": SLIPPAGE_MODEL,
            "slippage_bps": SLIPPAGE_BPS,
            "market_impact_enabled": ENABLE_MARKET_IMPACT,
            "market_impact_model": MARKET_IMPACT_MODEL,
            "capacity_target_dollars": CAPACITY_TARGET_DOLLARS,
            "max_adv_participation": MAX_ADV_PARTICIPATION,
            "impact_coefficient": IMPACT_COEFFICIENT,
            "liquidity_filter": ENABLE_LIQUIDITY_FILTER,
            "max_position_weight": MAX_POSITION_WEIGHT,
            "factor_analysis_enabled": ENABLE_FACTOR_ANALYSIS,
            "factor_frequency": FACTOR_FREQUENCY,
            "ff5_analysis_enabled": ENABLE_FF5_ANALYSIS,
        },
        {
            "momentum_lookback": MOMENTUM_LOOKBACK,
            "top_n": TOP_N,
            "rebalance_freq": REBALANCE_FREQ,
            "min_stock_price": MIN_STOCK_PRICE,
            "max_missing_data_fraction": MAX_MISSING_DATA_FRACTION,
            "adv_lookback_window": ADV_LOOKBACK_WINDOW,
            "min_average_dollar_volume": MIN_AVERAGE_DOLLAR_VOLUME,
            "factor_rolling_model": FACTOR_ROLLING_MODEL,
            "factor_rolling_window": FACTOR_ROLLING_WINDOW,
        },
    )
    print(f"\nReport saved: {report_directory}")

# Show all plots at once
plt.show()
