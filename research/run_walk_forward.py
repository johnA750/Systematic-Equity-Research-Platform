import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics.rolling import compute_rolling_performance
from config.modes import WALK_FORWARD_VALIDATION
from config.config import (
    TICKERS,
    BENCHMARK_TICKER,
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
    WALK_FORWARD_MODE,
    WALK_FORWARD_TRAIN_LENGTH,
    WALK_FORWARD_TEST_LENGTH,
    WALK_FORWARD_STEP_LENGTH,
    WALK_FORWARD_ROLLING_METRICS_WINDOW,
    WALK_FORWARD_MOMENTUM_GRID,
    WALK_FORWARD_MEAN_REVERSION_GRID,
    PRODUCTION_MOMENTUM_PARAMS,
    PRODUCTION_MEAN_REVERSION_PARAMS,
    REPORT_ROOT,
    SAVE_REPORTS,
    SAVE_PLOTS,
    VERBOSE_ANALYTICS,
)
from data.liquidity import build_liquidity_eligibility_mask
from data.loader import clean_price_data, load_market_data, load_price_data
from data.sp500 import get_sp500_constituents
from analytics.returns import compute_daily_returns
from reporting.plots import plot_rolling_performance
from reporting.experiments import (
    create_experiment_directory,
    save_figures,
    save_metadata,
    save_tables,
)
from research.walk_forward import run_walk_forward_evaluation


CAPM_USE_HAC = True
CAPM_HAC_LAGS = 5
USE_SP500_UNIVERSE = WALK_FORWARD_VALIDATION.use_sp500_universe
ENABLE_LIQUIDITY_FILTER = WALK_FORWARD_VALIDATION.enable_liquidity_filter
MAX_POSITION_WEIGHT = WALK_FORWARD_VALIDATION.max_position_weight
SLIPPAGE_MODEL = WALK_FORWARD_VALIDATION.slippage_model


def load_research_data():
    if USE_SP500_UNIVERSE:
        constituents = get_sp500_constituents()
        universe_tickers = [record["ticker"] for record in constituents]
        universe_name = "Current S&P 500"
    else:
        universe_tickers = TICKERS
        universe_name = "Configured ticker list"

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
    if prices.empty:
        raise ValueError("No tradable equities remained after universe data filtering.")

    liquidity_eligibility = (
        build_liquidity_eligibility_mask(
            prices,
            volumes.reindex(index=prices.index, columns=prices.columns),
            min_average_dollar_volume=MIN_AVERAGE_DOLLAR_VOLUME,
            lookback_window=ADV_LOOKBACK_WINDOW,
        )
        if ENABLE_LIQUIDITY_FILTER else None
    )

    benchmark_prices_df = clean_price_data(
        load_price_data(BENCHMARK_TICKER, START_DATE, END_DATE)
    )
    benchmark_prices = benchmark_prices_df.iloc[:, 0]
    benchmark_returns = compute_daily_returns(benchmark_prices_df).iloc[:, 0]
    return (
        prices,
        compute_daily_returns(prices),
        benchmark_prices,
        benchmark_returns,
        liquidity_eligibility,
        universe_name,
        len(universe_tickers),
    )


def main():
    (
        prices,
        returns,
        benchmark_prices,
        benchmark_returns,
        liquidity_eligibility,
        universe_name,
        requested_count,
    ) = load_research_data()
    print(f"Walk-Forward Universe: {universe_name} ({len(prices.columns)} retained of {requested_count} requested)")
    print(
        f"Walk-Forward Design: {WALK_FORWARD_MODE}; "
        f"train={WALK_FORWARD_TRAIN_LENGTH}, test={WALK_FORWARD_TEST_LENGTH}, "
        f"step={WALK_FORWARD_STEP_LENGTH} observations"
    )
    print(
        f"Parameter Grid: {len(WALK_FORWARD_MOMENTUM_GRID)} momentum x "
        f"{len(WALK_FORWARD_MEAN_REVERSION_GRID)} mean-reversion configurations"
    )
    if ENABLE_LIQUIDITY_FILTER and liquidity_eligibility is not None:
        print(
            f"Liquidity Filter: ADV >= ${MIN_AVERAGE_DOLLAR_VOLUME:,.0f}; "
            f"{ADV_LOOKBACK_WINDOW}-day lagged window"
        )
    elif ENABLE_LIQUIDITY_FILTER:
        print("Liquidity Filter: volume unavailable; continuing with price-only behavior")

    results = run_walk_forward_evaluation(
        prices=prices,
        returns=returns,
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns,
        liquidity_eligibility=liquidity_eligibility,
        momentum_grid=WALK_FORWARD_MOMENTUM_GRID,
        meanrev_grid=WALK_FORWARD_MEAN_REVERSION_GRID,
        train_length=WALK_FORWARD_TRAIN_LENGTH,
        test_length=WALK_FORWARD_TEST_LENGTH,
        step_length=WALK_FORWARD_STEP_LENGTH,
        mode=WALK_FORWARD_MODE,
        transaction_cost=TRANSACTION_COST,
        max_position_weight=MAX_POSITION_WEIGHT,
        slippage_settings={
            "slippage_bps": SLIPPAGE_BPS,
            "model": SLIPPAGE_MODEL,
            "volatility_window": SLIPPAGE_VOLATILITY_WINDOW,
            "volatility_reference": SLIPPAGE_VOLATILITY_REFERENCE,
            "turnover_multiplier": SLIPPAGE_TURNOVER_MULTIPLIER,
        },
        fixed_momentum_params=PRODUCTION_MOMENTUM_PARAMS,
        fixed_meanrev_params=PRODUCTION_MEAN_REVERSION_PARAMS,
        capm_use_hac=CAPM_USE_HAC,
        capm_hac_lags=CAPM_HAC_LAGS,
    )

    selected_rolling_metrics = compute_rolling_performance(
        results["oos_returns"],
        benchmark_returns.reindex(results["oos_returns"].index),
        window=WALK_FORWARD_ROLLING_METRICS_WINDOW,
        use_hac=CAPM_USE_HAC,
        hac_lags=CAPM_HAC_LAGS,
    )
    fixed_rolling_metrics = compute_rolling_performance(
        results["fixed_oos_returns"],
        benchmark_returns.reindex(results["fixed_oos_returns"].index),
        window=WALK_FORWARD_ROLLING_METRICS_WINDOW,
        use_hac=CAPM_USE_HAC,
        hac_lags=CAPM_HAC_LAGS,
    )
    results["rolling_metrics"] = selected_rolling_metrics
    results["fixed_rolling_metrics"] = fixed_rolling_metrics

    selected_columns = [
        "window_id",
        "parameter_id",
        "train_start",
        "test_start",
        "train_sharpe",
        "test_sharpe",
        "sharpe_degradation",
        "test_annualized_return",
        "test_beta",
        "test_alpha_annualized",
        "test_alpha_p_value",
    ]
    parameter_summary = results["parameter_summary"]
    median_row = parameter_summary.iloc[len(parameter_summary) // 2]
    print("\nSelection Stability Summary:")
    print(results["stability_summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSelected vs Fixed Production OOS Comparison:")
    print(results["path_comparison"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest Parameter Set by Median Test Sharpe:")
    print(parameter_summary.head(1).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if VERBOSE_ANALYTICS:
        print("\nSelected Configuration Out-of-Sample Results:")
        print(results["selected_results"][selected_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nFixed Production Parameter Out-of-Sample Results:")
        print(results["fixed_parameter_results"][selected_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nMedian Parameter Set by Median Test Sharpe:")
        print(median_row.to_frame().T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    selected_figure = plot_rolling_performance(selected_rolling_metrics, title_prefix="Selected OOS")
    fixed_figure = plot_rolling_performance(fixed_rolling_metrics, title_prefix="Fixed Production OOS")
    if SAVE_REPORTS:
        report_path, timestamp = create_experiment_directory(
            REPORT_ROOT, WALK_FORWARD_VALIDATION.name
        )
        save_tables(report_path, results)
        if SAVE_PLOTS:
            save_figures(
                report_path,
                {
                    "selected_rolling_performance": selected_figure,
                    "fixed_production_rolling_performance": fixed_figure,
                },
            )
        save_metadata(
            report_path,
            timestamp,
            WALK_FORWARD_VALIDATION,
            {
                "description": universe_name,
                "requested_count": requested_count,
                "retained_count": len(prices.columns),
                "benchmark": BENCHMARK_TICKER[0],
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
            {
                "transaction_cost": TRANSACTION_COST,
                "slippage_model": SLIPPAGE_MODEL,
                "slippage_bps": SLIPPAGE_BPS,
                "liquidity_filter": ENABLE_LIQUIDITY_FILTER,
                "max_position_weight": MAX_POSITION_WEIGHT,
            },
            {
                "walk_forward_mode": WALK_FORWARD_MODE,
                "train_length": WALK_FORWARD_TRAIN_LENGTH,
                "test_length": WALK_FORWARD_TEST_LENGTH,
                "step_length": WALK_FORWARD_STEP_LENGTH,
                "rolling_metrics_window": WALK_FORWARD_ROLLING_METRICS_WINDOW,
                "fixed_production_momentum_params": PRODUCTION_MOMENTUM_PARAMS,
                "fixed_production_mean_reversion_params": PRODUCTION_MEAN_REVERSION_PARAMS,
                "momentum_grid": WALK_FORWARD_MOMENTUM_GRID,
                "mean_reversion_grid": WALK_FORWARD_MEAN_REVERSION_GRID,
            },
        )
        print(f"\nReport saved: {report_path}")

    plt.show()


if __name__ == "__main__":
    main()
