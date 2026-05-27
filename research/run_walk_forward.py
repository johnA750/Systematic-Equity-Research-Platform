import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics.rolling import compute_rolling_performance
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
    WALK_FORWARD_EXPORT_RESULTS,
    WALK_FORWARD_REPORT_DIR,
    WALK_FORWARD_SAVE_PLOTS,
    WALK_FORWARD_MOMENTUM_GRID,
    WALK_FORWARD_MEAN_REVERSION_GRID,
)
from data.liquidity import build_liquidity_eligibility_mask
from data.loader import clean_price_data, load_market_data, load_price_data
from data.sp500 import get_sp500_constituents
from analytics.returns import compute_daily_returns
from reporting.plots import plot_rolling_performance
from research.walk_forward import export_walk_forward_results, run_walk_forward_evaluation


CAPM_USE_HAC = True
CAPM_HAC_LAGS = 5


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
        capm_use_hac=CAPM_USE_HAC,
        capm_hac_lags=CAPM_HAC_LAGS,
    )

    rolling_metrics = compute_rolling_performance(
        results["oos_returns"],
        benchmark_returns.reindex(results["oos_returns"].index),
        window=WALK_FORWARD_ROLLING_METRICS_WINDOW,
        use_hac=CAPM_USE_HAC,
        hac_lags=CAPM_HAC_LAGS,
    )
    results["rolling_metrics"] = rolling_metrics

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
    print("\nSelected Configuration Out-of-Sample Results:")
    print(results["selected_results"][selected_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    parameter_summary = results["parameter_summary"]
    median_row = parameter_summary.iloc[len(parameter_summary) // 2]
    print("\nBest Parameter Set by Median Test Sharpe:")
    print(parameter_summary.head(1).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMedian Parameter Set by Median Test Sharpe:")
    print(median_row.to_frame().T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSelection Stability Summary:")
    print(results["stability_summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    report_path = Path(WALK_FORWARD_REPORT_DIR)
    if WALK_FORWARD_EXPORT_RESULTS:
        export_walk_forward_results(results, report_path)
        print(f"\nCSV research tables saved under: {report_path}")

    figure = plot_rolling_performance(rolling_metrics, title_prefix="Selected OOS")
    if WALK_FORWARD_SAVE_PLOTS:
        report_path.mkdir(parents=True, exist_ok=True)
        plot_path = report_path / "rolling_performance.png"
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Rolling performance chart saved to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
