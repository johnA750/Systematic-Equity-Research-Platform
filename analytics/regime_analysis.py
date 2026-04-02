import numpy as np
import pandas as pd

from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics


def classify_ma_regime(
    benchmark_prices: pd.Series,
    ma_window: int = 200,
) -> pd.Series:
    """
    Classify market regime using benchmark price vs moving average.
    
    Returns a series with values:
    - 'bull'
    - 'bear'
    """
    benchmark_prices = benchmark_prices.dropna().copy()
    ma = benchmark_prices.rolling(ma_window).mean()

    regime = pd.Series(index=benchmark_prices.index, dtype="object")
    regime[benchmark_prices >= ma] = "bull"
    regime[benchmark_prices < ma] = "bear"

    return regime


def classify_vol_regime(
    benchmark_returns: pd.Series,
    vol_window: int = 20,
    threshold_quantile: float = 0.5,
) -> pd.Series:
    """
    Classify market regime using rolling realized volatility.
    
    Returns:
    - 'high_vol'
    - 'low_vol'
    """
    benchmark_returns = benchmark_returns.dropna().copy()
    realized_vol = benchmark_returns.rolling(vol_window).std()

    threshold = realized_vol.quantile(threshold_quantile)

    regime = pd.Series(index=benchmark_returns.index, dtype="object")
    regime[realized_vol >= threshold] = "high_vol"
    regime[realized_vol < threshold] = "low_vol"

    return regime


def compute_regime_metrics(
    strategy_returns: pd.Series,
    regime_labels: pd.Series,
    strategy_turnover: pd.Series | None = None,
) -> dict:
    """
    Compute performance metrics for each regime bucket.
    """
    df = pd.concat(
        [
            strategy_returns.rename("returns"),
            regime_labels.rename("regime"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        raise ValueError("No overlapping data between strategy returns and regime labels.")

    if strategy_turnover is not None:
        df = pd.concat(
            [
                df,
                strategy_turnover.rename("turnover"),
            ],
            axis=1,
        )

    results = {}

    for regime in sorted(df["regime"].dropna().unique()):
        subset = df[df["regime"] == regime]

        if subset.empty:
            continue

        perf = compute_performance_metrics(subset["returns"])

        if strategy_turnover is not None and "turnover" in subset.columns:
            turnover_metrics = compute_turnover_metrics(subset["turnover"].fillna(0.0))
        else:
            turnover_metrics = None

        results[regime] = {
            "performance": perf,
            "turnover": turnover_metrics,
            "num_observations": len(subset),
        }

    return results


def print_regime_metrics(title: str, regime_results: dict) -> None:
    """
    Pretty-print regime metrics.
    """
    print(f"\n{title}")
    for regime, result in regime_results.items():
        print(f"\n{regime.upper()} ({result['num_observations']} observations):")

        for key, value in result["performance"].items():
            print(f"{key}: {value:.4f}")

        if result["turnover"] is not None:
            print("Turnover:")
            for key, value in result["turnover"].items():
                print(f"  {key}: {value:.4f}")