import pandas as pd


def compute_regime_multiplier(benchmark_prices, ma_window=200, sensitivity=5.0):
    """
    Compute a partial exposure regime multiplier using benchmark trend.

    Parameters:
        benchmark_prices (pd.Series): benchmark price series
        ma_window (int): moving average window
        sensitivity (float): controls how fast exposure changes

    Returns:
        pd.Series: exposure multiplier between 0 and 1
    """
    moving_avg = benchmark_prices.rolling(ma_window).mean()

    trend_strength = (benchmark_prices / moving_avg) - 1
    trend_strength = trend_strength.fillna(0.0)

    # Map trend strength into [0, 1]
    multiplier = 0.5 + sensitivity * trend_strength

    # Clip to valid exposure range
    multiplier = multiplier.clip(lower=0.0, upper=1.0)

    return multiplier