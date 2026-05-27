import pandas as pd


def compute_average_dollar_volume(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback_window: int = 20,
) -> pd.DataFrame:
    """
    Compute lagged rolling average dollar volume for each asset.

    The one-day shift ensures today's strategy selection uses liquidity
    observed no later than the previous trading day.
    """
    if volumes is None or volumes.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    if lookback_window < 1:
        raise ValueError("lookback_window must be at least 1.")

    aligned_volumes = volumes.reindex(index=prices.index, columns=prices.columns)
    dollar_volume = prices * aligned_volumes
    return dollar_volume.rolling(lookback_window).mean().shift(1)


def build_liquidity_eligibility_mask(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_average_dollar_volume: float,
    lookback_window: int = 20,
) -> pd.DataFrame | None:
    """
    Return a daily eligibility mask, or None when volume is unavailable.
    """
    if volumes is None or volumes.empty or volumes.notna().sum().sum() == 0:
        return None

    adv = compute_average_dollar_volume(prices, volumes, lookback_window)
    return adv >= min_average_dollar_volume
