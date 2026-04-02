import pandas as pd
import numpy as np


def apply_vol_targeting(
    returns: pd.Series,
    target_annual_vol: float = 0.12,
    rolling_window: int = 20,
    max_leverage: float = 1.5,
    min_leverage: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """
    Apply trailing volatility targeting to a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    target_annual_vol : float
        Desired annualized volatility, e.g. 0.12 for 12%.
    rolling_window : int
        Window for realized daily vol estimate.
    max_leverage : float
        Cap on leverage multiplier.
    min_leverage : float
        Floor on leverage multiplier.
    periods_per_year : int
        Usually 252.

    Returns
    -------
    targeted_returns : pd.Series
        Vol-targeted daily returns.
    leverage_series : pd.Series
        Daily leverage multiplier used.
    """
    returns = returns.copy()

    realized_daily_vol = returns.rolling(rolling_window).std()
    realized_annual_vol = realized_daily_vol * np.sqrt(periods_per_year)

    leverage = target_annual_vol / realized_annual_vol
    leverage = leverage.replace([np.inf, -np.inf], np.nan)
    leverage = leverage.clip(lower=min_leverage, upper=max_leverage)

    # Shift by 1 day so you only use information known at the close of previous day
    leverage = leverage.shift(1).fillna(1.0)

    targeted_returns = returns * leverage
    targeted_returns.name = f"{returns.name}_vol_targeted" if returns.name else "vol_targeted"

    return targeted_returns, leverage