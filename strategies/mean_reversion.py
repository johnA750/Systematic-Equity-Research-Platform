import pandas as pd
import numpy as np


def generate_mean_reversion_weights(
    prices,
    lookback=5,
    top_n=10,
    rebalance_freq=5,
    trend_window=50,
    zscore_window=20,
    min_zscore=1.0,
    use_trend_filter=True,
    use_positive_reversal_filter=False,
    weighting_scheme="zscore",
):
    """
    Long-only mean reversion with optional trend filter and z-score ranking.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data.
    lookback : int
        Short-term return lookback used for optional reversal confirmation.
    top_n : int
        Max number of names to hold.
    rebalance_freq : int
        Rebalance every N rows.
    trend_window : int
        Rolling window for long-term trend filter.
    zscore_window : int
        Rolling window for z-score calculation.
    min_zscore : float
        Minimum absolute oversold threshold. A stock must have z-score <= -min_zscore.
    use_trend_filter : bool
        If True, only buy names above their rolling mean trend filter.
    use_positive_reversal_filter : bool
        If True, require today's return to be positive after being oversold.
    weighting_scheme : str
        One of:
        - "zscore": weights proportional to oversold z-score magnitude
        - "equal": equal weight selected names
        - "hybrid": average of z-score rank and short-term drawdown rank
    """

    prices = prices.copy()

    # Core ingredients
    rolling_mean = prices.rolling(zscore_window).mean()
    rolling_std = prices.rolling(zscore_window).std()

    zscore = (prices - rolling_mean) / rolling_std
    short_returns = prices.pct_change(lookback)
    daily_returns = prices.pct_change()

    # Trend filter: above longer-term moving average
    long_trend = prices / prices.rolling(trend_window).mean() - 1

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i % rebalance_freq == 0:
            z = zscore.loc[date]
            r = short_returns.loc[date]
            trend = long_trend.loc[date]
            one_day = daily_returns.loc[date]

            eligible = pd.Series(True, index=prices.columns)

            # Require enough data
            eligible &= z.notna()
            eligible &= r.notna()

            # Oversold condition
            eligible &= z <= -abs(min_zscore)

            # Optional trend filter
            if use_trend_filter:
                eligible &= trend > 0

            # Optional reversal confirmation:
            # stock is oversold, but today's move is positive
            if use_positive_reversal_filter:
                eligible &= one_day > 0

            candidates = prices.columns[eligible.fillna(False)]

            if len(candidates) == 0:
                last_weights = pd.Series(0.0, index=prices.columns)
            else:
                candidate_z = z.loc[candidates]
                candidate_r = r.loc[candidates]

                # Most negative z-scores are most oversold
                selected = candidate_z.nsmallest(min(top_n, len(candidate_z)))

                new_weights = pd.Series(0.0, index=prices.columns)

                if weighting_scheme == "equal":
                    new_weights.loc[selected.index] = 1.0

                elif weighting_scheme == "zscore":
                    # Convert negative z-score magnitude into positive scores
                    scores = -selected
                    scores = scores.clip(lower=0)
                    new_weights.loc[scores.index] = scores.values

                elif weighting_scheme == "hybrid":
                    # Blend oversold z-score magnitude and short-term drawdown magnitude
                    selected_r = candidate_r.loc[selected.index]

                    z_scores = (-selected).rank(method="average")
                    r_scores = (-selected_r).rank(method="average")

                    hybrid_scores = 0.5 * z_scores + 0.5 * r_scores
                    new_weights.loc[hybrid_scores.index] = hybrid_scores.values

                else:
                    raise ValueError("weighting_scheme must be 'zscore', 'equal', or 'hybrid'.")

                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                last_weights = new_weights

        weights.loc[date] = last_weights

    return weights