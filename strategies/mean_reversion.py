import pandas as pd
import numpy as np

from strategies.position_sizing import normalize_with_position_cap


def generate_mean_reversion_weights(
    prices,
    lookback=2,
    top_n=10,
    rebalance_freq=5,
    trend_window=100,
    min_pullback=-0.06,
    max_pullback=-0.01,
    weighting_scheme="equal",
    max_position_weight=None,
    eligibility_mask=None,
):
    """
    Long-only pullback strategy:
    - find names in long-term uptrends
    - buy modest short-term pullbacks
    - avoid extreme crash names
    - optionally cap final selected-name weights
    - optionally exclude names failing an external tradability screen
    """

    prices = prices.copy()

    short_returns = prices.pct_change(lookback)
    long_trend = prices / prices.rolling(trend_window).mean() - 1

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i % rebalance_freq == 0:
            sr = short_returns.loc[date]
            trend = long_trend.loc[date]

            eligible = sr.notna() & trend.notna()

            if eligibility_mask is not None:
                liquidity_eligible = eligibility_mask.reindex(
                    index=[date],
                    columns=prices.columns,
                    fill_value=False,
                ).iloc[0]
                eligible &= liquidity_eligible

            # must already be in a healthy uptrend
            eligible &= trend > 0.03

            # only modest pullbacks, not huge collapses
            eligible &= sr >= min_pullback
            eligible &= sr <= max_pullback

            candidates = sr[eligible].dropna()

            if len(candidates) == 0:
                last_weights = pd.Series(0.0, index=prices.columns)
            else:
                selected = candidates.nsmallest(min(top_n, len(candidates)))
                new_weights = pd.Series(0.0, index=prices.columns)

                if weighting_scheme == "equal":
                    new_weights.loc[selected.index] = 1.0
                else:
                    scores = -selected
                    scores = scores.clip(lower=0)
                    new_weights.loc[scores.index] = scores.values

                new_weights = normalize_with_position_cap(
                    new_weights,
                    max_position_weight=max_position_weight,
                )
                last_weights = new_weights

        weights.loc[date] = last_weights

    return weights
