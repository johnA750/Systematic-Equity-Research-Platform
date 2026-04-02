import pandas as pd
import numpy as np  

def generate_mean_reversion_weights(prices, lookback=5, top_n=10, rebalance_freq=5):
    """
    Mean reversion WITH trend filter
    """

    short_returns = prices.pct_change(lookback)
    long_trend = prices / prices.rolling(50).mean() - 1  # trend filter

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i % rebalance_freq == 0:
            r = short_returns.loc[date]
            trend = long_trend.loc[date]

            valid = (trend > 0)  # ONLY uptrend stocks

            r = r[valid].dropna()

            losers = r.nsmallest(min(top_n, len(r)))

            scores = -losers

            new_weights = pd.Series(0.0, index=prices.columns)
            new_weights.loc[scores.index] = scores.values

            if new_weights.sum() > 0:
                new_weights /= new_weights.sum()

            last_weights = new_weights

        weights.loc[date] = last_weights

    return weights