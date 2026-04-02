import pandas as pd


def generate_mean_reversion_weights(prices, lookback=5, top_n=10, rebalance_freq=5, trend_window=50):
    """
    Long-only mean reversion with trend filter.

    Rules:
    - compute short-term returns
    - only consider stocks in an uptrend
    - only consider stocks with negative short-term returns
    - buy the most oversold of those
    - weight proportional to oversold magnitude
    """

    short_returns = prices.pct_change(lookback)
    long_trend = prices / prices.rolling(trend_window).mean() - 1

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i % rebalance_freq == 0:
            r = short_returns.loc[date]
            trend = long_trend.loc[date]

            # Only mean revert stocks that are in a long-term uptrend
            # AND currently down over the short lookback window
            eligible = (trend > 0) & (r < 0)

            r = r[eligible].dropna()

            if len(r) == 0:
                last_weights = pd.Series(0.0, index=prices.columns)
            else:
                losers = r.nsmallest(min(top_n, len(r)))

                # Convert negative returns into positive sizing scores
                scores = -losers

                new_weights = pd.Series(0.0, index=prices.columns)
                new_weights.loc[scores.index] = scores.values

                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                last_weights = new_weights

        weights.loc[date] = last_weights

    return weights