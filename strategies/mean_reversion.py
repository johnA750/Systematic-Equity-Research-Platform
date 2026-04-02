import pandas as pd


def generate_mean_reversion_weights(prices, lookback=5, top_n=10, rebalance_freq=5):
    """
    Short-term reversal strategy (stronger mean reversion)

    Rules:
    - compute short-term returns (e.g. 5-day)
    - pick worst performers (most negative returns)
    - assume bounce (mean reversion)
    - rebalance frequently
    """

    returns = prices.pct_change(lookback)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i % rebalance_freq == 0:
            daily_returns = returns.loc[date].dropna()

            # pick worst performers
            losers = daily_returns.nsmallest(min(top_n, len(daily_returns)))

            # convert to positive scores
            scores = -losers

            new_weights = pd.Series(0.0, index=prices.columns)
            new_weights.loc[scores.index] = scores.values

            if new_weights.sum() > 0:
                new_weights = new_weights / new_weights.sum()

            last_weights = new_weights

        weights.loc[date] = last_weights

    return weights