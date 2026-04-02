import pandas as pd


def generate_mean_reversion_weights(prices, lookback=20, top_n=10, rebalance_freq=20):
    """
    Generate long-only mean reversion portfolio weights.

    Rules:
    - compute moving average over lookback window
    - compute distance below moving average
    - only consider stocks trading below their moving average
    - select up to top_n most oversold stocks
    - weight proportional to oversold magnitude
    - rebalance every rebalance_freq days
    """

    moving_avg = prices.rolling(lookback).mean()
    deviation = (prices / moving_avg) - 1

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i % rebalance_freq == 0:
            daily_dev = deviation.loc[date].dropna()

            # Only consider stocks below moving average
            oversold = daily_dev[daily_dev < 0]

            if len(oversold) == 0:
                last_weights = pd.Series(0.0, index=prices.columns)
            else:
                # Most negative = most oversold
                top = oversold.nsmallest(min(top_n, len(oversold)))

                # Convert negatives to positive scores
                scores = -top

                new_weights = pd.Series(0.0, index=prices.columns)
                new_weights.loc[scores.index] = scores.values

                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                last_weights = new_weights

        weights.loc[date] = last_weights

    return weights