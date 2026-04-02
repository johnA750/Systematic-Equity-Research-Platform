import pandas as pd


def generate_momentum_weights(prices, lookback=20, top_n=3, rebalance_freq=5):
    """
    Generate long-only momentum portfolio weights.

    Rules:
    - compute lookback momentum
    - only consider stocks with positive momentum
    - select up to top_n positive-momentum stocks
    - weight proportional to momentum strength
    - rebalance every rebalance_freq days
    """

    momentum = prices / prices.shift(lookback) - 1
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(momentum.index):
        daily = momentum.loc[date].dropna()

        if i % rebalance_freq == 0:
            positive = daily[daily > 0]

            if len(positive) == 0:
                last_weights = pd.Series(0.0, index=prices.columns)
            else:
                top = positive.nlargest(min(top_n, len(positive)))

                new_weights = pd.Series(0.0, index=prices.columns)
                new_weights.loc[top.index] = top.values

                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                last_weights = new_weights

        weights.loc[date] = last_weights

    return weights