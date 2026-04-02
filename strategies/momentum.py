import pandas as pd


def generate_momentum_weights(prices, lookback=20, top_n=10, rebalance_freq=20, vol_lookback=20):
    """
    Generate long-only volatility-adjusted momentum portfolio weights.

    Rules:
    - compute lookback momentum
    - compute rolling volatility from daily returns
    - score stocks by momentum / volatility
    - only consider stocks with positive momentum
    - select up to top_n stocks by score
    - weight proportional to score
    - rebalance every rebalance_freq days
    """

    momentum = prices / prices.shift(lookback) - 1
    daily_returns = prices.pct_change()
    volatility = daily_returns.rolling(vol_lookback).std()

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i % rebalance_freq == 0:
            daily_momentum = momentum.loc[date].dropna()
            daily_vol = volatility.loc[date].dropna()

            common = daily_momentum.index.intersection(daily_vol.index)
            daily_momentum = daily_momentum.loc[common]
            daily_vol = daily_vol.loc[common]

            # Avoid divide-by-zero / bad values
            valid = daily_vol > 0
            daily_momentum = daily_momentum.loc[valid]
            daily_vol = daily_vol.loc[valid]

            # Long-only: only keep positive-momentum names
            positive = daily_momentum[daily_momentum > 0]

            if len(positive) == 0:
                last_weights = pd.Series(0.0, index=prices.columns)
            else:
                positive_vol = daily_vol.loc[positive.index]
                score = positive / positive_vol

                top = score.nlargest(min(top_n, len(score)))

                new_weights = pd.Series(0.0, index=prices.columns)
                new_weights.loc[top.index] = top.values

                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                last_weights = new_weights

        weights.loc[date] = last_weights

    return weights