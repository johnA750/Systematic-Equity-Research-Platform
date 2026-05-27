import pandas as pd

from strategies.position_sizing import normalize_with_position_cap


def generate_momentum_weights(
    prices,
    lookback=20,
    top_n=10,
    rebalance_freq=20,
    vol_lookback=20,
    max_position_weight=None,
    eligibility_mask=None,
):
    """
    Generate long-only volatility-adjusted momentum portfolio weights.

    Rules:
    - compute lookback momentum
    - compute rolling volatility from daily returns
    - score stocks by momentum / volatility
    - only consider stocks with positive momentum
    - select up to top_n stocks by score
    - weight proportional to score
    - optionally cap final selected-name weights
    - optionally exclude names failing an external tradability screen
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

            if eligibility_mask is not None:
                eligible_names = eligibility_mask.reindex(
                    index=[date],
                    columns=prices.columns,
                    fill_value=False,
                ).iloc[0]
                liquid_names = eligible_names[eligible_names].index
                daily_momentum = daily_momentum.loc[
                    daily_momentum.index.intersection(liquid_names)
                ]
                daily_vol = daily_vol.loc[
                    daily_vol.index.intersection(liquid_names)
                ]

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

                new_weights = normalize_with_position_cap(
                    new_weights,
                    max_position_weight=max_position_weight,
                )

                last_weights = new_weights

        weights.loc[date] = last_weights

    return weights
