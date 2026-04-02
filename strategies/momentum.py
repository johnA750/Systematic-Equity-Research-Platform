import pandas as pd


def generate_momentum_weights(prices, lookback=20, top_n=3):
    """
    Generate momentum-based portfolio weights.

    For each day:
    - compute lookback return for each stock
    - select the top_n stocks by momentum
    - assign weights proportional to momentum strength
    - normalize selected weights to sum to 1

    Parameters:
        prices (pd.DataFrame): price data
        lookback (int): momentum lookback window
        top_n (int): number of stocks to hold

    Returns:
        pd.DataFrame: portfolio weights by date and ticker
    """
    momentum = prices / prices.shift(lookback) - 1

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for date in momentum.index:
        daily_momentum = momentum.loc[date].dropna()

        if len(daily_momentum) < top_n:
            continue

        top_stocks = daily_momentum.nlargest(top_n)

        if top_stocks.sum() <= 0:
            continue

        normalized_weights = top_stocks / top_stocks.sum()
        weights.loc[date, normalized_weights.index] = normalized_weights.values

    return weights