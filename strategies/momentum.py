import pandas as pd


def generate_momentum_signals(prices, lookback=20, top_n=2):
    """
    Generate ranking-based momentum signals.

    For each day:
    - compute lookback return for each stock
    - rank stocks by momentum
    - assign signal = 1 to top_n stocks
    - assign signal = 0 to all others
    """
    momentum = prices / prices.shift(lookback) - 1

    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    for date in momentum.index:
        daily_momentum = momentum.loc[date].dropna()

        if len(daily_momentum) < top_n:
            continue

        top_stocks = daily_momentum.nlargest(top_n).index
        signals.loc[date, top_stocks] = 1

    return signals