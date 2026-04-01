import pandas as pd


def generate_mean_reversion_signals(prices, lookback=5, top_n=3):
    """
    Generate ranking-based mean reversion signals.

    For each day:
    - compute short-term return for each stock
    - rank stocks by short-term return
    - assign signal = 1 to the worst top_n performers
    - assign signal = 0 to all others
    """
    short_term_returns = prices / prices.shift(lookback) - 1

    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    for date in short_term_returns.index:
        daily_returns = short_term_returns.loc[date].dropna()

        if len(daily_returns) < top_n:
            continue

        worst_stocks = daily_returns.nsmallest(top_n).index
        signals.loc[date, worst_stocks] = 1

    return signals