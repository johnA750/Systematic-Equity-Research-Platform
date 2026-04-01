import pandas as pd


def generate_momentum_signals(prices, lookback=20, top_n=3, buffer_n=5):
    """
    Momentum with buffer to reduce turnover.

    - Enter top_n stocks
    - Hold stocks as long as they remain in top buffer_n
    """

    momentum = prices / prices.shift(lookback) - 1
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    prev_positions = set()

    for date in momentum.index:
        daily_momentum = momentum.loc[date].dropna()

        if len(daily_momentum) < buffer_n:
            continue

        ranked = daily_momentum.sort_values(ascending=False)

        top = set(ranked.head(top_n).index)
        buffer = set(ranked.head(buffer_n).index)

        new_positions = set()

        # Always include top_n
        new_positions |= top

        # Keep previous positions if still in buffer
        for stock in prev_positions:
            if stock in buffer:
                new_positions.add(stock)

        signals.loc[date, list(new_positions)] = 1
        prev_positions = new_positions

    return signals