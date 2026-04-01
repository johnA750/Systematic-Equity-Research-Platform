import pandas as pd


def generate_momentum_signals(prices, lookback=20):
    """
    Generate long-only momentum signals.

    Signal = 1 if price is above its price N days ago
    Signal = 0 otherwise
    """
    momentum = prices / prices.shift(lookback) - 1
    signals = (momentum > 0).astype(int)
    return signals