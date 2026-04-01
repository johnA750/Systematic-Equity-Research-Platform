import pandas as pd


def compute_daily_returns(prices):
    """
    Compute daily percentage returns from price data.
    """
    returns = prices.pct_change()
    returns = returns.dropna()
    return returns