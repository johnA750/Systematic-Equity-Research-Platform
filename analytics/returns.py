import pandas as pd


def compute_daily_returns(prices):
    """
    Compute daily percentage returns from price data.
    """
    returns = prices.pct_change(fill_method=None)
    returns = returns.dropna(how="all")
    return returns
