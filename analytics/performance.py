import pandas as pd


def compute_cumulative_returns(returns):
    """
    Convert daily returns into cumulative return curve.
    """
    cumulative = (1 + returns).cumprod()
    return cumulative