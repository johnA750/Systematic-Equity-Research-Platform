import numpy as np


def compute_performance_metrics(returns):
    """
    Compute key performance metrics from daily returns.
    """

    # Total return
    total_return = (1 + returns).prod() - 1

    # Annualized return (assuming 252 trading days)
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (risk-free rate = 0 for now)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }