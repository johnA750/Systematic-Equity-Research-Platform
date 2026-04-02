import numpy as np


def compute_performance_metrics(returns, risk_free_rate=0.0):
    """
    Compute key performance metrics from daily returns.

    Parameters:
        returns (pd.Series): daily strategy returns
        risk_free_rate (float): annual risk-free rate, default 0.0

    Returns:
        dict: performance metrics
    """

    # Total return
    total_return = (1 + returns).prod() - 1

    # Annualized return
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)

    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    # Excess daily returns
    excess_returns = returns - daily_rf

    # Sharpe ratio
    sharpe_ratio = (
        excess_returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() != 0 else 0
    )

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