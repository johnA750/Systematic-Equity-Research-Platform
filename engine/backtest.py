import pandas as pd


def run_backtest(signals, returns, transaction_cost=0.0):
    """
    Run a simple long-only equal-weight backtest with optional transaction costs.

    Parameters:
        signals (pd.DataFrame): 0/1 position signals
        returns (pd.DataFrame): daily asset returns
        transaction_cost (float): cost per unit turnover, e.g. 0.001 = 10 bps

    Returns:
        pd.Series: daily portfolio returns after transaction costs
    """
    shifted_signals = signals.shift(1).fillna(0)

    strategy_asset_returns = shifted_signals * returns
    active_positions = shifted_signals.sum(axis=1)

    portfolio_returns = strategy_asset_returns.sum(axis=1) / active_positions
    portfolio_returns = portfolio_returns.fillna(0)

    turnover = shifted_signals.diff().abs().sum(axis=1)
    turnover = turnover.fillna(0)

    costs = turnover * transaction_cost
    portfolio_returns_after_costs = portfolio_returns - costs

    return portfolio_returns_after_costs