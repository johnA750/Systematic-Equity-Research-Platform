import pandas as pd


def run_backtest(signals, returns):
    """
    Run a simple long-only equal-weight backtest.

    Parameters:
        signals (pd.DataFrame): 0/1 position signals
        returns (pd.DataFrame): daily asset returns

    Returns:
        pd.Series: daily portfolio returns
    """
    # Use yesterday's signal to earn today's return
    shifted_signals = signals.shift(1)

    # Strategy asset-level returns
    strategy_asset_returns = shifted_signals * returns

    # Count how many positions are active each day
    active_positions = shifted_signals.sum(axis=1)

    # Equal-weight across active signals
    portfolio_returns = strategy_asset_returns.sum(axis=1) / active_positions

    # Replace NaN days (no positions) with 0 return
    portfolio_returns = portfolio_returns.fillna(0)

    return portfolio_returns