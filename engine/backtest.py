def run_backtest(signals, returns):
    shifted_signals = signals.shift(1)
    strategy_asset_returns = shifted_signals * returns
    active_positions = shifted_signals.sum(axis=1)
    portfolio_returns = strategy_asset_returns.sum(axis=1) / active_positions
    portfolio_returns = portfolio_returns.fillna(0)
    return portfolio_returns