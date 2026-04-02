def run_backtest(weights, returns, transaction_cost=0.0):
    shifted_weights = weights.shift(1).fillna(0)

    portfolio_returns = (shifted_weights * returns).sum(axis=1)

    turnover = shifted_weights.diff().abs().sum(axis=1)
    turnover = turnover.fillna(0)

    costs = turnover * transaction_cost

    return portfolio_returns - costs