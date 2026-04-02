def run_backtest(weights, returns, transaction_cost=0.0, return_turnover=False):
    shifted_weights = weights.shift(1).fillna(0.0)

    portfolio_returns = (shifted_weights * returns).sum(axis=1)

    turnover = shifted_weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * transaction_cost

    portfolio_returns_after_costs = portfolio_returns - costs

    if return_turnover:
        return portfolio_returns_after_costs, turnover

    return portfolio_returns_after_costs