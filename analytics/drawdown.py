def compute_drawdown_series(cumulative_returns):
    """
    Compute drawdown series from cumulative returns.
    """
    running_peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_peak) / running_peak
    return drawdown