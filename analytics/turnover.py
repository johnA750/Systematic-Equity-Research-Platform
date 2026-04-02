def compute_turnover_metrics(turnover):
    """
    Compute turnover statistics from a daily turnover series.
    """

    average_daily_turnover = turnover.mean()
    annualized_turnover = average_daily_turnover * 252
    total_turnover = turnover.sum()

    return {
        "Average Daily Turnover": average_daily_turnover,
        "Annualized Turnover": annualized_turnover,
        "Total Turnover": total_turnover
    }