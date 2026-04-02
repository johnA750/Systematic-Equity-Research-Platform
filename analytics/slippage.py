import pandas as pd


def apply_slippage_to_returns(
    gross_returns: pd.Series,
    turnover: pd.Series,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """
    Apply slippage to strategy returns using turnover.

    Daily slippage cost is approximated as:
        turnover * slippage_rate

    where:
        slippage_rate = slippage_bps / 10000

    Parameters
    ----------
    gross_returns : pd.Series
        Strategy returns before slippage adjustment.
    turnover : pd.Series
        Daily turnover series.
    slippage_bps : float
        Slippage in basis points per unit turnover.

    Returns
    -------
    pd.Series
        Net returns after slippage adjustment.
    """
    gross_returns = gross_returns.copy()
    turnover = turnover.reindex(gross_returns.index).fillna(0.0)

    slippage_rate = slippage_bps / 10000.0
    slippage_cost = turnover * slippage_rate

    net_returns = gross_returns - slippage_cost
    net_returns.name = getattr(gross_returns, "name", "returns")

    return net_returns


def compute_slippage_cost_series(
    turnover: pd.Series,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """
    Return a slippage cost series for reporting/inspection.
    """
    turnover = turnover.copy().fillna(0.0)
    slippage_rate = slippage_bps / 10000.0
    slippage_cost = turnover * slippage_rate
    slippage_cost.name = "slippage_cost"
    return slippage_cost