import pandas as pd

from execution.slippage import apply_execution_slippage, compute_slippage_costs


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
    return apply_execution_slippage(
        gross_returns,
        turnover,
        slippage_bps=slippage_bps,
        model="fixed",
    )


def compute_slippage_cost_series(
    turnover: pd.Series,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """
    Return a slippage cost series for reporting/inspection.
    """
    zero_returns = pd.Series(0.0, index=turnover.index)
    return compute_slippage_costs(
        zero_returns,
        turnover,
        slippage_bps=slippage_bps,
        model="fixed",
    )
