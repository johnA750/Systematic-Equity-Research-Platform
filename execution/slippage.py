import pandas as pd


VALID_SLIPPAGE_MODELS = {"fixed", "volatility_adjusted", "turnover_adjusted"}


def compute_slippage_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    slippage_bps: float = 0.0,
    model: str = "fixed",
    volatility_window: int = 20,
    volatility_reference: float = 0.02,
    turnover_multiplier: float = 1.0,
) -> pd.Series:
    """
    Compute execution slippage costs under selectable models.

    The fixed model reproduces the prior behavior. Advanced models scale the
    same base bps charge using information observable at the previous close.
    """
    if model not in VALID_SLIPPAGE_MODELS:
        raise ValueError(f"Unknown slippage model: {model}.")

    aligned_turnover = turnover.reindex(gross_returns.index).fillna(0.0)
    base_rate = slippage_bps / 10000.0
    multiplier = pd.Series(1.0, index=gross_returns.index)

    if model == "volatility_adjusted":
        if volatility_window < 1 or volatility_reference <= 0:
            raise ValueError("Volatility slippage settings must be positive.")
        trailing_volatility = gross_returns.rolling(volatility_window).std().shift(1)
        multiplier = (trailing_volatility / volatility_reference).clip(lower=1.0).fillna(1.0)
    elif model == "turnover_adjusted":
        if turnover_multiplier < 0:
            raise ValueError("turnover_multiplier must be non-negative.")
        trailing_turnover = aligned_turnover.shift(1).fillna(0.0)
        multiplier = 1.0 + turnover_multiplier * trailing_turnover

    costs = aligned_turnover * base_rate * multiplier
    costs.name = "slippage_cost"
    return costs


def apply_execution_slippage(
    gross_returns: pd.Series,
    turnover: pd.Series,
    slippage_bps: float = 0.0,
    model: str = "fixed",
    volatility_window: int = 20,
    volatility_reference: float = 0.02,
    turnover_multiplier: float = 1.0,
) -> pd.Series:
    """Apply the selected execution slippage model to portfolio returns."""
    costs = compute_slippage_costs(
        gross_returns,
        turnover,
        slippage_bps=slippage_bps,
        model=model,
        volatility_window=volatility_window,
        volatility_reference=volatility_reference,
        turnover_multiplier=turnover_multiplier,
    )
    net_returns = gross_returns.copy() - costs
    net_returns.name = getattr(gross_returns, "name", "returns")
    return net_returns
