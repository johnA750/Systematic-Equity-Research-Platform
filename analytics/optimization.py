import numpy as np
import pandas as pd
from scipy.optimize import minimize


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan

    vol = returns.std()
    if vol == 0 or np.isnan(vol):
        return np.nan

    return np.sqrt(periods_per_year) * returns.mean() / vol


def annualized_turnover_penalty(turnover: pd.Series, penalty_strength: float = 0.0) -> float:
    if turnover is None or penalty_strength == 0:
        return 0.0

    turnover = turnover.dropna()
    if len(turnover) == 0:
        return 0.0

    annual_turnover = turnover.mean() * 252
    return penalty_strength * annual_turnover


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    total = np.sum(w)
    if total == 0:
        return np.ones_like(w) / len(w)
    return w / total


def optimize_strategy_mix(
    strategy_returns: pd.DataFrame,
    bounds: list[tuple[float, float]] | None = None,
    penalty_strength: float = 0.0,
    strategy_turnovers: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> dict:
    """
    Optimize static portfolio weights across strategy sleeves by maximizing
    Sharpe ratio minus optional turnover penalty.
    """
    strategy_returns = strategy_returns.dropna(how="any").copy()
    n = strategy_returns.shape[1]

    if n == 0:
        raise ValueError("strategy_returns has no usable columns after dropping NaNs.")

    if bounds is None:
        bounds = [(0.0, 1.0)] * n

    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        w = _normalize_weights(w)
        combined_returns = strategy_returns @ w

        penalty = 0.0
        if strategy_turnovers is not None:
            aligned_turnover = strategy_turnovers.loc[strategy_returns.index]
            portfolio_turnover = aligned_turnover @ np.abs(w)
            penalty = annualized_turnover_penalty(
                portfolio_turnover,
                penalty_strength=penalty_strength,
            )

        sharpe = annualized_sharpe(combined_returns, periods_per_year=periods_per_year)

        if np.isnan(sharpe):
            return 1e6

        return -(sharpe - penalty)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    w_opt = _normalize_weights(result.x)
    optimized_returns = (strategy_returns @ w_opt).rename("optimized_combined")
    optimized_sharpe = annualized_sharpe(optimized_returns, periods_per_year=periods_per_year)

    return {
        "success": result.success,
        "message": result.message,
        "weights": pd.Series(w_opt, index=strategy_returns.columns, name="optimized_weight"),
        "optimized_returns": optimized_returns,
        "optimized_sharpe": optimized_sharpe,
        "objective_value": result.fun,
    }