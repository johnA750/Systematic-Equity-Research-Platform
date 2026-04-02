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


def annualized_turnover_penalty(
    turnover: pd.Series,
    penalty_strength: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if turnover is None or penalty_strength == 0:
        return 0.0

    turnover = turnover.dropna()
    if len(turnover) == 0:
        return 0.0

    annual_turnover = turnover.mean() * periods_per_year
    return penalty_strength * annual_turnover


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    total = np.sum(w)
    if total == 0:
        return np.ones_like(w) / len(w)
    return w / total


def compute_inverse_vol_weights(strategy_returns: pd.DataFrame) -> pd.Series:
    strategy_returns = strategy_returns.dropna(how="any").copy()

    if strategy_returns.shape[1] == 0:
        raise ValueError("strategy_returns has no usable columns.")

    vols = strategy_returns.std()
    inv_vol = 1.0 / vols.replace(0, np.nan)
    inv_vol = inv_vol.fillna(0.0)

    weight_sum = inv_vol.sum()
    if weight_sum == 0:
        weights = pd.Series(
            np.ones(len(inv_vol)) / len(inv_vol),
            index=inv_vol.index,
            name="inverse_vol_weight",
        )
    else:
        weights = (inv_vol / weight_sum).rename("inverse_vol_weight")

    return weights


def optimize_strategy_mix(
    strategy_returns: pd.DataFrame,
    bounds: list[tuple[float, float]] | None = None,
    penalty_strength: float = 0.0,
    strategy_turnovers: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    regularization_strength: float = 0.0,
    regularization_target: str = "equal",
) -> dict:
    strategy_returns = strategy_returns.dropna(how="any").copy()
    n = strategy_returns.shape[1]

    if n == 0:
        raise ValueError("strategy_returns has no usable columns after dropping NaNs.")

    if bounds is None:
        bounds = [(0.0, 1.0)] * n

    if len(bounds) != n:
        raise ValueError("Length of bounds must match number of strategy columns.")

    if strategy_turnovers is not None:
        strategy_turnovers = strategy_turnovers.reindex(strategy_returns.index).copy()

    x0 = np.ones(n) / n

    if regularization_target == "equal":
        target_weights = np.ones(n) / n
    elif regularization_target == "inverse_vol":
        target_weights = compute_inverse_vol_weights(strategy_returns).values
    else:
        raise ValueError("regularization_target must be 'equal' or 'inverse_vol'.")

    def objective(w: np.ndarray) -> float:
        w = _normalize_weights(w)

        combined_returns = strategy_returns @ w
        sharpe = annualized_sharpe(combined_returns, periods_per_year=periods_per_year)

        if np.isnan(sharpe):
            return 1e6

        turnover_penalty = 0.0
        if strategy_turnovers is not None:
            aligned_turnover = strategy_turnovers.loc[strategy_returns.index]
            portfolio_turnover = aligned_turnover @ np.abs(w)
            turnover_penalty = annualized_turnover_penalty(
                portfolio_turnover,
                penalty_strength=penalty_strength,
                periods_per_year=periods_per_year,
            )

        regularization_penalty = 0.0
        if regularization_strength > 0:
            regularization_penalty = regularization_strength * np.sum((w - target_weights) ** 2)

        score = sharpe - turnover_penalty - regularization_penalty
        return -score

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimized_weights = _normalize_weights(result.x)
    optimized_returns = (strategy_returns @ optimized_weights).rename("optimized_combined")
    optimized_sharpe = annualized_sharpe(optimized_returns, periods_per_year=periods_per_year)

    optimized_portfolio_turnover = None
    turnover_penalty_value = 0.0
    if strategy_turnovers is not None:
        optimized_portfolio_turnover = (
            strategy_turnovers.loc[strategy_returns.index] @ np.abs(optimized_weights)
        ).rename("optimized_turnover")

        turnover_penalty_value = annualized_turnover_penalty(
            optimized_portfolio_turnover,
            penalty_strength=penalty_strength,
            periods_per_year=periods_per_year,
        )

    regularization_penalty_value = 0.0
    if regularization_strength > 0:
        regularization_penalty_value = regularization_strength * np.sum(
            (optimized_weights - target_weights) ** 2
        )

    return {
        "success": result.success,
        "message": result.message,
        "weights": pd.Series(
            optimized_weights,
            index=strategy_returns.columns,
            name="optimized_weight",
        ),
        "optimized_returns": optimized_returns,
        "optimized_sharpe": optimized_sharpe,
        "optimized_turnover": optimized_portfolio_turnover,
        "turnover_penalty_value": turnover_penalty_value,
        "regularization_penalty_value": regularization_penalty_value,
        "target_weights": pd.Series(
            target_weights,
            index=strategy_returns.columns,
            name="target_weight",
        ),
        "objective_value": result.fun,
    }


def rolling_optimize_strategy_mix(
    strategy_returns: pd.DataFrame,
    strategy_turnovers: pd.DataFrame | None = None,
    lookback_window: int = 252,
    rebalance_frequency: int = 21,
    bounds: list[tuple[float, float]] | None = None,
    penalty_strength: float = 0.0,
    periods_per_year: int = 252,
    regularization_strength: float = 0.0,
    regularization_target: str = "inverse_vol",
) -> dict:
    """
    Rolling walk-forward optimization.

    For each rebalance date:
    - optimize using trailing lookback_window data
    - hold those weights for the next rebalance_frequency days

    Returns
    -------
    dict with:
    - rolling_weights
    - rolling_returns
    - rolling_turnover
    """
    strategy_returns = strategy_returns.dropna(how="any").copy()

    if strategy_returns.empty:
        raise ValueError("strategy_returns is empty after dropping NaNs.")

    if strategy_turnovers is not None:
        strategy_turnovers = strategy_turnovers.reindex(strategy_returns.index).copy()

    n = strategy_returns.shape[1]

    if bounds is None:
        bounds = [(0.0, 1.0)] * n

    if len(bounds) != n:
        raise ValueError("Length of bounds must match number of strategy columns.")

    dates = strategy_returns.index
    rolling_weights = pd.DataFrame(index=dates, columns=strategy_returns.columns, dtype=float)
    rolling_returns = pd.Series(index=dates, dtype=float, name="rolling_optimized")
    rolling_turnover = pd.Series(index=dates, dtype=float, name="rolling_portfolio_turnover")

    for start_idx in range(lookback_window, len(dates), rebalance_frequency):
        train_start_idx = start_idx - lookback_window
        train_end_idx = start_idx
        hold_end_idx = min(start_idx + rebalance_frequency, len(dates))

        train_returns = strategy_returns.iloc[train_start_idx:train_end_idx]

        if strategy_turnovers is not None:
            train_turnovers = strategy_turnovers.iloc[train_start_idx:train_end_idx]
        else:
            train_turnovers = None

        opt_result = optimize_strategy_mix(
            strategy_returns=train_returns,
            bounds=bounds,
            penalty_strength=penalty_strength,
            strategy_turnovers=train_turnovers,
            periods_per_year=periods_per_year,
            regularization_strength=regularization_strength,
            regularization_target=regularization_target,
        )

        w = opt_result["weights"]

        hold_returns = strategy_returns.iloc[start_idx:hold_end_idx]
        rolling_weights.iloc[start_idx:hold_end_idx] = w.values
        rolling_returns.iloc[start_idx:hold_end_idx] = hold_returns @ w.values

        if strategy_turnovers is not None:
            hold_turnovers = strategy_turnovers.iloc[start_idx:hold_end_idx]
            rolling_turnover.iloc[start_idx:hold_end_idx] = hold_turnovers @ np.abs(w.values)

    rolling_weights = rolling_weights.dropna(how="all")
    rolling_returns = rolling_returns.dropna()

    if strategy_turnovers is not None:
        rolling_turnover = rolling_turnover.dropna()
    else:
        rolling_turnover = None

    return {
        "rolling_weights": rolling_weights,
        "rolling_returns": rolling_returns,
        "rolling_turnover": rolling_turnover,
    }