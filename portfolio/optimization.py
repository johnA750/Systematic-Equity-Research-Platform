import numpy as np
import pandas as pd


def optimize_long_only_weights(
    returns: pd.DataFrame,
    max_weight: float = 0.75,
) -> pd.Series:
    """
    Compute minimum-variance long-only weights using SciPy.

    This is an optional research tool; production portfolio construction
    continues to use inverse volatility unless explicitly enabled by callers.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("SciPy is required for constrained optimization.") from exc

    valid_returns = returns.dropna()
    if valid_returns.empty:
        raise ValueError("Returns must contain overlapping observations.")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1].")
    if max_weight * len(valid_returns.columns) < 1:
        raise ValueError("max_weight is too small for a fully invested portfolio.")

    covariance = valid_returns.cov().to_numpy()
    asset_count = len(valid_returns.columns)
    initial = np.repeat(1.0 / asset_count, asset_count)
    objective_scale = 1_000_000.0

    result = minimize(
        lambda weights: float(objective_scale * weights @ covariance @ weights),
        initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * asset_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Constrained optimization failed: {result.message}")

    return pd.Series(result.x, index=valid_returns.columns, name="optimized_weight")
