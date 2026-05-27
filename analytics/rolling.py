import numpy as np
import pandas as pd

from analytics.factor_regression import run_capm_regression


def compute_rolling_performance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
    periods_per_year: int = 252,
    use_hac: bool = True,
    hac_lags: int = 5,
) -> pd.DataFrame:
    """
    Compute trailing volatility, Sharpe, beta, and annualized CAPM alpha.
    """
    if window < 2:
        raise ValueError("window must be at least 2.")

    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    output = pd.DataFrame(index=aligned.index)
    output["Rolling Volatility"] = (
        aligned["strategy"].rolling(window).std() * np.sqrt(periods_per_year)
    )
    rolling_std = aligned["strategy"].rolling(window).std()
    output["Rolling Sharpe"] = (
        aligned["strategy"].rolling(window).mean() / rolling_std * np.sqrt(periods_per_year)
    )

    beta = pd.Series(index=aligned.index, dtype=float)
    alpha = pd.Series(index=aligned.index, dtype=float)
    for end in range(window - 1, len(aligned)):
        sample = aligned.iloc[end - window + 1:end + 1]
        regression = run_capm_regression(
            sample["strategy"],
            sample["benchmark"],
            periods_per_year=periods_per_year,
            use_hac=use_hac,
            hac_lags=hac_lags,
        )
        date = sample.index[-1]
        beta.loc[date] = regression["beta"]
        alpha.loc[date] = regression["alpha_annualized"]

    output["Rolling Beta"] = beta
    output["Rolling Alpha Annualized"] = alpha
    return output
