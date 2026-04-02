import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_capm_regression(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Run a simple CAPM-style regression of strategy excess returns on benchmark excess returns.

    Returns daily alpha, annualized alpha, beta, R-squared, t-stat, and p-value.
    """
    df = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        raise ValueError("No overlapping data between strategy and benchmark returns.")

    daily_rf = risk_free_rate / periods_per_year

    df["strategy_excess"] = df["strategy"] - daily_rf
    df["benchmark_excess"] = df["benchmark"] - daily_rf

    y = df["strategy_excess"]
    X = sm.add_constant(df["benchmark_excess"])

    model = sm.OLS(y, X).fit()

    alpha_daily = model.params["const"]
    beta = model.params["benchmark_excess"]

    alpha_annualized = alpha_daily * periods_per_year

    return {
        "alpha_daily": alpha_daily,
        "alpha_annualized": alpha_annualized,
        "beta": beta,
        "r_squared": model.rsquared,
        "t_stat_alpha": model.tvalues["const"],
        "p_value_alpha": model.pvalues["const"],
        "t_stat_beta": model.tvalues["benchmark_excess"],
        "p_value_beta": model.pvalues["benchmark_excess"],
        "model_summary": model.summary(),
    }


def summarize_capm_results(results: dict) -> dict:
    """
    Cleaner rounded summary for printing.
    """
    return {
        "Alpha Daily": results["alpha_daily"],
        "Alpha Annualized": results["alpha_annualized"],
        "Beta": results["beta"],
        "R-Squared": results["r_squared"],
        "Alpha t-stat": results["t_stat_alpha"],
        "Alpha p-value": results["p_value_alpha"],
        "Beta t-stat": results["t_stat_beta"],
        "Beta p-value": results["p_value_beta"],
    }