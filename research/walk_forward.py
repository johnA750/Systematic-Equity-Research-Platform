from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from analytics.factor_regression import run_capm_regression
from analytics.metrics import compute_performance_metrics
from engine.backtest import run_backtest
from execution.slippage import apply_execution_slippage
from portfolio.regime import compute_regime_multiplier
from strategies.mean_reversion import generate_mean_reversion_weights
from strategies.momentum import generate_momentum_weights


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    train_dates: pd.Index
    test_dates: pd.Index

    def as_record(self) -> dict:
        return {
            "window_id": self.window_id,
            "train_start": self.train_dates[0],
            "train_end": self.train_dates[-1],
            "test_start": self.test_dates[0],
            "test_end": self.test_dates[-1],
            "train_observations": len(self.train_dates),
            "test_observations": len(self.test_dates),
        }


def generate_walk_forward_windows(
    dates: pd.Index,
    train_length: int,
    test_length: int,
    step_length: int | None = None,
    mode: str = "rolling",
) -> list[WalkForwardWindow]:
    """Generate rolling or expanding train/test windows using observation counts."""
    if train_length < 1 or test_length < 1:
        raise ValueError("train_length and test_length must be positive.")
    if mode not in {"rolling", "expanding"}:
        raise ValueError("mode must be 'rolling' or 'expanding'.")

    step_length = test_length if step_length is None else step_length
    if step_length < 1:
        raise ValueError("step_length must be positive.")

    windows = []
    train_end = train_length
    window_id = 1
    while train_end + test_length <= len(dates):
        train_start = 0 if mode == "expanding" else train_end - train_length
        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_dates=dates[train_start:train_end],
                test_dates=dates[train_end:train_end + test_length],
            )
        )
        train_end += step_length
        window_id += 1

    return windows


def _inverse_vol_weights(momentum_returns: pd.Series, meanrev_returns: pd.Series) -> tuple[float, float]:
    momentum_vol = momentum_returns.std()
    meanrev_vol = meanrev_returns.std()
    inv_momentum = 1 / momentum_vol if momentum_vol != 0 else 0
    inv_meanrev = 1 / meanrev_vol if meanrev_vol != 0 else 0
    total = inv_momentum + inv_meanrev
    if total == 0:
        return 0.5, 0.5
    return inv_momentum / total, inv_meanrev / total


def _metric_columns(prefix: str, returns: pd.Series) -> dict:
    metrics = compute_performance_metrics(returns)
    return {
        f"{prefix}_total_return": metrics["Total Return"],
        f"{prefix}_annualized_return": metrics["Annualized Return"],
        f"{prefix}_volatility": metrics["Volatility"],
        f"{prefix}_sharpe": metrics["Sharpe Ratio"],
        f"{prefix}_max_drawdown": metrics["Max Drawdown"],
    }


def _capm_columns(
    prefix: str,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    use_hac: bool,
    hac_lags: int,
) -> dict:
    results = run_capm_regression(
        strategy_returns,
        benchmark_returns,
        use_hac=use_hac,
        hac_lags=hac_lags,
    )
    return {
        f"{prefix}_alpha_annualized": results["alpha_annualized"],
        f"{prefix}_beta": results["beta"],
        f"{prefix}_r_squared": results["r_squared"],
        f"{prefix}_alpha_t_stat": results["t_stat_alpha"],
        f"{prefix}_alpha_p_value": results["p_value_alpha"],
        f"{prefix}_covariance": results["covariance_type"],
    }


def evaluate_parameter_pair(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_prices: pd.Series,
    benchmark_returns: pd.Series,
    liquidity_eligibility: pd.DataFrame | None,
    window: WalkForwardWindow,
    parameter_id: str,
    momentum_params: dict,
    meanrev_params: dict,
    transaction_cost: float,
    max_position_weight: float | None,
    slippage_settings: dict,
    capm_use_hac: bool = True,
    capm_hac_lags: int = 5,
) -> tuple[dict, pd.Series]:
    """Evaluate one strategy parameter pair using train-selected sleeve allocation."""
    context_end = window.test_dates[-1]
    context_prices = prices.loc[:context_end]
    context_returns = returns.reindex(context_prices.index)
    context_benchmark_prices = benchmark_prices.reindex(context_prices.index).ffill()
    context_eligibility = (
        liquidity_eligibility.reindex(context_prices.index)
        if liquidity_eligibility is not None else None
    )

    momentum_weights = generate_momentum_weights(
        context_prices,
        max_position_weight=max_position_weight,
        eligibility_mask=context_eligibility,
        **momentum_params,
    )
    regime_multiplier = compute_regime_multiplier(
        context_benchmark_prices,
        ma_window=200,
        sensitivity=2.0,
    )
    momentum_weights = momentum_weights.mul(
        regime_multiplier.reindex(momentum_weights.index).fillna(0.0),
        axis=0,
    )
    momentum_gross, momentum_turnover = run_backtest(
        momentum_weights,
        context_returns,
        transaction_cost=transaction_cost,
        return_turnover=True,
    )
    momentum_net = apply_execution_slippage(
        momentum_gross,
        momentum_turnover,
        **slippage_settings,
    )

    meanrev_weights = generate_mean_reversion_weights(
        context_prices,
        max_position_weight=max_position_weight,
        eligibility_mask=context_eligibility,
        **meanrev_params,
    )
    meanrev_gross, meanrev_turnover = run_backtest(
        meanrev_weights,
        context_returns,
        transaction_cost=transaction_cost,
        return_turnover=True,
    )
    meanrev_net = apply_execution_slippage(
        meanrev_gross,
        meanrev_turnover,
        **slippage_settings,
    )

    train_dates = window.train_dates.intersection(returns.index)
    test_dates = window.test_dates.intersection(returns.index)
    momentum_allocation, meanrev_allocation = _inverse_vol_weights(
        momentum_net.reindex(train_dates).fillna(0.0),
        meanrev_net.reindex(train_dates).fillna(0.0),
    )
    combined = (
        momentum_allocation * momentum_net
        + meanrev_allocation * meanrev_net
    )
    combined.name = "combined_inverse_vol"
    train_returns = combined.reindex(train_dates).fillna(0.0)
    test_returns = combined.reindex(test_dates).fillna(0.0)

    row = window.as_record()
    row.update({"parameter_id": parameter_id})
    row.update({f"momentum_{key}": value for key, value in momentum_params.items()})
    row.update({f"meanrev_{key}": value for key, value in meanrev_params.items()})
    row["selected_momentum_weight"] = momentum_allocation
    row["selected_meanrev_weight"] = meanrev_allocation
    row.update(_metric_columns("train", train_returns))
    row.update(_metric_columns("test", test_returns))
    row["sharpe_degradation"] = row["train_sharpe"] - row["test_sharpe"]
    row.update(
        _capm_columns(
            "train",
            train_returns,
            benchmark_returns.reindex(train_dates),
            use_hac=capm_use_hac,
            hac_lags=capm_hac_lags,
        )
    )
    row.update(
        _capm_columns(
            "test",
            test_returns,
            benchmark_returns.reindex(test_dates),
            use_hac=capm_use_hac,
            hac_lags=capm_hac_lags,
        )
    )
    return row, test_returns


def summarize_parameter_robustness(all_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize parameter performance across walk-forward windows."""
    parameter_columns = [
        column for column in all_results.columns
        if column.startswith("momentum_") or column.startswith("meanrev_")
    ]
    grouped = all_results.groupby(["parameter_id"] + parameter_columns, dropna=False)
    summary = grouped.agg(
        windows_evaluated=("window_id", "count"),
        median_train_sharpe=("train_sharpe", "median"),
        median_test_sharpe=("test_sharpe", "median"),
        mean_test_sharpe=("test_sharpe", "mean"),
        median_test_return=("test_annualized_return", "median"),
        median_sharpe_degradation=("sharpe_degradation", "median"),
        positive_test_fraction=("test_annualized_return", lambda values: (values > 0).mean()),
    ).reset_index()
    return summary.sort_values("median_test_sharpe", ascending=False).reset_index(drop=True)


def compute_selection_stability(selected_results: pd.DataFrame) -> pd.DataFrame:
    """Compute high-level stability diagnostics for train-selected parameters."""
    counts = selected_results["parameter_id"].value_counts()
    correlation = selected_results["train_sharpe"].corr(selected_results["test_sharpe"])
    summary = {
        "Windows Evaluated": len(selected_results),
        "Unique Parameters Selected": selected_results["parameter_id"].nunique(),
        "Most Frequent Parameter": counts.index[0],
        "Parameter Retention Rate": counts.iloc[0] / len(selected_results),
        "Average Train Sharpe": selected_results["train_sharpe"].mean(),
        "Average Test Sharpe": selected_results["test_sharpe"].mean(),
        "Average Sharpe Degradation": selected_results["sharpe_degradation"].mean(),
        "Median Sharpe Degradation": selected_results["sharpe_degradation"].median(),
        "Train Test Sharpe Correlation": correlation,
        "Positive Test Fraction": (selected_results["test_annualized_return"] > 0).mean(),
    }
    return pd.DataFrame([summary])


def summarize_oos_path_comparison(
    selected_results: pd.DataFrame,
    fixed_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare optimized selection against fixed production parameters by window."""
    rows = []
    for path_name, results in (
        ("selected_parameters", selected_results),
        ("fixed_production_parameters", fixed_results),
    ):
        rows.append(
            {
                "path": path_name,
                "windows_evaluated": len(results),
                "average_test_sharpe": results["test_sharpe"].mean(),
                "median_test_sharpe": results["test_sharpe"].median(),
                "positive_test_fraction": (results["test_annualized_return"] > 0).mean(),
                "average_test_drawdown": results["test_max_drawdown"].mean(),
                "average_test_return": results["test_annualized_return"].mean(),
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward_evaluation(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_prices: pd.Series,
    benchmark_returns: pd.Series,
    liquidity_eligibility: pd.DataFrame | None,
    momentum_grid: list[dict],
    meanrev_grid: list[dict],
    train_length: int,
    test_length: int,
    step_length: int,
    mode: str,
    transaction_cost: float,
    max_position_weight: float | None,
    slippage_settings: dict,
    fixed_momentum_params: dict | None = None,
    fixed_meanrev_params: dict | None = None,
    capm_use_hac: bool = True,
    capm_hac_lags: int = 5,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run parameter selection on training windows and stitch selected OOS returns."""
    dates = prices.index.intersection(returns.index)
    windows = generate_walk_forward_windows(
        dates,
        train_length=train_length,
        test_length=test_length,
        step_length=step_length,
        mode=mode,
    )
    if not windows:
        raise ValueError("No walk-forward windows fit within the available data.")

    rows = []
    test_return_map = {}
    for window in windows:
        for momentum_number, meanrev_number in product(
            range(len(momentum_grid)), range(len(meanrev_grid))
        ):
            parameter_id = f"M{momentum_number + 1}_MR{meanrev_number + 1}"
            row, test_returns = evaluate_parameter_pair(
                prices=prices,
                returns=returns,
                benchmark_prices=benchmark_prices,
                benchmark_returns=benchmark_returns,
                liquidity_eligibility=liquidity_eligibility,
                window=window,
                parameter_id=parameter_id,
                momentum_params=momentum_grid[momentum_number],
                meanrev_params=meanrev_grid[meanrev_number],
                transaction_cost=transaction_cost,
                max_position_weight=max_position_weight,
                slippage_settings=slippage_settings,
                capm_use_hac=capm_use_hac,
                capm_hac_lags=capm_hac_lags,
            )
            rows.append(row)
            test_return_map[(window.window_id, parameter_id)] = test_returns

    all_results = pd.DataFrame(rows)
    selected_rows = all_results.loc[
        all_results.groupby("window_id")["train_sharpe"].idxmax()
    ].sort_values("window_id").reset_index(drop=True)
    selected_oos_returns = pd.concat(
        [
            test_return_map[(row.window_id, row.parameter_id)]
            for row in selected_rows.itertuples()
        ]
    )
    selected_oos_returns = selected_oos_returns[
        ~selected_oos_returns.index.duplicated(keep="first")
    ].sort_index()
    selected_oos_returns.name = "selected_out_of_sample_return"

    fixed_momentum_params = fixed_momentum_params or momentum_grid[0]
    fixed_meanrev_params = fixed_meanrev_params or meanrev_grid[0]
    fixed_rows = []
    fixed_returns = []
    for window in windows:
        row, test_returns = evaluate_parameter_pair(
            prices=prices,
            returns=returns,
            benchmark_prices=benchmark_prices,
            benchmark_returns=benchmark_returns,
            liquidity_eligibility=liquidity_eligibility,
            window=window,
            parameter_id="fixed_production",
            momentum_params=fixed_momentum_params,
            meanrev_params=fixed_meanrev_params,
            transaction_cost=transaction_cost,
            max_position_weight=max_position_weight,
            slippage_settings=slippage_settings,
            capm_use_hac=capm_use_hac,
            capm_hac_lags=capm_hac_lags,
        )
        fixed_rows.append(row)
        fixed_returns.append(test_returns)

    fixed_results = pd.DataFrame(fixed_rows).sort_values("window_id").reset_index(drop=True)
    fixed_oos_returns = pd.concat(fixed_returns)
    fixed_oos_returns = fixed_oos_returns[
        ~fixed_oos_returns.index.duplicated(keep="first")
    ].sort_index()
    fixed_oos_returns.name = "fixed_production_out_of_sample_return"

    return {
        "windows": pd.DataFrame([window.as_record() for window in windows]),
        "all_results": all_results.sort_values(["window_id", "train_sharpe"], ascending=[True, False]),
        "selected_results": selected_rows,
        "fixed_parameter_results": fixed_results,
        "parameter_summary": summarize_parameter_robustness(all_results),
        "stability_summary": compute_selection_stability(selected_rows),
        "path_comparison": summarize_oos_path_comparison(selected_rows, fixed_results),
        "oos_returns": selected_oos_returns,
        "fixed_oos_returns": fixed_oos_returns,
    }


def export_walk_forward_results(results: dict, report_directory: str | Path) -> Path:
    """Export structured walk-forward tables to CSV files."""
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    for name, value in results.items():
        if isinstance(value, pd.Series):
            value.to_frame().to_csv(report_path / f"{name}.csv")
        elif isinstance(value, pd.DataFrame):
            value.to_csv(report_path / f"{name}.csv", index=False)
    return report_path
