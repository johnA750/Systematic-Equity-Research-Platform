from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.liquidity import compute_average_dollar_volume


VALID_MARKET_IMPACT_MODELS = {"square_root", "linear"}


@dataclass
class MarketImpactResult:
    applied_costs: pd.Series
    daily_analysis: pd.DataFrame
    trade_details: pd.DataFrame
    capacity_summary: pd.DataFrame


def summarize_capacity(
    daily_analysis: pd.DataFrame,
    trade_details: pd.DataFrame,
    portfolio_name: str,
    target_dollars: float,
    max_adv_participation: float,
    impact_enabled: bool,
    model: str,
) -> pd.DataFrame:
    """Summarize deployable capital and participation diagnostics."""
    valid_trades = trade_details[
        (trade_details["trade_weight"] > 0)
        & trade_details["adv_dollars"].notna()
        & (trade_details["adv_dollars"] > 0)
    ].copy()
    if valid_trades.empty:
        return pd.DataFrame(
            [
                {
                    "portfolio": portfolio_name,
                    "market_impact_enabled": impact_enabled,
                    "market_impact_model": model,
                    "capacity_target_dollars": target_dollars,
                    "estimated_capacity_dollars": np.nan,
                    "binding_capacity_dollars": np.nan,
                    "average_participation_rate": np.nan,
                    "maximum_participation_rate": np.nan,
                    "average_liquidity_utilization": np.nan,
                    "maximum_liquidity_utilization": np.nan,
                    "participation_breach_fraction": np.nan,
                    "modeled_market_impact_cost": 0.0,
                    "applied_market_impact_cost": 0.0,
                    "liquidity_data_available": False,
                }
            ]
        )

    per_trade_capacity = (
        max_adv_participation
        * valid_trades["adv_dollars"]
        / valid_trades["trade_weight"]
    )
    daily = daily_analysis.copy()
    return pd.DataFrame(
        [
            {
                "portfolio": portfolio_name,
                "market_impact_enabled": impact_enabled,
                "market_impact_model": model,
                "capacity_target_dollars": target_dollars,
                "estimated_capacity_dollars": per_trade_capacity.quantile(0.05),
                "binding_capacity_dollars": per_trade_capacity.min(),
                "average_participation_rate": daily["average_participation_rate"].mean(),
                "maximum_participation_rate": daily["maximum_participation_rate"].max(),
                "average_liquidity_utilization": daily["liquidity_utilization"].mean(),
                "maximum_liquidity_utilization": daily["liquidity_utilization"].max(),
                "participation_breach_fraction": (
                    valid_trades["participation_rate"] > max_adv_participation
                ).mean(),
                "modeled_market_impact_cost": daily["modeled_market_impact_cost"].sum(),
                "applied_market_impact_cost": daily["applied_market_impact_cost"].sum(),
                "liquidity_data_available": True,
            }
        ]
    )


def estimate_market_impact(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    portfolio_name: str,
    target_dollars: float,
    impact_enabled: bool = False,
    model: str = "square_root",
    max_adv_participation: float = 0.10,
    impact_coefficient: float = 0.10,
    adv_lookback_window: int = 20,
    volatility_window: int = 20,
) -> MarketImpactResult:
    """
    Estimate portfolio market impact from observable trade size and liquidity.

    A trade's participation rate is target-capital trade dollars divided by
    lagged ADV. Volatility is also lagged so the model does not use same-day
    information to price the rebalance.
    """
    if model not in VALID_MARKET_IMPACT_MODELS:
        raise ValueError(f"Unknown market impact model: {model}.")
    if target_dollars <= 0 or max_adv_participation <= 0 or impact_coefficient < 0:
        raise ValueError("Market impact capital, participation, and coefficient settings must be valid.")
    if volatility_window < 1:
        raise ValueError("volatility_window must be at least 1.")

    aligned_weights = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    trade_weights = aligned_weights.diff().abs().fillna(0.0)
    turnover = trade_weights.sum(axis=1).rename("turnover")
    realized_volatility = prices.pct_change(fill_method=None).rolling(volatility_window).std().shift(1)
    if volumes is None or volumes.empty or volumes.notna().sum().sum() == 0:
        adv = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    else:
        adv = compute_average_dollar_volume(
            prices,
            volumes.reindex(index=prices.index, columns=prices.columns),
            lookback_window=adv_lookback_window,
        )

    trade_dollars = trade_weights * target_dollars
    participation = trade_dollars.div(adv.where(adv > 0))
    if model == "square_root":
        impact_rates = impact_coefficient * realized_volatility * np.sqrt(participation)
    else:
        impact_rates = impact_coefficient * realized_volatility * participation
    impact_rates = impact_rates.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    modeled_costs = (trade_weights * impact_rates).sum(axis=1).rename("modeled_market_impact_cost")
    applied_costs = (
        modeled_costs.copy()
        if impact_enabled
        else pd.Series(0.0, index=modeled_costs.index, name="applied_market_impact_cost")
    )
    applied_costs.name = "applied_market_impact_cost"

    active_trade_weight = trade_weights.where(trade_weights > 0)
    weighted_participation = (
        (participation * trade_weights).sum(axis=1)
        .div(turnover.where(turnover > 0))
        .rename("average_participation_rate")
    )
    maximum_participation = participation.where(active_trade_weight.notna()).max(axis=1)
    daily_analysis = pd.DataFrame(
        {
            "portfolio": portfolio_name,
            "turnover": turnover,
            "trade_dollars": trade_dollars.sum(axis=1),
            "average_participation_rate": weighted_participation,
            "maximum_participation_rate": maximum_participation,
            "liquidity_utilization": maximum_participation / max_adv_participation,
            "modeled_market_impact_cost": modeled_costs,
            "applied_market_impact_cost": applied_costs,
        }
    )
    daily_analysis.index.name = "date"

    trade_details = pd.concat(
        {
            "trade_weight": trade_weights.stack(),
            "trade_dollars": trade_dollars.stack(),
            "adv_dollars": adv.stack(),
            "realized_volatility": realized_volatility.stack(),
            "participation_rate": participation.stack(),
            "impact_rate": impact_rates.stack(),
        },
        axis=1,
    ).reset_index(names=["date", "ticker"])
    trade_details["portfolio"] = portfolio_name
    trade_details = trade_details[trade_details["trade_weight"] > 0].reset_index(drop=True)
    trade_details["modeled_market_impact_cost"] = (
        trade_details["trade_weight"] * trade_details["impact_rate"]
    )

    capacity_summary = summarize_capacity(
        daily_analysis,
        trade_details,
        portfolio_name=portfolio_name,
        target_dollars=target_dollars,
        max_adv_participation=max_adv_participation,
        impact_enabled=impact_enabled,
        model=model,
    )
    return MarketImpactResult(applied_costs, daily_analysis, trade_details, capacity_summary)


def _performance_attribution(returns: pd.Series) -> tuple[float, float]:
    """Return annualized return and Sharpe for capital-sensitivity diagnostics."""
    valid = returns.dropna()
    if valid.empty:
        return np.nan, np.nan
    total_return = (1.0 + valid).prod() - 1.0
    annualized_return = (1.0 + total_return) ** (252.0 / len(valid)) - 1.0
    daily_volatility = valid.std()
    sharpe = (
        valid.mean() / daily_volatility * np.sqrt(252)
        if not np.isclose(daily_volatility, 0.0)
        else 0.0
    )
    return annualized_return, sharpe


def compute_capacity_sensitivity(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    slippage_only_returns: pd.Series,
    portfolio_name: str,
    target_dollars: float,
    model: str = "square_root",
    max_adv_participation: float = 0.10,
    impact_coefficient: float = 0.10,
    adv_lookback_window: int = 20,
    volatility_window: int = 20,
    capital_multipliers: tuple[float, ...] = (0.25, 0.50, 1.0, 2.0, 5.0),
) -> pd.DataFrame:
    """Evaluate execution degradation as modeled portfolio capital scales."""
    base_annualized_return, base_sharpe = _performance_attribution(slippage_only_returns)
    rows = []
    for multiplier in capital_multipliers:
        capital = target_dollars * multiplier
        impact = estimate_market_impact(
            weights,
            prices,
            volumes,
            portfolio_name=portfolio_name,
            target_dollars=capital,
            impact_enabled=True,
            model=model,
            max_adv_participation=max_adv_participation,
            impact_coefficient=impact_coefficient,
            adv_lookback_window=adv_lookback_window,
            volatility_window=volatility_window,
        )
        impact_returns = slippage_only_returns - impact.applied_costs.reindex(
            slippage_only_returns.index
        ).fillna(0.0)
        annualized_return, sharpe = _performance_attribution(impact_returns)
        summary = impact.capacity_summary.iloc[0]
        rows.append(
            {
                "portfolio": portfolio_name,
                "capital_dollars": capital,
                "capital_multiple": multiplier,
                "annualized_return_slippage_only": base_annualized_return,
                "annualized_return_impact_enabled": annualized_return,
                "return_degradation": base_annualized_return - annualized_return,
                "sharpe_slippage_only": base_sharpe,
                "sharpe_impact_enabled": sharpe,
                "sharpe_degradation": base_sharpe - sharpe,
                "modeled_market_impact_cost": summary["modeled_market_impact_cost"],
                "average_participation_rate": summary["average_participation_rate"],
                "maximum_participation_rate": summary["maximum_participation_rate"],
                "average_liquidity_utilization": summary["average_liquidity_utilization"],
                "maximum_liquidity_utilization": summary["maximum_liquidity_utilization"],
                "participation_breach_fraction": summary["participation_breach_fraction"],
                "estimated_capacity_dollars": summary["estimated_capacity_dollars"],
            }
        )
    return pd.DataFrame(rows)


def compute_path_capacity_sensitivity(
    configured_returns: pd.Series,
    daily_analysis: pd.DataFrame,
    capacity_summary: pd.Series,
    portfolio_name: str,
    target_dollars: float,
    impact_enabled: bool,
    model: str,
    capital_multipliers: tuple[float, ...] = (0.25, 0.50, 1.0, 2.0, 5.0),
) -> pd.DataFrame:
    """Scale a stitched realized OOS path without reconstructing selected weights."""
    applied = daily_analysis["applied_market_impact_cost"].reindex(configured_returns.index).fillna(0.0)
    slippage_only_returns = configured_returns + applied if impact_enabled else configured_returns.copy()
    base_annualized_return, base_sharpe = _performance_attribution(slippage_only_returns)
    base_cost = daily_analysis["modeled_market_impact_cost"].reindex(configured_returns.index).fillna(0.0)
    exponent = 0.5 if model == "square_root" else 1.0
    rows = []
    for multiplier in capital_multipliers:
        scaled_cost = base_cost * multiplier ** exponent
        annualized_return, sharpe = _performance_attribution(slippage_only_returns - scaled_cost)
        rows.append(
            {
                "portfolio": portfolio_name,
                "capital_dollars": target_dollars * multiplier,
                "capital_multiple": multiplier,
                "annualized_return_slippage_only": base_annualized_return,
                "annualized_return_impact_enabled": annualized_return,
                "return_degradation": base_annualized_return - annualized_return,
                "sharpe_slippage_only": base_sharpe,
                "sharpe_impact_enabled": sharpe,
                "sharpe_degradation": base_sharpe - sharpe,
                "modeled_market_impact_cost": scaled_cost.sum(),
                "average_participation_rate": capacity_summary["average_participation_rate"] * multiplier,
                "maximum_participation_rate": capacity_summary["maximum_participation_rate"] * multiplier,
                "average_liquidity_utilization": capacity_summary["average_liquidity_utilization"] * multiplier,
                "maximum_liquidity_utilization": capacity_summary["maximum_liquidity_utilization"] * multiplier,
                "participation_breach_fraction": np.nan,
                "estimated_capacity_dollars": capacity_summary["estimated_capacity_dollars"],
            }
        )
    return pd.DataFrame(rows)


def build_impact_diagnostic_tables(
    impact_result: MarketImpactResult,
    portfolio_name: str,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Create concise tables for expensive trades, impact dates, and bottlenecks."""
    trades = impact_result.trade_details.copy()
    daily = impact_result.daily_analysis.copy().reset_index()
    if trades.empty:
        empty = pd.DataFrame()
        return {
            "execution_dashboard_summary": impact_result.capacity_summary.copy(),
            "most_expensive_trades": empty,
            "highest_impact_dates": empty,
            "liquidity_bottlenecks": empty,
        }

    expensive = trades.sort_values("modeled_market_impact_cost", ascending=False).head(top_n)
    high_impact_dates = daily.sort_values("modeled_market_impact_cost", ascending=False).head(top_n)
    bottlenecks = trades.sort_values("participation_rate", ascending=False).head(top_n)
    summary = impact_result.capacity_summary.copy()
    summary["most_expensive_trade_cost"] = expensive["modeled_market_impact_cost"].iloc[0]
    summary["highest_impact_date"] = high_impact_dates["date"].iloc[0]
    summary["highest_daily_impact_cost"] = high_impact_dates["modeled_market_impact_cost"].iloc[0]
    summary["binding_liquidity_ticker"] = bottlenecks["ticker"].iloc[0]
    summary["binding_participation_rate"] = bottlenecks["participation_rate"].iloc[0]
    summary["portfolio"] = portfolio_name
    return {
        "execution_dashboard_summary": summary,
        "most_expensive_trades": expensive,
        "highest_impact_dates": high_impact_dates,
        "liquidity_bottlenecks": bottlenecks,
    }


def build_execution_cost_report(
    turnover: pd.Series,
    transaction_costs: pd.Series,
    slippage_costs: pd.Series,
    impact_result: MarketImpactResult,
    portfolio_name: str,
) -> pd.DataFrame:
    """Create daily execution-cost reporting from existing and impact layers."""
    index = turnover.index
    output = pd.DataFrame(index=index)
    output["portfolio"] = portfolio_name
    output["turnover"] = turnover.reindex(index).fillna(0.0)
    output["transaction_cost"] = transaction_costs.reindex(index).fillna(0.0)
    output["slippage_cost"] = slippage_costs.reindex(index).fillna(0.0)
    output["market_impact_cost"] = impact_result.applied_costs.reindex(index).fillna(0.0)
    output["modeled_market_impact_cost"] = (
        impact_result.daily_analysis["modeled_market_impact_cost"].reindex(index).fillna(0.0)
    )
    output["total_execution_cost"] = (
        output["transaction_cost"] + output["slippage_cost"] + output["market_impact_cost"]
    )
    output["turnover_adjusted_execution_cost"] = output["total_execution_cost"].div(
        output["turnover"].where(output["turnover"] > 0)
    )
    output.index.name = "date"
    return output
