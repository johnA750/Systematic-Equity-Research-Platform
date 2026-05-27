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
