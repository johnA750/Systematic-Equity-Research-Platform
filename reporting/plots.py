import matplotlib.pyplot as plt


def plot_strategy_vs_benchmark(strategy_cumulative, benchmark_cumulative, strategy_label="Strategy", benchmark_label="Benchmark"):
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(strategy_cumulative.index, strategy_cumulative.values, label=strategy_label)
    axis.plot(benchmark_cumulative.index, benchmark_cumulative.values, label=benchmark_label)
    axis.set_title(f"{strategy_label} vs {benchmark_label}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Return")
    axis.legend()
    axis.grid(True)
    return fig


def plot_drawdown(drawdown_series, title="Drawdown"):
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(drawdown_series.index, drawdown_series.values)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.grid(True)
    return fig


def plot_gross_exposure(
    exposure_series,
    title="Gross Exposure",
    label="Gross Exposure",
):
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(exposure_series.index, exposure_series.values, label=label)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Gross Exposure")
    axis.legend()
    axis.grid(True)
    return fig


def plot_backtest_dashboard(
    cumulative_returns,
    drawdowns,
    gross_exposures,
    metrics_summary,
    capm_summary,
    title="Backtest Comparison Dashboard",
):
    """Plot a consolidated performance comparison dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=False)
    equity_axis, drawdown_axis, exposure_axis, summary_axis = axes.flat

    for label, values in cumulative_returns.items():
        equity_axis.plot(values.index, values.values, label=label, linewidth=2)
    equity_axis.set_title("Cumulative Value")
    equity_axis.set_ylabel("Growth of $1")
    equity_axis.legend(fontsize=9)
    equity_axis.grid(True)

    for label, values in drawdowns.items():
        drawdown_axis.plot(values.index, values.values, label=label, linewidth=1.6)
    drawdown_axis.set_title("Drawdown")
    drawdown_axis.set_ylabel("Drawdown")
    drawdown_axis.legend(fontsize=9)
    drawdown_axis.grid(True)

    for label, values in gross_exposures.items():
        exposure_axis.plot(values.index, values.values, label=label, linewidth=1.8)
    exposure_axis.set_title("Gross Exposure")
    exposure_axis.set_ylabel("Exposure")
    exposure_axis.legend(fontsize=9)
    exposure_axis.grid(True)

    summary_axis.axis("off")
    summary_rows = []
    for portfolio, metrics in metrics_summary.items():
        summary_rows.append(
            [
                portfolio,
                f"{metrics['Annualized Return']:.1%}",
                f"{metrics['Sharpe Ratio']:.2f}",
                f"{metrics['Max Drawdown']:.1%}",
            ]
        )
    table = summary_axis.table(
        cellText=summary_rows,
        colLabels=["Portfolio", "Ann. Return", "Sharpe", "Max DD"],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    capm_lines = [
        "Combined Portfolio CAPM",
        f"Alpha (annualized): {capm_summary['Alpha Annualized']:.2%}",
        f"Beta: {capm_summary['Beta']:.3f}",
        f"Alpha p-value: {capm_summary['Alpha p-value']:.4f}",
    ]
    summary_axis.text(
        0.02,
        0.43,
        "\n".join(capm_lines),
        fontsize=11,
        va="top",
        bbox={"facecolor": "whitesmoke", "edgecolor": "lightgray", "pad": 10},
    )
    summary_axis.set_title("Summary")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def plot_rolling_performance(rolling_metrics, title_prefix="Out-of-Sample"):
    """Plot rolling return and factor diagnostics in a single figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    series_to_plot = [
        ("Rolling Sharpe", axes[0, 0]),
        ("Rolling Volatility", axes[0, 1]),
        ("Rolling Beta", axes[1, 0]),
        ("Rolling Alpha Annualized", axes[1, 1]),
    ]

    for column, axis in series_to_plot:
        axis.plot(rolling_metrics.index, rolling_metrics[column])
        axis.set_title(f"{title_prefix} {column}")
        axis.grid(True)

    fig.tight_layout()
    return fig


def plot_rolling_factor_exposures(rolling_exposures, title_prefix="Portfolio"):
    """Plot rolling Ken French alpha and factor loadings."""
    columns = [
        ("alpha_annualized", "Rolling Alpha (Annualized)"),
        ("market_beta", "Rolling Market Beta"),
        ("smb", "Rolling SMB Exposure"),
        ("hml", "Rolling HML Exposure"),
    ]
    available = [(column, label) for column, label in columns if column in rolling_exposures]
    if not available:
        raise ValueError("No supported rolling factor exposure columns were supplied.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (column, label) in zip(axes.flat, available):
        axis.plot(rolling_exposures.index, rolling_exposures[column])
        axis.set_title(f"{title_prefix} {label}")
        axis.grid(True)
    for axis in axes.flat[len(available):]:
        axis.axis("off")
    fig.tight_layout()
    return fig


def plot_execution_dashboard(
    execution_costs,
    market_impact_analysis,
    cost_contributions,
    capacity_summary,
    title="Execution And Market Impact Dashboard",
):
    """Plot execution costs, impact attribution, and liquidity utilization."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    cumulative_axis, contribution_axis, scatter_axis, participation_axis, utilization_axis, summary_axis = axes.flat

    costs = execution_costs.copy()
    if "path" in costs:
        costs = costs[costs["path"] == costs["path"].iloc[0]]
    costs = costs.sort_values("date") if "date" in costs else costs.sort_index()
    dates = costs["date"] if "date" in costs else costs.index
    for column, label in (
        ("transaction_cost", "Transaction Cost"),
        ("slippage_cost", "Slippage"),
        ("modeled_market_impact_cost", "Modeled Impact"),
    ):
        if column in costs:
            cumulative_axis.plot(dates, costs[column].fillna(0.0).cumsum(), label=label)
    cumulative_axis.set_title("Cumulative Execution Costs")
    cumulative_axis.set_ylabel("Return Cost")
    cumulative_axis.legend(fontsize=9)
    cumulative_axis.grid(True)

    contributions = cost_contributions.copy()
    contribution_axis.bar(
        contributions["component"],
        contributions["modeled_market_impact_cost"],
        color="steelblue",
    )
    contribution_axis.set_title("Impact Cost Contribution")
    contribution_axis.set_ylabel("Modeled Return Cost")
    contribution_axis.tick_params(axis="x", rotation=25)
    contribution_axis.grid(True, axis="y")

    scatter_axis.scatter(
        costs["turnover"],
        costs["modeled_market_impact_cost"],
        alpha=0.55,
        s=20,
    )
    scatter_axis.set_title("Turnover Vs Impact Cost")
    scatter_axis.set_xlabel("Daily Turnover")
    scatter_axis.set_ylabel("Modeled Impact Cost")
    scatter_axis.grid(True)

    impact = market_impact_analysis.copy()
    if "path" in impact:
        impact = impact[impact["path"] == impact["path"].iloc[0]]
    participation_axis.hist(
        impact["maximum_participation_rate"].dropna(),
        bins=25,
        color="darkcyan",
        alpha=0.8,
    )
    participation_axis.set_title("ADV Participation Distribution")
    participation_axis.set_xlabel("Maximum Daily Participation")
    participation_axis.grid(True, axis="y")

    utilization_axis.hist(
        impact["liquidity_utilization"].dropna(),
        bins=25,
        color="darkorange",
        alpha=0.8,
    )
    utilization_axis.axvline(1.0, color="crimson", linestyle="--", label="ADV Limit")
    utilization_axis.set_title("Liquidity Utilization Distribution")
    utilization_axis.set_xlabel("Participation / Limit")
    utilization_axis.legend(fontsize=9)
    utilization_axis.grid(True, axis="y")

    summary_axis.axis("off")
    summary = capacity_summary.iloc[0]
    lines = [
        "Capacity Summary",
        f"Average participation: {summary['average_participation_rate']:.2%}",
        f"Maximum participation: {summary['maximum_participation_rate']:.2%}",
        f"Estimated capacity: ${summary['estimated_capacity_dollars']:,.0f}",
        f"Binding capacity: ${summary['binding_capacity_dollars']:,.0f}",
        f"ADV breach fraction: {summary['participation_breach_fraction']:.2%}",
        f"Modeled impact cost: {summary['modeled_market_impact_cost']:.2%}",
    ]
    summary_axis.text(
        0.03,
        0.93,
        "\n".join(lines),
        va="top",
        fontsize=12,
        linespacing=1.5,
        bbox={"facecolor": "whitesmoke", "edgecolor": "lightgray", "pad": 12},
    )
    summary_axis.set_title("Headline Diagnostics")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def plot_capacity_curves(capacity_sensitivity, title="Portfolio Capacity Sensitivity"):
    """Plot participation, impact cost, return, and Sharpe as capital scales."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    data = capacity_sensitivity.sort_values("capital_dollars")
    for portfolio, group in data.groupby("portfolio"):
        capital_millions = group["capital_dollars"] / 1_000_000.0
        axes[0, 0].plot(capital_millions, group["maximum_participation_rate"], marker="o", label=portfolio)
        axes[0, 1].plot(capital_millions, group["modeled_market_impact_cost"], marker="o", label=portfolio)
        axes[1, 0].plot(capital_millions, group["annualized_return_impact_enabled"], marker="o", label=portfolio)
        axes[1, 1].plot(capital_millions, group["sharpe_impact_enabled"], marker="o", label=portfolio)
    axes[0, 0].set_title("Maximum ADV Participation")
    axes[0, 0].set_ylabel("Participation Rate")
    axes[0, 1].set_title("Modeled Market Impact Cost")
    axes[0, 1].set_ylabel("Return Cost")
    axes[1, 0].set_title("Annualized Return After Impact")
    axes[1, 0].set_ylabel("Annualized Return")
    axes[1, 1].set_title("Sharpe After Impact")
    axes[1, 1].set_ylabel("Sharpe Ratio")
    for axis in axes.flat:
        axis.set_xlabel("Capital ($ millions)")
        axis.grid(True)
        axis.legend(fontsize=8)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def plot_impact_comparison(
    comparison_table,
    cumulative_paths=None,
    title="Execution Realism Comparison",
):
    """Compare execution scenarios and, when supplied, their cumulative paths."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    paths_axis, return_axis, sharpe_axis = axes
    if cumulative_paths:
        for label, values in cumulative_paths.items():
            paths_axis.plot(values.index, values.values, label=label, linewidth=2)
        paths_axis.legend(fontsize=9)
    paths_axis.set_title("Cumulative Value By Execution Scenario")
    paths_axis.set_ylabel("Growth Of $1")
    paths_axis.grid(True)

    table = comparison_table.copy()
    return_axis.bar(table["scenario"], table["annualized_return"], color="steelblue")
    return_axis.set_title("Annualized Return Comparison")
    return_axis.set_ylabel("Annualized Return")
    return_axis.tick_params(axis="x", rotation=25)
    return_axis.grid(True, axis="y")

    sharpe_axis.bar(table["scenario"], table["sharpe"], color="darkorange")
    sharpe_axis.set_title("Sharpe Comparison")
    sharpe_axis.set_ylabel("Sharpe Ratio")
    sharpe_axis.tick_params(axis="x", rotation=25)
    sharpe_axis.grid(True, axis="y")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def plot_survivorship_bias_analysis(
    point_in_time_cumulative,
    survivor_only_cumulative,
    universe_turnover,
    survivorship_comparison,
    title="Survivorship Bias Analysis",
):
    """Plot point-in-time versus survivor-only performance and membership changes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cumulative_axis, spread_axis, changes_axis, summary_axis = axes.flat

    cumulative_axis.plot(
        point_in_time_cumulative.index,
        point_in_time_cumulative.values,
        label="Point-In-Time Membership",
        linewidth=2,
    )
    cumulative_axis.plot(
        survivor_only_cumulative.index,
        survivor_only_cumulative.values,
        label="End-Date Survivors Only",
        linewidth=2,
    )
    cumulative_axis.set_title("Cumulative Value Comparison")
    cumulative_axis.set_ylabel("Growth Of $1")
    cumulative_axis.legend()
    cumulative_axis.grid(True)

    aligned = survivor_only_cumulative.reindex(point_in_time_cumulative.index).ffill()
    spread_axis.plot(
        point_in_time_cumulative.index,
        aligned - point_in_time_cumulative,
        color="darkorange",
        linewidth=2,
    )
    spread_axis.axhline(0.0, color="gray", linewidth=1)
    spread_axis.set_title("Survivor-Only Value Minus Point-In-Time")
    spread_axis.set_ylabel("Cumulative Value Difference")
    spread_axis.grid(True)

    annual = universe_turnover.copy()
    annual["year"] = annual["date"].dt.year
    annual = annual.groupby("year", as_index=False)[["additions", "removals"]].sum()
    changes_axis.bar(annual["year"] - 0.18, annual["additions"], width=0.36, label="Additions")
    changes_axis.bar(annual["year"] + 0.18, annual["removals"], width=0.36, label="Removals")
    changes_axis.set_title("Annual Membership Changes")
    changes_axis.set_ylabel("Securities")
    changes_axis.legend()
    changes_axis.grid(True, axis="y")

    point_row = survivorship_comparison[
        survivorship_comparison["universe"] == "point_in_time_membership"
    ].iloc[0]
    survivor_row = survivorship_comparison[
        survivorship_comparison["universe"] == "end_date_survivors_only"
    ].iloc[0]
    summary_axis.axis("off")
    lines = [
        "Estimated Survivorship Effect",
        f"Point-in-time ann. return: {point_row['Annualized Return']:.2%}",
        f"Survivor-only ann. return: {survivor_row['Annualized Return']:.2%}",
        f"Return bias: {survivor_row['annualized_return_bias_vs_point_in_time']:.2%}",
        f"Point-in-time Sharpe: {point_row['Sharpe Ratio']:.2f}",
        f"Survivor-only Sharpe: {survivor_row['Sharpe Ratio']:.2f}",
        f"Sharpe bias: {survivor_row['sharpe_bias_vs_point_in_time']:.2f}",
    ]
    summary_axis.text(
        0.05,
        0.9,
        "\n".join(lines),
        va="top",
        fontsize=12,
        linespacing=1.5,
        bbox={"facecolor": "whitesmoke", "edgecolor": "lightgray", "pad": 12},
    )
    summary_axis.set_title("Summary")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig
