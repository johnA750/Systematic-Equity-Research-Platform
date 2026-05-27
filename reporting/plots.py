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
