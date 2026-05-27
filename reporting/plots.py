import matplotlib.pyplot as plt


def plot_strategy_vs_benchmark(strategy_cumulative, benchmark_cumulative, strategy_label="Strategy", benchmark_label="Benchmark"):
    plt.figure(figsize=(10, 6))
    plt.plot(strategy_cumulative.index, strategy_cumulative.values, label=strategy_label)
    plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, label=benchmark_label)
    plt.title(f"{strategy_label} vs {benchmark_label}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)


def plot_drawdown(drawdown_series, title="Drawdown"):
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown_series.index, drawdown_series.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)


def plot_gross_exposure(
    exposure_series,
    title="Gross Exposure",
    label="Gross Exposure",
):
    plt.figure(figsize=(10, 6))
    plt.plot(exposure_series.index, exposure_series.values, label=label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Gross Exposure")
    plt.legend()
    plt.grid(True)


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
