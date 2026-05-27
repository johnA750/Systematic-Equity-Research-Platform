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
