import matplotlib.pyplot as plt


def plot_strategy_vs_benchmark(strategy_cumulative, benchmark_cumulative,
                               strategy_label="Strategy", benchmark_label="Benchmark"):
    plt.figure(figsize=(10, 6))
    plt.plot(strategy_cumulative, label=strategy_label)
    plt.plot(benchmark_cumulative, label=benchmark_label)

    plt.title(f"{strategy_label} vs {benchmark_label}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_drawdown(drawdown_series, title="Drawdown"):
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown_series, label="Drawdown")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.show()