from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from strategies.momentum import generate_momentum_signals
from engine.backtest import run_backtest
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics


tickers = ["AAPL", "MSFT", "AMZN"]
benchmark_ticker = ["SPY"]
start = "2020-01-01"
end = "2023-01-01"
lookback = 20
top_n = 2

# Strategy universe
prices = load_price_data(tickers, start, end)
prices = clean_price_data(prices)

returns = compute_daily_returns(prices)
signals = generate_momentum_signals(prices, lookback=lookback, top_n=top_n)

portfolio_returns = run_backtest(signals, returns)
strategy_cumulative = compute_cumulative_returns(portfolio_returns)
strategy_metrics = compute_performance_metrics(portfolio_returns)

# Benchmark
benchmark_prices = load_price_data(benchmark_ticker, start, end)
benchmark_prices = clean_price_data(benchmark_prices)

benchmark_returns_df = compute_daily_returns(benchmark_prices)
benchmark_returns = benchmark_returns_df.iloc[:, 0]

benchmark_cumulative = compute_cumulative_returns(benchmark_returns)
benchmark_metrics = compute_performance_metrics(benchmark_returns)

print("Strategy Performance Metrics:")
for key, value in strategy_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Strategy: {strategy_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(strategy_cumulative, label="Strategy")
plt.plot(benchmark_cumulative, label="SPY")

plt.title("Strategy vs SPY Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")

plt.legend()
plt.grid(True)

plt.show()