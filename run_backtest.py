import matplotlib.pyplot as plt

from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics
from analytics.turnover import compute_turnover_metrics
from strategies.momentum import generate_momentum_weights
from engine.backtest import run_backtest


tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
    "META", "TSLA", "JPM", "XOM", "UNH",
    "HD", "PG", "MA", "V", "DIS"
]
benchmark_ticker = ["SPY"]

start = "2020-01-01"
end = "2023-01-01"

momentum_lookback = 20
top_n = 3
rebalance_freq = 20
transaction_cost = 0.001

# Load and clean price data
prices = load_price_data(tickers, start, end)
prices = clean_price_data(prices)

# Compute asset returns
returns = compute_daily_returns(prices)

# Generate momentum weights
momentum_weights = generate_momentum_weights(
    prices,
    lookback=momentum_lookback,
    top_n=top_n,
    rebalance_freq=rebalance_freq
)

# Run momentum backtest
momentum_returns, momentum_turnover = run_backtest(
    momentum_weights,
    returns,
    transaction_cost=transaction_cost,
    return_turnover=True
)

momentum_cumulative = compute_cumulative_returns(momentum_returns)
momentum_metrics = compute_performance_metrics(momentum_returns)
turnover_metrics = compute_turnover_metrics(momentum_turnover)

# Load benchmark
benchmark_prices = load_price_data(benchmark_ticker, start, end)
benchmark_prices = clean_price_data(benchmark_prices)

benchmark_returns_df = compute_daily_returns(benchmark_prices)
benchmark_returns = benchmark_returns_df.iloc[:, 0]

benchmark_cumulative = compute_cumulative_returns(benchmark_returns)
benchmark_metrics = compute_performance_metrics(benchmark_returns)

# Print metrics
print("Momentum Strategy Metrics:")
for key, value in momentum_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMomentum Turnover Metrics:")
for key, value in turnover_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")

# Plot performance
plt.figure(figsize=(10, 6))
plt.plot(momentum_cumulative, label="Momentum (Monthly Rebalance)")
plt.plot(benchmark_cumulative, label="SPY")

plt.title("Momentum vs SPY")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()