import matplotlib.pyplot as plt

from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from analytics.performance import compute_cumulative_returns
from analytics.metrics import compute_performance_metrics
from strategies.momentum import generate_momentum_weights
from strategies.mean_reversion import generate_mean_reversion_signals
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
meanrev_lookback = 5
top_n = 3
transaction_cost = 0.001

# Load and clean price data
prices = load_price_data(tickers, start, end)
prices = clean_price_data(prices)

# Compute asset returns
returns = compute_daily_returns(prices)

# Generate strategy inputs
momentum_weights = generate_momentum_weights(
    prices,
    lookback=momentum_lookback,
    top_n=top_n
)

meanrev_signals = generate_mean_reversion_signals(
    prices,
    lookback=meanrev_lookback,
    top_n=top_n
)

# Run backtests
momentum_returns = run_backtest(
    momentum_weights,
    returns,
    transaction_cost=transaction_cost
)

meanrev_returns = run_backtest(
    meanrev_signals,
    returns,
    transaction_cost=transaction_cost
)

combined_returns = 0.5 * momentum_returns + 0.5 * meanrev_returns

# Compute cumulative returns
momentum_cumulative = compute_cumulative_returns(momentum_returns)
meanrev_cumulative = compute_cumulative_returns(meanrev_returns)
combined_cumulative = compute_cumulative_returns(combined_returns)

# Compute strategy metrics
momentum_metrics = compute_performance_metrics(momentum_returns)
meanrev_metrics = compute_performance_metrics(meanrev_returns)
combined_metrics = compute_performance_metrics(combined_returns)

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

print("\nMean Reversion Strategy Metrics:")
for key, value in meanrev_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSPY Benchmark Metrics:")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nCombined Strategy Metrics:")
for key, value in combined_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nFinal Cumulative Values:")
print(f"Momentum: {momentum_cumulative.iloc[-1]:.4f}")
print(f"Mean Reversion: {meanrev_cumulative.iloc[-1]:.4f}")
print(f"SPY: {benchmark_cumulative.iloc[-1]:.4f}")
print(f"Combined: {combined_cumulative.iloc[-1]:.4f}")

# Plot performance
plt.figure(figsize=(10, 6))
plt.plot(momentum_cumulative, label="Momentum")
plt.plot(meanrev_cumulative, label="Mean Reversion")
plt.plot(benchmark_cumulative, label="SPY")
plt.plot(combined_cumulative, label="Combined")

plt.title("Strategy Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()