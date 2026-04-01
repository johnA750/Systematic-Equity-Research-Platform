from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from strategies.momentum import generate_momentum_signals
from engine.backtest import run_backtest
from analytics.performance import compute_cumulative_returns


tickers = ["AAPL", "MSFT", "AMZN"]
start = "2020-01-01"
end = "2023-01-01"

prices = load_price_data(tickers, start, end)
prices = clean_price_data(prices)

returns = compute_daily_returns(prices)
signals = generate_momentum_signals(prices, lookback=20)

portfolio_returns = run_backtest(signals, returns)

print("Portfolio Returns:")
print(portfolio_returns.head(30))

cumulative = compute_cumulative_returns(portfolio_returns)

print("Cumulative Performance:")
print(cumulative.tail())

print("\nSignals tail:")
print(signals.tail(10))

print("\nShifted signals tail:")
print(signals.shift(1).tail(10))

print("\nPortfolio returns tail:")
print(portfolio_returns.tail(10))

print("\nActive positions tail:")
print(signals.shift(1).sum(axis=1).tail(10))