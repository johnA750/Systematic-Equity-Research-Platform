from data.loader import load_price_data, clean_price_data
from analytics.returns import compute_daily_returns
from strategies.momentum import generate_momentum_signals
from engine.backtest import run_backtest


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