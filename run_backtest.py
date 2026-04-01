from data.loader import load_price_data


tickers = ["AAPL", "MSFT", "AMZN"]
start = "2020-01-01"
end = "2023-01-01"

prices = load_price_data(tickers, start, end)

print(prices.head())