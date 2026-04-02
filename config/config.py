TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH",
    "HD", "PG", "MA", "V", "DIS", "BAC", "KO", "PEP", "AVGO", "COST",
    "ABBV", "MRK", "LLY", "WMT", "MCD", "CVX", "ADBE", "CRM", "ACN", "AMD",
    "CSCO", "INTC", "QCOM", "TXN", "NFLX", "TMO", "DHR", "LIN", "NEE", "PM",
    "RTX", "HON", "IBM", "CAT", "GS", "MS", "BLK", "AMGN", "LOW", "INTU"
]

BENCHMARK_TICKER = ["SPY"]

START_DATE = "2020-01-01"
END_DATE = "2023-01-01"

MOMENTUM_LOOKBACK = 20
TOP_N = 10
REBALANCE_FREQ = 20
TRANSACTION_COST = 0.001

# OPTIMIZATION_CONFIG = {
#     "enabled": True,
#     "turnover_penalty_strength": 0.003,
#     "regularization_strength": 10.0,
#     "regularization_target": "inverse_vol",
#     "weight_bounds": {
#         "momentum": (0.0, 1.0),
#         "mean_reversion": (0.0, 1.0),
#     },
#     "rolling_lookback_window": 504,
#     "rolling_rebalance_frequency": 63,
# } 