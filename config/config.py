TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH",
    "HD", "PG", "MA", "V", "DIS", "BAC", "KO", "PEP", "AVGO", "COST",
    "ABBV", "MRK", "LLY", "WMT", "MCD", "CVX", "ADBE", "CRM", "ACN", "AMD",
    "CSCO", "INTC", "QCOM", "TXN", "NFLX", "TMO", "DHR", "LIN", "NEE", "PM",
    "RTX", "HON", "IBM", "CAT", "GS", "MS", "BLK", "AMGN", "LOW", "INTU"
]

BENCHMARK_TICKER = ["SPY"]

# Leave disabled to retain the existing fixed-universe research baseline.
# When enabled, current S&P 500 membership is loaded at runtime.
USE_SP500_UNIVERSE = False

START_DATE = "2020-01-01"
END_DATE = "2023-01-01"

MOMENTUM_LOOKBACK = 120
TOP_N = 10
REBALANCE_FREQ = 10
TRANSACTION_COST = 0.001

# Tradable-universe quality controls. The price screen uses the first
# available close in the test window so it does not use an ending price.
MAX_MISSING_DATA_FRACTION = 0.20
MIN_STOCK_PRICE = 5.0
DATA_DOWNLOAD_BATCH_SIZE = 100

# Optional daily liquidity eligibility screen. ADV is computed from trailing
# adjusted-close dollar volume and lagged one day before use.
ENABLE_LIQUIDITY_FILTER = False
MIN_AVERAGE_DOLLAR_VOLUME = 5_000_000
ADV_LOOKBACK_WINDOW = 20

# Set to a decimal such as 0.10 to limit either sleeve to 10% per name.
# None preserves the original weighting behavior.
MAX_POSITION_WEIGHT = None

# Execution model settings. "fixed" preserves the existing results.
SLIPPAGE_MODEL = "fixed"
SLIPPAGE_BPS = 5.0
SLIPPAGE_VOLATILITY_WINDOW = 20
SLIPPAGE_VOLATILITY_REFERENCE = 0.02
SLIPPAGE_TURNOVER_MULTIPLIER = 1.0

# Optional research-only SciPy optimizer; inverse-vol remains production default.
ENABLE_CONSTRAINED_OPTIMIZER = False
OPTIMIZER_MAX_WEIGHT = 0.75
