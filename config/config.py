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

# Reproducible experiment/report settings shared by run modes.
REPORT_ROOT = "reports"
SAVE_REPORTS = True
SAVE_PLOTS = True
SAVE_INDIVIDUAL_PLOTS = False
VERBOSE_ANALYTICS = False

# Walk-forward research settings. These are used only by
# research/run_walk_forward.py; the production full-sample runner is unchanged.
WALK_FORWARD_MODE = "rolling"
WALK_FORWARD_TRAIN_LENGTH = 252
WALK_FORWARD_TEST_LENGTH = 126
WALK_FORWARD_STEP_LENGTH = 126
WALK_FORWARD_ROLLING_METRICS_WINDOW = 63
WALK_FORWARD_EXPORT_RESULTS = True
WALK_FORWARD_SAVE_PLOTS = True

# Fixed production parameters used as the robustness benchmark in walk-forward
# validation. These mirror the production runner without changing its logic.
PRODUCTION_MOMENTUM_PARAMS = {
    "lookback": MOMENTUM_LOOKBACK,
    "top_n": TOP_N,
    "rebalance_freq": REBALANCE_FREQ,
    "vol_lookback": 20,
}

PRODUCTION_MEAN_REVERSION_PARAMS = {
    "lookback": 5,
    "top_n": 15,
    "rebalance_freq": 10,
    "trend_window": 40,
    "min_pullback": -0.10,
    "max_pullback": -0.03,
    "weighting_scheme": "equal",
}

WALK_FORWARD_MOMENTUM_GRID = [
    {"lookback": 90, "top_n": TOP_N, "rebalance_freq": REBALANCE_FREQ, "vol_lookback": 20},
    {"lookback": MOMENTUM_LOOKBACK, "top_n": TOP_N, "rebalance_freq": REBALANCE_FREQ, "vol_lookback": 20},
    {"lookback": 180, "top_n": TOP_N, "rebalance_freq": REBALANCE_FREQ, "vol_lookback": 20},
]

WALK_FORWARD_MEAN_REVERSION_GRID = [
    {
        "lookback": 3,
        "top_n": 15,
        "rebalance_freq": 10,
        "trend_window": 40,
        "min_pullback": -0.10,
        "max_pullback": -0.03,
        "weighting_scheme": "equal",
    },
    {
        "lookback": 5,
        "top_n": 15,
        "rebalance_freq": 10,
        "trend_window": 40,
        "min_pullback": -0.10,
        "max_pullback": -0.03,
        "weighting_scheme": "equal",
    },
    {
        "lookback": 7,
        "top_n": 15,
        "rebalance_freq": 10,
        "trend_window": 40,
        "min_pullback": -0.10,
        "max_pullback": -0.03,
        "weighting_scheme": "equal",
    },
]
