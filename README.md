# Systematic Equity Backtesting and Research Platform

A modular Python platform for testing long-only U.S. equity strategies across a fixed production baseline, a current S&P 500 research universe, and walk-forward out-of-sample validation.

The platform keeps signal logic simple and auditable while adding portfolio construction, execution modeling, exposure analysis, factor decomposition, and reproducible experiment reporting.

## Highlights

- Momentum and pullback-based mean reversion strategy sleeves
- Inverse-volatility production portfolio construction with volatility targeting
- Production 50-stock baseline and optional current S&P 500 research universe
- Missing-data, price, and optional lagged average-dollar-volume eligibility filters
- Transaction costs plus fixed, volatility-adjusted, or turnover-adjusted slippage
- Optional ADV- and volatility-based market impact with deployable-capacity estimates
- Position caps, turnover tracking, gross exposure, drawdown, and sector exposure reporting
- CAPM and Fama-French 3-factor / optional 5-factor attribution with rolling exposures
- Rolling or expanding walk-forward validation comparing selected parameters with fixed production parameters
- Timestamped report folders containing CSV outputs, plots, and configuration metadata

## Research Caveat

The S&P 500 mode retrieves the current constituent list and applies it across historical returns. It is useful for testing scalability and execution realism, but it is not a survivorship-bias-free historical index-membership study. Results from that mode should be described as current-constituent research, not point-in-time index performance.

## Strategy And Portfolio Design

The production strategy behavior remains intentionally stable:

| Component | Implementation |
| --- | --- |
| Momentum sleeve | Ranks equities by trailing returns and holds the strongest names on rebalance dates. |
| Mean reversion sleeve | Selects qualifying pullbacks within the existing trend and pullback rules. |
| Portfolio construction | Combines sleeves using inverse-volatility weights by default. |
| Volatility control | Applies portfolio-level volatility targeting. |
| Risk controls | Optional maximum position weight and gross exposure reporting. |
| Execution | Transaction costs, configurable slippage, and optional market impact applied to observed trades. |

Research features evaluate these decisions; they do not replace the production signal logic or production parameter defaults.

## Experiment Modes

| Mode | Entry Point | Universe | Purpose |
| --- | --- | --- | --- |
| Production baseline | `run_production_baseline.py` | Fixed 50-stock universe | Stable production-parameter reference run |
| S&P 500 research | `research/run_sp500_research.py` | Current S&P 500 constituents | Scalability, liquidity, sector, and execution research |
| Walk-forward validation | `research/run_walk_forward.py` | Configured validation universe | Rolling or expanding out-of-sample robustness testing |

Run the modes from the repository root:

```powershell
# Production 50-stock baseline
.\.venv\Scripts\python.exe run_production_baseline.py

# Current S&P 500 scalability research
.\.venv\Scripts\python.exe research\run_sp500_research.py

# Walk-forward out-of-sample validation
.\.venv\Scripts\python.exe research\run_walk_forward.py
```

`run_backtest.py` remains a backwards-compatible production baseline entry point.

## Quick Start

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy matplotlib yfinance statsmodels scipy
python run_production_baseline.py
```

`scipy` is used only when the optional constrained portfolio research optimizer is enabled. Factor analysis downloads official Ken French factor files at run time when enabled.

## Configuration

Primary settings are in `config/config.py`. The mode definitions in `config/modes.py` provide coherent baseline, S&P 500, and walk-forward profiles.

| Configuration Area | Key Settings | Purpose |
| --- | --- | --- |
| Universe | `USE_SP500_UNIVERSE`, `TICKERS`, `BENCHMARK_TICKER` | Choose the fixed universe or current S&P 500 constituents. |
| Dates and strategy | `START_DATE`, `END_DATE`, `MOMENTUM_LOOKBACK`, `TOP_N`, `REBALANCE_FREQ` | Define the backtest period and existing strategy defaults. |
| Data quality | `MAX_MISSING_DATA_FRACTION`, `MIN_STOCK_PRICE`, `DATA_DOWNLOAD_BATCH_SIZE` | Filter incomplete or impractical data and support large downloads. |
| Liquidity | `ENABLE_LIQUIDITY_FILTER`, `MIN_AVERAGE_DOLLAR_VOLUME`, `ADV_LOOKBACK_WINDOW` | Apply lagged rolling dollar-volume eligibility when volume is available. |
| Position risk | `MAX_POSITION_WEIGHT` | Optionally cap individual position weights. |
| Execution | `TRANSACTION_COST`, `SLIPPAGE_MODEL`, `SLIPPAGE_BPS` | Control transaction costs and fixed or adaptive slippage assumptions. |
| Market impact | `ENABLE_MARKET_IMPACT`, `MARKET_IMPACT_MODEL`, `MAX_ADV_PARTICIPATION`, `IMPACT_COEFFICIENT`, `CAPACITY_TARGET_DOLLARS` | Estimate trade-size impact and constrain capacity assumptions against lagged ADV. |
| Reporting | `SAVE_REPORTS`, `SAVE_PLOTS`, `SAVE_INDIVIDUAL_PLOTS`, `VERBOSE_ANALYTICS` | Control exported artifacts and console detail. |
| Factor analysis | `ENABLE_FACTOR_ANALYSIS`, `ENABLE_FF5_ANALYSIS`, `FACTOR_FREQUENCY`, `FACTOR_ROLLING_WINDOW` | Configure attribution and rolling exposure reporting. |
| Walk-forward | `WALK_FORWARD_MODE`, `WALK_FORWARD_TRAIN_LENGTH`, `WALK_FORWARD_TEST_LENGTH`, `WALK_FORWARD_STEP_LENGTH` | Configure out-of-sample evaluation windows. |

The fixed baseline defaults preserve the original execution path. Advanced liquidity constraints, adaptive slippage, market impact charges, S&P 500 selection, and constrained optimization are research controls that can be enabled explicitly. Capacity diagnostics can still be reported while market impact charges are disabled.

## Market Impact And Capacity

`execution/market_impact.py` models rebalance trades as portfolio-weight changes multiplied by `CAPACITY_TARGET_DOLLARS`. Each trade is evaluated against lagged rolling average dollar volume and lagged realized volatility, keeping the execution estimate free from same-day liquidity lookahead.

Two optional impact specifications are available:

| Model | Impact Rate |
| --- | --- |
| `square_root` | `IMPACT_COEFFICIENT * realized_volatility * sqrt(participation_rate)` |
| `linear` | `IMPACT_COEFFICIENT * realized_volatility * participation_rate` |

Participation rate is traded dollars divided by lagged ADV. Estimated capacity is reported as a conservative lower-tail capital level at which trade participation reaches `MAX_ADV_PARTICIPATION`; a binding minimum capacity estimate and breach frequency are also exported. When `ENABLE_MARKET_IMPACT = False`, modeled costs and capacity remain visible for research, but zero market impact is deducted from returns.

## Walk-Forward Research

The walk-forward runner supports both rolling-window and expanding-window studies. For each test period it evaluates:

- A train-selected parameter path chosen using training-window results.
- A fixed-production-parameter path using the unchanged production defaults.

This comparison directly tests whether parameter selection improves out-of-sample performance or merely adds instability. Reported outputs include train and test metrics, test-period CAPM summaries, Sharpe degradation, drawdown comparisons, positive test-period fraction, and parameter stability information.

Key exports include:

```text
selected_results.csv
fixed_parameter_results.csv
path_comparison.csv
parameter_summary.csv
stability_summary.csv
rolling_metrics.csv
fixed_rolling_metrics.csv
```

## Factor Attribution

Factor attribution extends the existing SPY-based CAPM workflow without changing signals or positions. The reporting layer retrieves factor data from the official [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and estimates regressions with `statsmodels` OLS, using HAC/Newey-West standard errors by default.

| Model | Exposures |
| --- | --- |
| CAPM | Market excess return (`Mkt-RF`) |
| Fama-French 3-factor | `Mkt-RF`, size (`SMB`), value (`HML`) |
| Fama-French 5-factor | `Mkt-RF`, `SMB`, `HML`, profitability (`RMW`), investment (`CMA`) |

Daily attribution aligns realized portfolio returns to factor observations on the same realized trading dates. Monthly attribution compounds portfolio returns to calendar months before matching monthly factor data. Factors are used for after-the-fact analysis only, avoiding lookahead into strategy construction.

Exports include:

```text
factor_summary.csv
rolling_factor_exposures.csv
```

## Reporting Artifacts

When `SAVE_REPORTS = True`, every mode creates a timestamped experiment directory:

```text
reports/
|-- production_baseline/
|   `-- <timestamp>/
|-- sp500_research/
|   `-- <timestamp>/
`-- walk_forward_validation/
    `-- <timestamp>/
```

Each experiment folder can contain:

- Summary and detailed metrics tables in CSV format
- CAPM, factor, exposure, sector, or walk-forward result tables where applicable
- `execution_costs.csv`, `capacity_summary.csv`, and `market_impact_analysis.csv`
- `comparison_dashboard.png`, combining headline performance views for quick review
- Rolling performance and rolling factor exposure plots
- `experiment_metadata.json`, recording the timestamp, universe, configuration snapshot, and execution assumptions

Set `VERBOSE_ANALYTICS = True` for detailed console reporting. The default console output stays focused on headline metrics while detailed results remain available in exported artifacts.

## Project Structure

```text
Trading-Backtester/
|-- analytics/
|   |-- factors.py                # CAPM, FF3/FF5, and rolling factor exposures
|   |-- factor_regression.py      # Existing benchmark regression workflow
|   |-- exposure.py               # Gross and sector exposure analytics
|   |-- metrics.py                # Portfolio performance metrics
|   |-- regime_analysis.py        # Regime-based performance reporting
|   `-- rolling.py                # Rolling performance statistics
|-- config/
|   |-- config.py                 # User configuration and production defaults
|   `-- modes.py                  # Explicit production/research mode profiles
|-- data/
|   |-- loader.py                 # Historical price and volume loading
|   |-- liquidity.py              # ADV-based eligibility filters
|   `-- sp500.py                  # Current S&P 500 constituents and sectors
|-- engine/
|   `-- backtest.py               # Backtest orchestration
|-- execution/
|   |-- slippage.py               # Fixed and optional adaptive slippage models
|   `-- market_impact.py          # ADV participation and capacity diagnostics
|-- portfolio/
|   |-- optimization.py           # Optional research-only constrained optimizer
|   `-- vol_targeting.py          # Volatility targeting utilities
|-- reporting/
|   |-- experiments.py            # Metadata and structured report exports
|   `-- plots.py                  # Dashboard and research plots
|-- research/
|   |-- walk_forward.py           # Windowing, evaluation, and result structures
|   |-- run_walk_forward.py       # Walk-forward comparison runner
|   |-- run_sp500_research.py     # Current S&P 500 research runner
|   |-- momentum_tuning.py        # Momentum research helpers
|   `-- mean_reversion_tuning.py  # Mean reversion research helpers
|-- strategies/
|   |-- momentum.py               # Momentum strategy logic
|   |-- mean_reversion.py         # Pullback strategy logic
|   `-- position_sizing.py        # Position cap utilities
|-- docs/
|   `-- project_plan.md           # Development roadmap
|-- run_backtest.py               # Backwards-compatible main runner
`-- run_production_baseline.py    # Production baseline launcher
```

## Example Validated Results

The following outputs are from the configured historical test window used during development. They are examples of the platform output, not live performance claims or investment advice.

| Run | Annualized Return | Sharpe | Maximum Drawdown |
| --- | ---: | ---: | ---: |
| Fixed 50-stock production baseline | 22.89% | 1.58 | -8.65% |
| Current S&P 500 research universe after filtering | 9.83% | 0.69 | -12.31% |

In the current S&P 500 research run, 503 constituents were retrieved and 487 were retained after filtering. In the walk-forward comparison, the fixed-production-parameter path achieved a higher average test Sharpe than the train-selected path, supporting the decision to keep production parameters stable while continuing robustness research.

## Data And Methodology Assumptions

- Historical equity prices and available volume data are downloaded through `yfinance`.
- S&P 500 constituents and sectors represent the current list retrieved for the research run, not historical membership.
- Missing-data and low-price filters are applied before eligible names are selected for trading.
- Liquidity eligibility uses lagged rolling average dollar volume when volume is available, so current-day volume is not used to choose current-day positions.
- Execution models estimate transaction costs, slippage, and liquidity-based market impact; they do not model complete institutional capacity, borrow constraints, or point-in-time corporate actions.
- Factor models are attribution tools and do not feed into strategy signals or portfolio optimization by default.

## Scope

The repository is a systematic research platform focused on transparent signal evaluation, portfolio risk handling, and out-of-sample robustness. It is not an execution system, brokerage integration, or investment recommendation.
