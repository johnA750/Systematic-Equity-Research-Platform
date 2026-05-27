import logging

import yfinance as yf
import pandas as pd

from data.sp500 import normalize_yahoo_ticker


# Report retained symbols at the application layer instead of printing
# yfinance batch-level no-history diagnostics for current-index research.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _extract_field(data, requested_tickers, field):
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if field in data.columns.get_level_values(0):
            values = data[field]
        elif field in data.columns.get_level_values(1):
            values = data.xs(field, axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if field not in data.columns:
            return pd.DataFrame()
        values = data[field]

    if isinstance(values, pd.Series):
        values = values.to_frame(name=requested_tickers[0])

    values.columns = [normalize_yahoo_ticker(column) for column in values.columns]
    return values.loc[:, ~values.columns.duplicated()]


def load_market_data(tickers, start_date, end_date, batch_size=100):
    """
    Load adjusted-close prices and volume for a universe of symbols.

    Volume is returned when Yahoo provides it. Callers can continue using
    prices if volume is empty or unavailable.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    normalized_tickers = list(
        dict.fromkeys(normalize_yahoo_ticker(ticker) for ticker in tickers)
    )
    price_batches = []
    volume_batches = []

    for start in range(0, len(normalized_tickers), batch_size):
        batch = normalized_tickers[start:start + batch_size]
        request = batch[0] if len(batch) == 1 else batch
        data = yf.download(
            request,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=len(batch) > 1,
        )

        prices = _extract_field(data, batch, "Adj Close")
        volumes = _extract_field(data, batch, "Volume")
        if not prices.empty:
            price_batches.append(prices)
        if not volumes.empty:
            volume_batches.append(volumes)

    if not price_batches:
        return pd.DataFrame(), pd.DataFrame()

    price_data = pd.concat(price_batches, axis=1)
    missing_tickers = [ticker for ticker in normalized_tickers if ticker not in price_data.columns]
    for ticker in missing_tickers:
        print(f"Warning: no data returned for {ticker}")

    volume_data = (
        pd.concat(volume_batches, axis=1)
        if volume_batches else pd.DataFrame(index=price_data.index)
    )
    return price_data, volume_data.reindex(columns=price_data.columns)


def load_price_data(tickers, start_date, end_date, batch_size=100):
    """
    Load historical adjusted close prices for given tickers.
    Returns a DataFrame with one column per ticker.
    """
    prices, _ = load_market_data(tickers, start_date, end_date, batch_size=batch_size)
    return prices


def clean_price_data(prices, max_missing_fraction=None, min_price=None):
    """
    Clean price data and optionally remove assets unsuitable for trading.

    Missing-data and price filters are evaluated before forward filling.
    The minimum-price filter uses the first observed price in the requested
    test window to avoid screening based on information from the end date.
    """
    prices = prices.dropna(how="all")
    prices = prices.dropna(axis=1, how="all")

    if max_missing_fraction is not None:
        missing_fraction = prices.isna().mean()
        prices = prices.loc[:, missing_fraction <= max_missing_fraction]

    if min_price is not None and not prices.empty:
        first_observed_prices = prices.bfill().iloc[0]
        prices = prices.loc[:, first_observed_prices >= min_price]

    prices = prices.ffill()
    return prices.dropna(how="all")
