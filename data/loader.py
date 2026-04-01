import yfinance as yf
import pandas as pd


def load_price_data(tickers, start_date, end_date):
    """
    Load historical adjusted close prices for given tickers.
    Returns a DataFrame with one column per ticker.
    """
    price_data = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if data.empty:
            print(f"Warning: no data returned for {ticker}")
            continue

        price_data[ticker] = data["Adj Close"]

    return price_data