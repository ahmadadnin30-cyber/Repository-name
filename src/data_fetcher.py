"""
data_fetcher.py
Fetches US stock OHLCV data using yfinance and formats it for Kronos.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given stock ticker.

    Args:
        ticker: Stock symbol e.g. 'AAPL', 'TSLA'
        period: Data period e.g. '1mo', '3mo', '6mo', '1y'
        interval: Data interval e.g. '1d', '1h', '15m'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    print(f"Fetched {len(df)} rows for {ticker}")
    return df


def format_for_kronos(df: pd.DataFrame) -> np.ndarray:
    """
    Format OHLCV DataFrame into numpy array for Kronos model input.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns

    Returns:
        numpy array of shape (sequence_length, 5)
    """
    data = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
    return data


def get_latest_price(ticker: str) -> dict:
    """
    Get the latest price info for a ticker.

    Args:
        ticker: Stock symbol

    Returns:
        dict with current price info
    """
    stock = yf.Ticker(ticker)
    info = stock.fast_info
    return {
        "ticker": ticker,
        "current_price": info.last_price,
        "previous_close": info.previous_close,
        "change_pct": ((info.last_price - info.previous_close) / info.previous_close) * 100
    }


if __name__ == "__main__":
    # Test the fetcher
    df = fetch_stock_data("AAPL", period="3mo", interval="1d")
    print(df.tail())
    print("\nFormatted for Kronos shape:", format_for_kronos(df).shape)

    price = get_latest_price("AAPL")
    print(f"\nLatest price: ${price['current_price']:.2f} ({price['change_pct']:+.2f}%)")
