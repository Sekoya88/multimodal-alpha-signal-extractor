"""indicators.py — Domain math for technical indicators.

These functions are pure and testable without infrastructure dependencies.
"""

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (middle, upper, lower)."""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def add_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """Add technical indicators to an OHLCV DataFrame in-place."""
    df["RSI"] = compute_rsi(df["Close"], period=rsi_period)
    bb_mid, bb_up, bb_low = compute_bollinger_bands(
        df["Close"], period=bb_period, num_std=bb_std
    )
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_up
    df["BB_Lower"] = bb_low
    return df
