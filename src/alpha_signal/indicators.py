"""indicators.py — Technical analysis indicators (shared module).

Extracted from 01_generate_dataset.py and 05_live_analysis.py to eliminate
code duplication. All indicator functions operate on pandas Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        series: Price series (typically 'Close').
        period: Look-back window for RSI computation.

    Returns:
        RSI values as a pandas Series (0–100 scale).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (middle, upper, lower).

    Args:
        series: Price series (typically 'Close').
        period: SMA window length.
        num_std: Number of standard deviations for the bands.

    Returns:
        Tuple of (middle_band, upper_band, lower_band).
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def add_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """Add RSI and Bollinger Bands columns to a DataFrame in-place.

    Args:
        df: DataFrame with a 'Close' column.
        rsi_period: RSI look-back window.
        bb_period: Bollinger Bands SMA window.
        bb_std: Bollinger Bands standard deviation multiplier.

    Returns:
        The same DataFrame with added columns: RSI, BB_Middle, BB_Upper, BB_Lower.
    """
    df["RSI"] = compute_rsi(df["Close"], period=rsi_period)
    bb_mid, bb_up, bb_low = compute_bollinger_bands(
        df["Close"], period=bb_period, num_std=bb_std
    )
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_up
    df["BB_Lower"] = bb_low
    return df
