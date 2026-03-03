"""Tests for technical indicators module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_signal.indicators import (
    add_indicators,
    compute_bollinger_bands,
    compute_rsi,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def price_series() -> pd.Series:
    """Generate a synthetic price series for testing."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.Series(prices, name="Close")


@pytest.fixture
def ohlcv_df(price_series: pd.Series) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    close = price_series
    return pd.DataFrame({
        "Open": close * (1 + np.random.uniform(-0.01, 0.01, len(close))),
        "High": close * (1 + np.random.uniform(0, 0.02, len(close))),
        "Low": close * (1 - np.random.uniform(0, 0.02, len(close))),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, len(close)),
    })


# ============================================================================
# RSI Tests
# ============================================================================

class TestComputeRSI:
    """Tests for compute_rsi()."""

    def test_returns_series(self, price_series: pd.Series):
        rsi = compute_rsi(price_series)
        assert isinstance(rsi, pd.Series)

    def test_rsi_range(self, price_series: pd.Series):
        rsi = compute_rsi(price_series).dropna()
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0

    def test_rsi_length(self, price_series: pd.Series):
        rsi = compute_rsi(price_series)
        assert len(rsi) == len(price_series)

    def test_rsi_default_period(self, price_series: pd.Series):
        rsi = compute_rsi(price_series, period=14)
        assert rsi.dropna().shape[0] > 0

    def test_rsi_custom_period(self, price_series: pd.Series):
        rsi_7 = compute_rsi(price_series, period=7)
        rsi_21 = compute_rsi(price_series, period=21)
        # Shorter period RSI should have more valid values
        assert rsi_7.dropna().shape[0] >= rsi_21.dropna().shape[0]

    def test_mostly_up_gives_high_rsi(self):
        """Mostly rising prices should give RSI well above 50."""
        np.random.seed(99)
        # Strongly uptrending with realistic noise (needs both up/down bars for RSI)
        prices = pd.Series(100 + np.arange(200) * 0.3 + np.random.randn(200) * 0.5)
        rsi = compute_rsi(prices, period=14).dropna()
        assert len(rsi) > 0
        assert rsi.iloc[-1] > 60

    def test_mostly_down_gives_low_rsi(self):
        """Mostly falling prices should give RSI well below 50."""
        np.random.seed(99)
        prices = pd.Series(200 - np.arange(100) * 0.5 + np.random.randn(100) * 0.05)
        rsi = compute_rsi(prices, period=14).dropna()
        assert len(rsi) > 0
        assert rsi.iloc[-1] < 30


# ============================================================================
# Bollinger Bands Tests
# ============================================================================

class TestComputeBollingerBands:
    """Tests for compute_bollinger_bands()."""

    def test_returns_three_series(self, price_series: pd.Series):
        middle, upper, lower = compute_bollinger_bands(price_series)
        assert isinstance(middle, pd.Series)
        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_upper_above_middle(self, price_series: pd.Series):
        middle, upper, lower = compute_bollinger_bands(price_series)
        valid = ~(middle.isna() | upper.isna())
        assert (upper[valid] >= middle[valid]).all()

    def test_lower_below_middle(self, price_series: pd.Series):
        middle, upper, lower = compute_bollinger_bands(price_series)
        valid = ~(middle.isna() | lower.isna())
        assert (lower[valid] <= middle[valid]).all()

    def test_band_width_increases_with_std(self, price_series: pd.Series):
        _, upper1, lower1 = compute_bollinger_bands(price_series, num_std=1.0)
        _, upper2, lower2 = compute_bollinger_bands(price_series, num_std=3.0)
        valid = ~(upper1.isna() | upper2.isna())
        width1 = (upper1 - lower1)[valid]
        width2 = (upper2 - lower2)[valid]
        assert (width2 >= width1).all()


# ============================================================================
# add_indicators Tests
# ============================================================================

class TestAddIndicators:
    """Tests for add_indicators()."""

    def test_adds_columns(self, ohlcv_df: pd.DataFrame):
        result = add_indicators(ohlcv_df.copy())
        assert "RSI" in result.columns
        assert "BB_Middle" in result.columns
        assert "BB_Upper" in result.columns
        assert "BB_Lower" in result.columns

    def test_returns_same_df(self, ohlcv_df: pd.DataFrame):
        df_copy = ohlcv_df.copy()
        result = add_indicators(df_copy)
        assert result is df_copy  # Modifies in-place
