#!/usr/bin/env python3
"""
05_live_analysis.py — Live Market Analysis with Real Data.

This script fetches REAL financial data (not synthetic):
  1. Real-time AAPL chart from yfinance (current market data)
  2. Real financial news from yfinance ticker.news
  3. Generates a live chart with RSI + Bollinger Bands
  4. Runs the full LangChain pipeline with both real inputs

This is the "production" entry point — no mocked data.

Usage:
    python 05_live_analysis.py                         # Default: AAPL
    python 05_live_analysis.py --ticker MSFT            # Any ticker
    python 05_live_analysis.py --ticker NVDA --days 90  # Custom window

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf

from config import DATASET_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technical Indicators (same as 01_generate_dataset.py)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger(
    series: pd.Series, period: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (mid, upper, lower)."""
    mid = series.rolling(window=period).mean()
    s = series.rolling(window=period).std()
    return mid, mid + std * s, mid - std * s


# ---------------------------------------------------------------------------
# Real Data Fetching
# ---------------------------------------------------------------------------

def fetch_real_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """Download real market data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'MSFT').
        days: Number of trading days to fetch.

    Returns:
        DataFrame with OHLCV + RSI + Bollinger Bands.
    """
    logger.info(f"📡 Fetching real market data for {ticker}...")

    # Fetch ~2x the days to have enough after warmup
    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{days * 2}d", interval="1d")

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")

    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"

    # Compute indicators
    df["RSI"] = compute_rsi(df["Close"], period=14)
    bb_mid, bb_up, bb_low = compute_bollinger(df["Close"])
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_up
    df["BB_Lower"] = bb_low

    # Keep only the last `days` bars (after warmup)
    df = df.iloc[-days:].copy()
    df = df.dropna()

    logger.info(
        f"  ✓ {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} "
        f"to {df.index[-1].strftime('%Y-%m-%d')}"
    )
    logger.info(
        f"  Current: ${df['Close'].iloc[-1]:.2f} | "
        f"RSI: {df['RSI'].iloc[-1]:.1f} | "
        f"BB: [{df['BB_Lower'].iloc[-1]:.2f} — {df['BB_Upper'].iloc[-1]:.2f}]"
    )

    return df


def fetch_real_news(ticker: str, max_articles: int = 5) -> str:
    """Fetch real financial news from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.
        max_articles: Maximum number of articles to include.

    Returns:
        Concatenated news text (titles + summaries).
    """
    logger.info(f"📰 Fetching real news for {ticker}...")

    tk = yf.Ticker(ticker)

    try:
        news = tk.news
    except Exception as e:
        logger.warning(f"Could not fetch news: {e}")
        return f"Pas de news disponible pour {ticker} en ce moment."

    if not news:
        return f"Pas de news récente pour {ticker}."

    # Build news digest
    articles: list[str] = []
    for i, article in enumerate(news[:max_articles]):
        # yfinance nests data under 'content' key
        content = article.get("content", article)
        title = content.get("title", "Sans titre")

        # Provider can be nested
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)

        summary = content.get("summary", "")

        entry = f"[{publisher}] {title}" if publisher else title
        if summary:
            # Clean HTML tags from summary
            import re
            summary_clean = re.sub(r"<[^>]+>", "", summary)
            if len(summary_clean) > 200:
                summary_clean = summary_clean[:200] + "..."
            entry += f" — {summary_clean}"
        articles.append(entry)

    news_text = "\n".join(f"• {a}" for a in articles)
    logger.info(f"  ✓ {len(articles)} articles récupérés")
    for a in articles[:3]:
        logger.info(f"    {a[:100]}...")

    return news_text


# ---------------------------------------------------------------------------
# Chart Rendering
# ---------------------------------------------------------------------------

def render_live_chart(df: pd.DataFrame, ticker: str) -> Path:
    """Render a live candlestick chart with RSI and Bollinger Bands.

    Args:
        df: DataFrame with OHLCV + indicators.
        ticker: Ticker symbol for chart title.

    Returns:
        Path to the saved chart image.
    """
    logger.info("📊 Rendering live chart...")

    add_plots = [
        mpf.make_addplot(df["BB_Upper"], color="steelblue", linestyle="--", width=0.8),
        mpf.make_addplot(df["BB_Lower"], color="steelblue", linestyle="--", width=0.8),
        mpf.make_addplot(df["BB_Middle"], color="orange", linestyle="-", width=0.8),
        mpf.make_addplot(df["RSI"], panel=2, color="purple", ylabel="RSI", width=1.0),
        mpf.make_addplot(
            pd.Series(30.0, index=df.index), panel=2, color="green",
            linestyle="--", width=0.5,
        ),
        mpf.make_addplot(
            pd.Series(70.0, index=df.index), panel=2, color="red",
            linestyle="--", width=0.5,
        ),
    ]

    mc = mpf.make_marketcolors(
        up="limegreen", down="tomato", edge="inherit",
        wick="inherit", volume="steelblue",
    )
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="gray")

    chart_path = DATASET_DIR / f"live_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    title = (
        f"{ticker} | {df.index[0].strftime('%Y-%m-%d')} → "
        f"{df.index[-1].strftime('%Y-%m-%d')} | LIVE"
    )

    mpf.plot(
        df[["Open", "High", "Low", "Close", "Volume"]],
        type="candle",
        style=style,
        addplot=add_plots,
        volume=True,
        title=title,
        figsize=(14, 9),
        panel_ratios=(4, 1, 2),
        savefig=dict(fname=str(chart_path), dpi=120, bbox_inches="tight"),
    )

    logger.info(f"  ✓ Chart saved → {chart_path}")
    return chart_path


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def run_live_analysis(ticker: str, days: int) -> dict[str, Any]:
    """Run the complete live analysis pipeline.

    Args:
        ticker: Stock ticker symbol.
        days: Number of trading days for the chart.

    Returns:
        Trading decision as a dict.
    """
    logger.info("=" * 70)
    logger.info(f"  LIVE ANALYSIS — {ticker}")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # 1. Fetch real market data
    df = fetch_real_data(ticker, days)

    # 2. Fetch real news
    news_text = fetch_real_news(ticker)

    # 3. Render chart
    chart_path = render_live_chart(df, ticker)

    # 4. Run LangChain pipeline
    logger.info("🤖 Running LangChain pipeline...")
    from _04_langchain_pipeline_import import run_pipeline_sync

    decision = run_pipeline_sync(chart_path, news_text)

    # 5. Save results
    output_path = DATASET_DIR / f"decision_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    output_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"💾 Decision saved → {output_path}")

    return decision.model_dump()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run live analysis from CLI."""
    parser = argparse.ArgumentParser(
        description="Live Market Analysis with Real Data"
    )
    parser.add_argument(
        "--ticker", "-t", type=str, default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--days", "-d", type=int, default=60,
        help="Number of trading days for the chart (default: 60)",
    )
    parser.add_argument(
        "--news-only", action="store_true",
        help="Only fetch and display news, don't run full pipeline",
    )
    parser.add_argument(
        "--chart-only", action="store_true",
        help="Only generate chart, don't run full pipeline",
    )

    args = parser.parse_args()

    try:
        if args.news_only:
            news = fetch_real_news(args.ticker, max_articles=10)
            print(f"\n📰 News pour {args.ticker}:\n{news}")
            return

        if args.chart_only:
            df = fetch_real_data(args.ticker, args.days)
            chart_path = render_live_chart(df, args.ticker)
            print(f"\n📊 Chart saved → {chart_path}")
            return

        # Full pipeline
        from _04_pipeline_runner import run_full_pipeline
    except ImportError:
        pass

    # Direct execution with inline import
    try:
        df = fetch_real_data(args.ticker, args.days)
        news_text = fetch_real_news(args.ticker)
        chart_path = render_live_chart(df, args.ticker)

        # Import and run pipeline
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        pipeline_mod = import_module("04_langchain_pipeline")

        decision = pipeline_mod.run_pipeline_sync(chart_path, news_text)

        # Display result
        output_json = decision.model_dump_json(indent=2)
        print(f"\n{output_json}")

        # Save
        output_path = (
            DATASET_DIR
            / f"decision_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        output_path.write_text(output_json, encoding="utf-8")
        logger.info(f"💾 Decision saved → {output_path}")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
