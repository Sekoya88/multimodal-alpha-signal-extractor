"""data_fetcher.py — Real-time market data and news fetching.

Extracted from 05_live_analysis.py for reuse in the Streamlit app.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
import yfinance as yf

from .indicators import add_indicators

logger = logging.getLogger(__name__)


def fetch_market_data(
    ticker: str,
    days: int = 60,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """Download real market data from Yahoo Finance with indicators.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA').
        days: Number of trading days to return.
        rsi_period: RSI look-back period.
        bb_period: Bollinger Bands SMA window.
        bb_std: Bollinger Bands std multiplier.

    Returns:
        DataFrame with OHLCV + RSI + Bollinger Bands.

    Raises:
        ValueError: If no data returned for the ticker.
    """
    logger.info(f"Fetching real market data for {ticker}...")

    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{days * 2}d", interval="1d")

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")

    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"

    # Compute indicators
    add_indicators(df, rsi_period=rsi_period, bb_period=bb_period, bb_std=bb_std)

    # Keep only last N days (after warmup)
    df = df.iloc[-days:].copy()
    df = df.dropna()

    logger.info(
        f"  {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} "
        f"to {df.index[-1].strftime('%Y-%m-%d')}"
    )

    return df


def fetch_news(ticker: str, max_articles: int = 5) -> list[dict[str, str]]:
    """Fetch real financial news from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.
        max_articles: Maximum number of articles.

    Returns:
        List of dicts with 'title', 'publisher', 'summary' keys.
    """
    logger.info(f"Fetching real news for {ticker}...")

    tk = yf.Ticker(ticker)

    try:
        news = tk.news
    except Exception as e:
        logger.warning(f"Could not fetch news: {e}")
        return []

    if not news:
        return []

    articles: list[dict[str, str]] = []
    for article in news[:max_articles]:
        content = article.get("content", article)
        title = content.get("title", "Sans titre")

        provider = content.get("provider", {})
        publisher = (
            provider.get("displayName", "")
            if isinstance(provider, dict)
            else str(provider)
        )

        summary = content.get("summary", "")
        if summary:
            summary = re.sub(r"<[^>]+>", "", summary)
            if len(summary) > 200:
                summary = summary[:200] + "..."

        articles.append({
            "title": title,
            "publisher": publisher,
            "summary": summary,
        })

    logger.info(f"  {len(articles)} articles retrieved")
    return articles


def format_news_text(articles: list[dict[str, str]], ticker: str = "") -> str:
    """Format news articles into a single text string for LLM input.

    Args:
        articles: List of article dicts from fetch_news().
        ticker: Ticker for fallback message.

    Returns:
        Formatted news text string.
    """
    if not articles:
        return f"Pas de news récente disponible pour {ticker}."

    lines = []
    for a in articles:
        entry = f"[{a['publisher']}] {a['title']}" if a["publisher"] else a["title"]
        if a["summary"]:
            entry += f" — {a['summary']}"
        lines.append(f"• {entry}")

    return "\n".join(lines)
