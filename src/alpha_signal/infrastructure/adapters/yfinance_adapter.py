"""yfinance_adapter.py — Market Data and News implementation using Yahoo Finance."""

import re

import numpy as np
import pandas as pd
import yfinance as yf

from alpha_signal.application.ports import MarketDataPort, NewsPort
from alpha_signal.infrastructure.logger import logger
from alpha_signal.domain.indicators import compute_rsi, compute_bollinger_bands


class YFinanceAdapter(MarketDataPort, NewsPort):
    """Adapter for fetching historical data and news via Yahoo Finance."""

    def fetch_data(self, ticker: str, days: int) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance and compute technical indicators."""
        logger.info(f"📡 Fetching real market data for {ticker}...")

        # Fetch ~2x the days to have enough after warmup
        tk = yf.Ticker(ticker)
        df = tk.history(period=f"{days * 2}d", interval="1d")

        if df.empty:
            logger.error(f"No data returned for {ticker}.")
            raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")

        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "Date"

        df["RSI"] = compute_rsi(df["Close"], period=14)
        bb_mid, bb_up, bb_low = compute_bollinger_bands(df["Close"])
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
        return df

    def fetch_news(self, ticker: str, max_articles: int = 5) -> str:
        """Fetch financial news and format them into a single context string."""
        articles = self.fetch_news_articles(ticker, max_articles)
        if not articles:
            return f"No recent news for {ticker}."
            
        lines = []
        for a in articles:
            entry = f"[{a['publisher']}] {a['title']}" if a["publisher"] else a["title"]
            if a["summary"]:
                entry += f" — {a['summary']}"
            lines.append(f"• {entry}")
            
        return "\n".join(lines)

    def fetch_news_articles(self, ticker: str, max_articles: int = 5) -> list[dict[str, str]]:
        """Fetch structured news articles."""
        logger.info(f"📰 Fetching real news for {ticker}...")
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
            try:
                content = article.get("content")
                if content is None:
                    content = article

                if not isinstance(content, dict):
                    continue

                title = content.get("title", "Untitled")

                provider = content.get("provider") or {}
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
                
                summary = content.get("summary") or ""
                if summary:
                    summary_clean = re.sub(r"<[^>]+>", "", summary)
                    if len(summary_clean) > 200:
                        summary_clean = summary_clean[:200] + "..."
                    summary = summary_clean

                url = ""
                click_through = content.get("clickThroughUrl")
                if isinstance(click_through, dict):
                    url = click_through.get("url", "")

                articles.append({
                    "title": title,
                    "publisher": publisher,
                    "summary": summary,
                    "url": url
                })
            except Exception as e:
                logger.error(f"❌ Erreur parsing article: {e}")
                continue

        logger.info(f"  ✓ Retrieved {len(articles)} articles")
        return articles

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI using Wilder's smoothing."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_bollinger(
        self, series: pd.Series, period: int = 20, std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands (mid, upper, lower)."""
        mid = series.rolling(window=period).mean()
        s = series.rolling(window=period).std()
        return mid, mid + std * s, mid - std * s
