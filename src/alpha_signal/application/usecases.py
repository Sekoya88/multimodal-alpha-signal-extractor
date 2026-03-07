"""usecases.py — Core orchestration logic for the pipeline.

This module houses the business logic to generate a trading decision by orchestrating
data fetchers, chart rendering, and LLM inference ports concurrently.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from alpha_signal.application.ports import (
    ChartRendererPort,
    MarketDataPort,
    NewsPort,
    SentimentPort,
    VlmPort,
)
from alpha_signal.domain.models import TradeAction, TradingDecision, TradingSignal, SentimentResult


class AnalyzeMarketUseCase:
    """Use case to fetch data, generate charts, and get an AI trading decision."""
    
    def __init__(
        self,
        market_data_port: MarketDataPort,
        news_port: NewsPort,
        chart_renderer_port: ChartRendererPort,
        vlm_port: VlmPort,
        sentiment_port: SentimentPort,
        output_dir: Path,
    ):
        self._market_data = market_data_port
        self._news = news_port
        self._renderer = chart_renderer_port
        self._vlm = vlm_port
        self._sentiment = sentiment_port
        self._output_dir = output_dir

    async def execute(self, ticker: str, days: int = 60) -> TradingDecision:
        """Execute the pipeline end-to-end for a given ticker.
        
        Args:
            ticker: The market asset symbol.
            days: Lookback window length in days.
            
        Returns:
            The final aggregated TradingDecision.
        """
        logger.info(f"Starting analysis use-case for {ticker} (window={days} days)")
        
        # 1. Fetch data synchronously (yfinance doesn't easily support async yet)
        df = self._market_data.fetch_data(ticker, days)
        news_text = self._news.fetch_news(ticker)
        
        # 2. Render the chart
        self._output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = self._renderer.render_chart(df, ticker, self._output_dir)
        
        logger.info("Market data gathered and chart rendered. Invoking LLMs concurrently.")

        # 3. Run LLM pipelines concurrently
        vlm_task = asyncio.create_task(self._vlm.analyze_chart(chart_path, news_text))
        sentiment_task = asyncio.create_task(self._sentiment.analyze_sentiment(news_text))
        
        vlm_signal, sentiment = await asyncio.gather(vlm_task, sentiment_task)
        
        # 4. Merge Signals (Domain logic applied in the usecase)
        from alpha_signal.domain.services import SignalMergerService
        decision = SignalMergerService.merge_signals(
            vlm_signal=vlm_signal,
            sentiment=sentiment,
            vlm_provider=self._vlm.provider_info,
            sentiment_provider=self._sentiment.provider_info,
        )
        
        logger.info(f"Analysis complete for {ticker}. Final decision: {decision.final_action.value}")
        return decision
