"""ports.py — Interfaces/Ports for the Application layer.

Defines the contract that infrastructure adapters must fulfill.
Keeps the application core independent of external tools like Yahoo Finance, Ollama, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from alpha_signal.domain.models import SentimentResult, TradingSignal


class MarketDataPort(ABC):
    """Port for fetching historical market data and technical indicators."""
    
    @abstractmethod
    def fetch_data(self, ticker: str, days: int) -> pd.DataFrame:
        """Fetch OHLCV + indicators for a given ticker and time window.
        
        Args:
            ticker: The market ticker symbol.
            days: The window length in days.
            
        Returns:
            A pandas DataFrame with OHLCV data + RSI + Bollinger Bands.
        """
        pass


class NewsPort(ABC):
    """Port for fetching financial news."""
    
    @abstractmethod
    def fetch_news(self, ticker: str, max_articles: int = 5) -> str:
        """Fetch news text relevant to the given ticker (formatted for LLM)."""
        pass

    @abstractmethod
    def fetch_news_articles(self, ticker: str, max_articles: int = 5) -> list[dict[str, str]]:
        """Fetch news articles as structured data (useful for UI)."""
        pass


class ChartRendererPort(ABC):
    """Port for rendering the market data into an image file."""
    
    @abstractmethod
    def render_chart(self, df: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
        """Renders the DataFrame into a candlestick chart.
        
        Args:
            df: The market data DataFrame.
            ticker: The market ticker symbol.
            output_dir: Directory to save the chart image.
            
        Returns:
            The path to the generated image file.
        """
        pass


class VlmPort(ABC):
    """Port for Vision-Language Model analysis."""
    
    @abstractmethod
    async def analyze_chart(
        self, image_path: Path, news_text: str
    ) -> TradingSignal:
        """Analyze a chart image and news text to generate a trading signal.
        
        Args:
            image_path: Path to the generated candlestick chart.
            news_text: Accompanying financial news.
            
        Returns:
            A structured TradingSignal value object.
        """
        pass
    
    @property
    @abstractmethod
    def provider_info(self) -> str:
        """Returns the identifier of the underlying VLM provider (e.g., 'ollama', 'vllm')."""
        pass


class SentimentPort(ABC):
    """Port for LLM-based sentiment analysis."""
    
    @abstractmethod
    async def analyze_sentiment(self, news_text: str) -> SentimentResult:
        """Analyze financial news to extract market sentiment.
        
        Args:
            news_text: The news text to analyze.
            
        Returns:
            A structured SentimentResult value object.
        """
        pass
    
    @property
    @abstractmethod
    def provider_info(self) -> str:
        """Returns the identifier of the underlying text LLM provider."""
        pass
