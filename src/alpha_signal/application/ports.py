"""ports.py — Interfaces/Ports for the Application layer.

Defines the contract that infrastructure adapters must fulfill.
Keeps the application core independent of external tools like Yahoo Finance, Ollama, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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


class DPOAlignmentPort(ABC):
    """Port for DPO (Direct Preference Optimization) alignment of the VLM."""

    @abstractmethod
    def build_preference_pairs(
        self,
        jsonl_path: str | Path,
        model,
        processor,
        device: str = "cuda",
        max_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build chosen/rejected pairs from training data using model inference as reward oracle.

        Args:
            jsonl_path: Path to training_data.jsonl.
            model: Loaded VLM for inference.
            processor: Model processor (tokenizer + image processor).
            device: Device for inference.
            max_samples: Optional cap on number of samples (for faster iteration).

        Returns:
            List of dicts with keys: images, prompt, chosen, rejected.
        """
        pass

    @abstractmethod
    def train(
        self,
        pairs: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> dict[str, float]:
        """Run DPO training and return metrics (loss, calibration improvement).

        Args:
            pairs: Chosen/rejected pairs from build_preference_pairs.
            output_dir: Where to save the DPO adapter.

        Returns:
            Dict with train_loss, calibration_improvement, etc.
        """
        pass


class RewardScorerPort(ABC):
    """Port for visual reward scoring (frozen VLM backbone + MLP head → [0,1])."""

    @abstractmethod
    def score(
        self,
        image_path: Path,
        predicted_action: str,
        predicted_confidence: float,
    ) -> float:
        """Score a (chart, prediction) pair. Returns reward in [0, 1].

        Args:
            image_path: Path to the candlestick chart image.
            predicted_action: Predicted trading action (BUY/SELL/HOLD).
            predicted_confidence: Model's confidence in the prediction.

        Returns:
            Reward score between 0.0 and 1.0.
        """
        pass

    @abstractmethod
    def train(
        self,
        data_path: Path,
    ) -> dict[str, float]:
        """Train the reward model MLP head on labeled data.

        Args:
            data_path: Path to reward training data JSONL.

        Returns:
            Dict with training metrics (loss, accuracy, etc.).
        """
        pass

    @abstractmethod
    def load_weights(self, weights_path: Path) -> None:
        """Load previously trained MLP head weights.

        Args:
            weights_path: Path to saved MLP weights file.
        """
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


class GRPOTrainingPort(ABC):
    """Port for GRPO (Group Relative Policy Optimization) training."""

    @abstractmethod
    def generate_group(
        self,
        image_path: Path,
        prompt: str,
        n: int = 8,
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Generate N diverse predictions for a single chart via temperature sampling.

        Args:
            image_path: Path to chart image.
            prompt: User prompt for the VLM.
            n: Number of predictions to generate.
            temperature: Sampling temperature for diversity.

        Returns:
            List of N prediction dicts with keys: action, confidence, full_text.
        """
        pass

    @abstractmethod
    def compute_rewards(
        self,
        predictions: list[dict[str, Any]],
        oracle_action: str,
        oracle_return: float,
    ) -> list[float]:
        """Compute composite rewards for a group of predictions.

        Reward = 0.6 * directional_accuracy + 0.4 * (1 - |conf_error|).
        Then normalize within group (subtract mean, divide by std).

        Args:
            predictions: Group of N predictions.
            oracle_action: Ground truth action from forward return.
            oracle_return: Actual forward return value.

        Returns:
            List of N normalized reward values.
        """
        pass

    @abstractmethod
    def train(
        self,
        dataset_path: Path,
        output_dir: Path,
    ) -> dict[str, float]:
        """Run the full GRPO training loop.

        Args:
            dataset_path: Path to training_data.jsonl.
            output_dir: Where to save the GRPO adapter.

        Returns:
            Dict with training metrics (avg_reward, loss, etc.).
        """
        pass


class TemporalSignalPort(ABC):
    """Port for temporal multi-frame chart analysis."""

    @abstractmethod
    def generate_temporal_sequence(
        self,
        df: "pd.DataFrame",
        n_frames: int = 8,
        window_size: int = 60,
        stride: int = 5,
    ) -> list[Path]:
        """Generate a sequence of N consecutive chart images from market data.

        Args:
            df: Full OHLCV DataFrame with indicators.
            n_frames: Number of frames to generate.
            window_size: Trading days per chart.
            stride: Days between consecutive frames.

        Returns:
            List of Paths to generated chart images.
        """
        pass

    @abstractmethod
    def analyze_temporal_sequence(
        self,
        image_paths: list[Path],
    ) -> dict[str, Any]:
        """Analyze a sequence of charts for temporal trend identification.

        Uses Qwen2.5-VL multi-image input to process all frames together.

        Args:
            image_paths: Ordered list of chart image paths (temporal sequence).

        Returns:
            Dict with keys: action, confidence, trend, reasoning.
        """
        pass

    @abstractmethod
    def benchmark(
        self,
        dataset_path: Path,
        n_frames: int = 8,
    ) -> dict[str, float]:
        """Benchmark single-frame vs N-frame accuracy on a holdout set.

        Args:
            dataset_path: Path to training_data.jsonl.
            n_frames: Number of frames for multi-frame analysis.

        Returns:
            Dict with single_frame_accuracy, multi_frame_accuracy, improvement.
        """
        pass


