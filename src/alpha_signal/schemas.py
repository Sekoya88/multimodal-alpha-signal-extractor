"""schemas.py — Pydantic models for structured pipeline outputs.

Extracted from 04_langchain_pipeline.py so they can be reused
across the pipeline, Streamlit app, and tests.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    """Allowed trading actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal(BaseModel):
    """Structured trading signal output from the VLM.

    This schema is enforced via LangChain's PydanticOutputParser
    to guarantee the VLM responds with valid, parseable JSON.
    """
    action: TradeAction = Field(
        description="Trading action: BUY, SELL, or HOLD"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence score between 0.0 and 1.0"
    )
    entry_price: float = Field(
        gt=0.0,
        description="Suggested entry price for the trade"
    )
    stop_loss: float = Field(
        gt=0.0,
        description="Stop-loss price level for risk management"
    )
    take_profit: float = Field(
        gt=0.0,
        description="Take-profit price target"
    )
    reasoning: str = Field(
        min_length=10,
        description="Brief technical reasoning for the signal"
    )


class SentimentResult(BaseModel):
    """Structured sentiment analysis output from the text LLM."""
    sentiment: str = Field(
        description="Market sentiment: BULLISH, BEARISH, or NEUTRAL"
    )
    intensity: float = Field(
        ge=0.0, le=1.0,
        description="Sentiment intensity score (0.0 = weak, 1.0 = strong)"
    )
    key_factors: list[str] = Field(
        description="List of key factors driving the sentiment"
    )
    summary: str = Field(
        description="One-sentence summary of the sentiment analysis"
    )


class TradingDecision(BaseModel):
    """Final merged trading decision combining VLM and sentiment signals."""
    vlm_signal: TradingSignal = Field(
        description="Trading signal from the Vision-Language Model"
    )
    sentiment: SentimentResult = Field(
        description="Sentiment analysis from the text LLM"
    )
    final_action: TradeAction = Field(
        description="Final recommended action after merging both signals"
    )
    final_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Merged confidence score"
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (timestamps, model versions, etc.)"
    )
