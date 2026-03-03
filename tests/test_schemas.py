"""Tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alpha_signal.schemas import (
    SentimentResult,
    TradeAction,
    TradingDecision,
    TradingSignal,
)


# ============================================================================
# TradeAction Tests
# ============================================================================

class TestTradeAction:
    def test_valid_values(self):
        assert TradeAction.BUY.value == "BUY"
        assert TradeAction.SELL.value == "SELL"
        assert TradeAction.HOLD.value == "HOLD"


# ============================================================================
# TradingSignal Tests
# ============================================================================

class TestTradingSignal:
    def test_valid_signal(self):
        signal = TradingSignal(
            action=TradeAction.BUY,
            confidence=0.85,
            entry_price=180.0,
            stop_loss=175.0,
            take_profit=190.0,
            reasoning="RSI en zone de survente, divergence haussière détectée.",
        )
        assert signal.action == TradeAction.BUY
        assert signal.confidence == 0.85

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValidationError):
            TradingSignal(
                action="BUY", confidence=1.5, entry_price=180.0,
                stop_loss=175.0, take_profit=190.0,
                reasoning="Too confident signal",
            )

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValidationError):
            TradingSignal(
                action="BUY", confidence=-0.1, entry_price=180.0,
                stop_loss=175.0, take_profit=190.0,
                reasoning="Negative confidence signal",
            )

    def test_zero_entry_price_rejected(self):
        with pytest.raises(ValidationError):
            TradingSignal(
                action="BUY", confidence=0.5, entry_price=0,
                stop_loss=175.0, take_profit=190.0,
                reasoning="Zero entry price signal",
            )

    def test_short_reasoning_rejected(self):
        with pytest.raises(ValidationError):
            TradingSignal(
                action="BUY", confidence=0.5, entry_price=180.0,
                stop_loss=175.0, take_profit=190.0,
                reasoning="Short",  # min_length=10
            )

    def test_from_json_string(self):
        data = {
            "action": "SELL",
            "confidence": 0.7,
            "entry_price": 200.0,
            "stop_loss": 205.0,
            "take_profit": 190.0,
            "reasoning": "RSI en zone de surachat, pression vendeuse anticipée.",
        }
        signal = TradingSignal(**data)
        assert signal.action == TradeAction.SELL


# ============================================================================
# SentimentResult Tests
# ============================================================================

class TestSentimentResult:
    def test_valid_sentiment(self):
        result = SentimentResult(
            sentiment="BULLISH",
            intensity=0.8,
            key_factors=["earnings beat", "guidance raised"],
            summary="Strong quarterly performance drives bullish sentiment.",
        )
        assert result.sentiment == "BULLISH"
        assert len(result.key_factors) == 2


# ============================================================================
# TradingDecision Tests
# ============================================================================

class TestTradingDecision:
    def test_full_decision(self):
        vlm = TradingSignal(
            action="BUY", confidence=0.8, entry_price=180.0,
            stop_loss=175.0, take_profit=190.0,
            reasoning="Technical signals bullish, RSI rebounding from oversold.",
        )
        sent = SentimentResult(
            sentiment="BULLISH", intensity=0.8,
            key_factors=["earnings beat"],
            summary="Positive outlook.",
        )
        decision = TradingDecision(
            vlm_signal=vlm,
            sentiment=sent,
            final_action="BUY",
            final_confidence=0.82,
            meta={"platform": "Apple Silicon M4"},
        )
        assert decision.final_action == TradeAction.BUY
        assert decision.meta["platform"] == "Apple Silicon M4"

    def test_json_roundtrip(self):
        vlm = TradingSignal(
            action="HOLD", confidence=0.5, entry_price=180.0,
            stop_loss=175.0, take_profit=190.0,
            reasoning="Indicators neutral, no clear directional signal.",
        )
        sent = SentimentResult(
            sentiment="NEUTRAL", intensity=0.3,
            key_factors=["mixed signals"],
            summary="No strong directional bias.",
        )
        decision = TradingDecision(
            vlm_signal=vlm, sentiment=sent,
            final_action="HOLD", final_confidence=0.4,
        )
        json_str = decision.model_dump_json()
        restored = TradingDecision.model_validate_json(json_str)
        assert restored.final_action == decision.final_action
        assert restored.final_confidence == decision.final_confidence
