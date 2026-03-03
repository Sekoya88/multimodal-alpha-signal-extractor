"""Tests for the signal merger logic."""

from __future__ import annotations

import pytest

from alpha_signal.schemas import (
    SentimentResult,
    TradeAction,
    TradingSignal,
)
from alpha_signal.pipeline import merge_signals


# ============================================================================
# Fixtures
# ============================================================================

def _make_signal(action: str, confidence: float) -> TradingSignal:
    return TradingSignal(
        action=action,
        confidence=confidence,
        entry_price=180.0,
        stop_loss=175.0,
        take_profit=190.0,
        reasoning="Test signal reasoning for unit test.",
    )


def _make_sentiment(sentiment: str, intensity: float) -> SentimentResult:
    return SentimentResult(
        sentiment=sentiment,
        intensity=intensity,
        key_factors=["test factor"],
        summary="Test sentiment summary.",
    )


# ============================================================================
# Merger Tests
# ============================================================================

class TestMergeSignals:
    """Tests for merge_signals()."""

    def test_aligned_buy_boosts_confidence(self):
        """When both VLM says BUY and sentiment is BULLISH, confidence is boosted."""
        decision = merge_signals(
            _make_signal("BUY", 0.8),
            _make_sentiment("BULLISH", 0.9),
        )
        assert decision.final_action == TradeAction.BUY
        # 0.8 * 0.7 + 0.9 * 0.3 = 0.56 + 0.27 = 0.83
        assert decision.final_confidence == pytest.approx(0.83, abs=0.01)

    def test_aligned_sell_boosts_confidence(self):
        decision = merge_signals(
            _make_signal("SELL", 0.8),
            _make_sentiment("BEARISH", 0.9),
        )
        assert decision.final_action == TradeAction.SELL
        assert decision.final_confidence > 0.5

    def test_conflicting_signals_reduce_confidence(self):
        """VLM BUY + BEARISH sentiment should reduce confidence."""
        decision = merge_signals(
            _make_signal("BUY", 0.8),
            _make_sentiment("BEARISH", 0.9),
        )
        assert decision.final_action == TradeAction.BUY  # VLM takes priority
        assert decision.final_confidence == pytest.approx(0.4, abs=0.01)

    def test_hold_always_respected(self):
        """HOLD from VLM is respected regardless of sentiment."""
        decision = merge_signals(
            _make_signal("HOLD", 0.6),
            _make_sentiment("BULLISH", 0.95),
        )
        assert decision.final_action == TradeAction.HOLD
        assert decision.final_confidence == pytest.approx(0.48, abs=0.01)

    def test_confidence_capped_at_099(self):
        """Even with max-aligned signals, confidence should not exceed 0.99."""
        decision = merge_signals(
            _make_signal("BUY", 1.0),
            _make_sentiment("BULLISH", 1.0),
        )
        assert decision.final_confidence <= 0.99

    def test_meta_contains_required_fields(self):
        decision = merge_signals(
            _make_signal("BUY", 0.8),
            _make_sentiment("BULLISH", 0.8),
        )
        assert "timestamp" in decision.meta
        assert "vlm_provider" in decision.meta
        assert "signals_aligned" in decision.meta

    def test_signals_aligned_true(self):
        decision = merge_signals(
            _make_signal("BUY", 0.8),
            _make_sentiment("BULLISH", 0.8),
        )
        assert decision.meta["signals_aligned"] is True

    def test_signals_aligned_false(self):
        decision = merge_signals(
            _make_signal("BUY", 0.8),
            _make_sentiment("BEARISH", 0.8),
        )
        assert decision.meta["signals_aligned"] is False
