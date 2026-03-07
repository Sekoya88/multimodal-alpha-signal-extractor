"""services.py — Domain Services for complex business logic.

These services coordinate multiple entities or value objects but do not depend
on application ports or infrastructure.
"""

from datetime import datetime, timezone
from typing import Optional

from alpha_signal.domain.models import TradeAction, TradingDecision, TradingSignal, SentimentResult


class SignalMergerService:
    """Domain service to merge trading signals and sentiment."""
    
    @staticmethod
    def merge_signals(
        vlm_signal: TradingSignal,
        sentiment: SentimentResult,
        vlm_provider: str = "unknown",
        sentiment_provider: str = "unknown",
    ) -> TradingDecision:
        """Merge VLM trading signal with text sentiment into a final decision.
        
        Merging logic:
          - If both signals agree → boost confidence.
          - If signals conflict → reduce confidence, VLM takes priority.
          - HOLD from VLM is respected regardless of sentiment.
        """
        sentiment_direction = {
            "BULLISH": TradeAction.BUY,
            "BEARISH": TradeAction.SELL,
            "NEUTRAL": TradeAction.HOLD,
        }
        sentiment_bias = sentiment_direction.get(sentiment.sentiment, TradeAction.HOLD)

        if vlm_signal.action == TradeAction.HOLD:
            final_action = TradeAction.HOLD
            final_confidence = vlm_signal.confidence * 0.8
        elif vlm_signal.action == sentiment_bias:
            final_action = vlm_signal.action
            final_confidence = min(
                vlm_signal.confidence * 0.7 + sentiment.intensity * 0.3,
                0.99,
            )
        else:
            final_action = vlm_signal.action
            final_confidence = vlm_signal.confidence * 0.5

        return TradingDecision(
            vlm_signal=vlm_signal,
            sentiment=sentiment,
            final_action=final_action,
            final_confidence=round(final_confidence, 3),
            meta={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vlm_provider": vlm_provider,
                "sentiment_model": sentiment_provider,
                "signals_aligned": vlm_signal.action == sentiment_bias,
            },
        )
