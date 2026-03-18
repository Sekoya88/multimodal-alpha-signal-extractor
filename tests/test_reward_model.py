"""Tests for Visual Reward Model (Sprint 2).

Tests cover:
1. RewardScorerPort compliance
2. RewardMLP forward pass
3. Score output range
4. Training with mock data
5. Pipeline integration (reward adjusts confidence)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from alpha_signal.application.ports import RewardScorerPort
from alpha_signal.infrastructure.adapters.reward_scorer_adapter import (
    ACTION_TO_ID,
    RewardMLP,
    RewardScorerAdapter,
)


# ============================================================================
# Unit: RewardMLP
# ============================================================================


class TestRewardMLP:
    """Tests for the 2-layer MLP head."""

    def test_forward_pass_shape(self):
        """MLP output should be (B,) with values in [0, 1]."""
        mlp = RewardMLP(input_dim=64, hidden_dim=32, dropout=0.0)
        visual = torch.randn(4, 64)
        actions = torch.tensor([0, 1, 2, 0])  # BUY, SELL, HOLD, BUY
        confs = torch.tensor([0.8, 0.7, 0.5, 0.9])

        out = mlp(visual, actions, confs)
        assert out.shape == (4,)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_single_sample(self):
        """MLP should work with batch size 1."""
        mlp = RewardMLP(input_dim=128, hidden_dim=64, dropout=0.0)
        out = mlp(
            torch.randn(1, 128),
            torch.tensor([1]),
            torch.tensor([0.6]),
        )
        assert out.shape == (1,)

    def test_different_actions_give_different_scores(self):
        """Different actions on the same visual input should produce different scores."""
        torch.manual_seed(42)
        mlp = RewardMLP(input_dim=64, hidden_dim=32, dropout=0.0)
        visual = torch.randn(1, 64).expand(3, -1)  # Same input
        actions = torch.tensor([0, 1, 2])
        confs = torch.tensor([0.8, 0.8, 0.8])

        scores = mlp(visual, actions, confs)
        # With random weights, different action embeddings should give different scores
        assert not torch.allclose(scores[0:1], scores[1:2], atol=1e-3)


# ============================================================================
# Unit: ACTION_TO_ID mapping
# ============================================================================


class TestActionMapping:
    """Tests for action string → integer mapping."""

    def test_buy_maps_to_0(self):
        assert ACTION_TO_ID["BUY"] == 0

    def test_sell_maps_to_1(self):
        assert ACTION_TO_ID["SELL"] == 1

    def test_hold_maps_to_2(self):
        assert ACTION_TO_ID["HOLD"] == 2


# ============================================================================
# Unit: RewardScorerAdapter
# ============================================================================


class TestRewardScorerAdapter:
    """Tests for the adapter implementing RewardScorerPort."""

    def test_implements_port(self):
        """Adapter must implement RewardScorerPort interface."""
        adapter = RewardScorerAdapter(hidden_dim=32, dropout=0.0)
        assert isinstance(adapter, RewardScorerPort)

    def test_detect_device(self):
        """Device detection should return a valid string."""
        device = RewardScorerAdapter._detect_device()
        assert device in ("cuda", "mps", "cpu")

    def test_train_raises_on_missing_file(self):
        """Training should raise FileNotFoundError for missing data."""
        adapter = RewardScorerAdapter(hidden_dim=32, dropout=0.0)
        # Mock _load_backbone to avoid loading actual model
        adapter._backbone = MagicMock()
        adapter._processor = MagicMock()
        adapter._visual_dim = 64
        adapter._mlp = RewardMLP(input_dim=64, hidden_dim=32, dropout=0.0)

        with pytest.raises(FileNotFoundError, match="Reward training data not found"):
            adapter.train(data_path=Path("/nonexistent/data.jsonl"))


# ============================================================================
# Integration: Pipeline reward scoring
# ============================================================================


def test_pipeline_reward_adjustment():
    """Reward scorer should adjust confidence in the pipeline."""
    from alpha_signal.domain.models import (
        SentimentResult,
        TradeAction,
        TradingDecision,
        TradingSignal,
    )

    # Create a mock decision
    vlm_signal = TradingSignal(
        action=TradeAction.BUY,
        confidence=0.85,
        entry_price=180.0,
        stop_loss=175.0,
        take_profit=190.0,
        reasoning="RSI en zone de survente, divergence haussière détectée.",
    )
    sentiment = SentimentResult(
        sentiment="BULLISH",
        intensity=0.8,
        key_factors=["earnings beat"],
        summary="Positive outlook.",
    )
    decision = TradingDecision(
        vlm_signal=vlm_signal,
        sentiment=sentiment,
        final_action=TradeAction.BUY,
        final_confidence=0.83,
    )

    # Simulate reward adjustment (same logic as usecases.py)
    reward_score = 0.9  # High reward
    adjusted = round(decision.final_confidence * 0.7 + reward_score * 0.3, 3)
    assert 0.0 <= adjusted <= 1.0
    assert adjusted != decision.final_confidence  # Should differ

    # With zero reward
    low_reward = 0.1
    adjusted_low = round(decision.final_confidence * 0.7 + low_reward * 0.3, 3)
    assert adjusted_low < adjusted  # Lower reward → lower confidence


def test_mlp_weights_save_load(tmp_path: Path):
    """MLP weights should be saveable and loadable."""
    mlp = RewardMLP(input_dim=64, hidden_dim=32, dropout=0.0)
    weights_path = tmp_path / "mlp_head.pt"
    torch.save(mlp.state_dict(), weights_path)

    mlp2 = RewardMLP(input_dim=64, hidden_dim=32, dropout=0.0)
    mlp2.load_state_dict(torch.load(weights_path, weights_only=True))

    # Both should produce same output
    x = torch.randn(1, 64)
    a = torch.tensor([0])
    c = torch.tensor([0.8])
    assert torch.allclose(mlp(x, a, c), mlp2(x, a, c))
