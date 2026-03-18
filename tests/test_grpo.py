"""Tests for GRPO training service (Sprint 3).

Tests cover:
1. Composite reward computation
2. Reward normalization
3. PPO clipping
4. Service port compliance
5. Config validation
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpha_signal.application.grpo_training_service import (
    GRPOTrainingService,
    calibration_error,
    composite_reward,
    directional_accuracy,
    normalize_rewards,
    ppo_clip_ratio,
)
from alpha_signal.application.ports import GRPOTrainingPort


# ============================================================================
# Unit: directional_accuracy
# ============================================================================


class TestDirectionalAccuracy:
    """Tests for directional_accuracy."""

    def test_correct_buy(self):
        assert directional_accuracy("BUY", "BUY") == 1.0

    def test_correct_sell(self):
        assert directional_accuracy("SELL", "SELL") == 1.0

    def test_incorrect(self):
        assert directional_accuracy("BUY", "SELL") == 0.0

    def test_case_insensitive(self):
        assert directional_accuracy("buy", "BUY") == 1.0


# ============================================================================
# Unit: calibration_error
# ============================================================================


class TestCalibrationError:
    """Tests for calibration_error."""

    def test_perfect_calibration(self):
        assert calibration_error(0.8, 0.8) == 1.0

    def test_overconfident(self):
        result = calibration_error(0.9, 0.5)
        assert result == pytest.approx(0.6, abs=0.01)

    def test_underconfident(self):
        result = calibration_error(0.3, 0.7)
        assert result == pytest.approx(0.6, abs=0.01)


# ============================================================================
# Unit: composite_reward
# ============================================================================


class TestCompositeReward:
    """Tests for composite_reward."""

    def test_correct_and_calibrated(self):
        """Correct prediction with good calibration should score high."""
        r = composite_reward("BUY", 0.8, "BUY", 0.8, w_direction=0.6, w_calibration=0.4)
        # 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert r == pytest.approx(1.0)

    def test_incorrect_but_calibrated(self):
        """Wrong prediction should reduce reward."""
        r = composite_reward("SELL", 0.8, "BUY", 0.8, w_direction=0.6, w_calibration=0.4)
        # 0.6 * 0.0 + 0.4 * 1.0 = 0.4
        assert r == pytest.approx(0.4)

    def test_correct_but_miscalibrated(self):
        """Correct but overconfident should partially reduce reward."""
        r = composite_reward("BUY", 0.95, "BUY", 0.5, w_direction=0.6, w_calibration=0.4)
        # 0.6 * 1.0 + 0.4 * (1 - |0.95 - 0.5|) = 0.6 + 0.4 * 0.55 = 0.82
        assert r == pytest.approx(0.82, abs=0.01)

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        from config import grpo_cfg
        total = grpo_cfg.reward_weight_direction + grpo_cfg.reward_weight_calibration
        assert total == pytest.approx(1.0)


# ============================================================================
# Unit: normalize_rewards
# ============================================================================


class TestNormalizeRewards:
    """Tests for normalize_rewards."""

    def test_normalized_mean_zero(self):
        """Normalized rewards should have mean ≈ 0."""
        rewards = [0.2, 0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4]
        normalized = normalize_rewards(rewards)
        mean = sum(normalized) / len(normalized)
        assert abs(mean) < 1e-6

    def test_normalized_unit_std(self):
        """Normalized rewards should have std ≈ 1."""
        import numpy as np
        rewards = [0.2, 0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4]
        normalized = normalize_rewards(rewards)
        std = float(np.std(normalized))
        assert abs(std - 1.0) < 0.1

    def test_single_reward_returns_zero(self):
        """Single reward should normalize to [0]."""
        assert normalize_rewards([0.5]) == [0.0]

    def test_all_same_returns_zeros(self):
        """All same rewards should normalize to zeros."""
        assert normalize_rewards([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]

    def test_empty_returns_empty(self):
        assert normalize_rewards([]) == []


# ============================================================================
# Unit: ppo_clip_ratio
# ============================================================================


class TestPPOClipRatio:
    """Tests for PPO clipping."""

    def test_no_change_in_policy(self):
        """When log probs are equal, ratio=1, no clipping needed."""
        result = ppo_clip_ratio(
            new_log_prob=-1.0, old_log_prob=-1.0, advantage=0.5, epsilon=0.2,
        )
        assert result == pytest.approx(0.5, abs=0.01)

    def test_positive_advantage_clips_high_ratio(self):
        """With positive advantage, high ratio should be clipped."""
        # ratio = exp(0.5) ≈ 1.65, clipped to 1.2 with epsilon=0.2
        result = ppo_clip_ratio(
            new_log_prob=-0.5, old_log_prob=-1.0, advantage=1.0, epsilon=0.2,
        )
        ratio = math.exp(0.5)
        clipped_val = min(ratio * 1.0, 1.2 * 1.0)
        assert result == pytest.approx(clipped_val, abs=0.01)

    def test_negative_advantage_clips_low_ratio(self):
        """With negative advantage, low ratio should limit penalty."""
        result = ppo_clip_ratio(
            new_log_prob=-2.0, old_log_prob=-1.0, advantage=-1.0, epsilon=0.2,
        )
        ratio = math.exp(-1.0)  # ≈ 0.37
        clipped = max(min(ratio, 1.2), 0.8)  # clipped to 0.8
        # min(0.37 * -1, 0.8 * -1) = min(-0.37, -0.8) = -0.8
        assert result == pytest.approx(min(ratio * -1.0, clipped * -1.0), abs=0.01)


# ============================================================================
# Service: GRPOTrainingService
# ============================================================================


class TestGRPOTrainingService:
    """Tests for GRPOTrainingService."""

    def test_implements_port(self):
        service = GRPOTrainingService()
        assert isinstance(service, GRPOTrainingPort)

    def test_compute_rewards_returns_correct_length(self):
        """compute_rewards should return N rewards for N predictions."""
        service = GRPOTrainingService()
        preds = [
            {"action": "BUY", "confidence": 0.8},
            {"action": "SELL", "confidence": 0.6},
            {"action": "BUY", "confidence": 0.9},
            {"action": "HOLD", "confidence": 0.5},
        ]
        rewards = service.compute_rewards(preds, "BUY", oracle_return=0.03)
        assert len(rewards) == 4

    def test_compute_rewards_normalized(self):
        """Rewards should be approximately normalized (mean ≈ 0)."""
        service = GRPOTrainingService()
        preds = [
            {"action": "BUY", "confidence": 0.8},
            {"action": "SELL", "confidence": 0.6},
            {"action": "BUY", "confidence": 0.7},
            {"action": "HOLD", "confidence": 0.5},
            {"action": "BUY", "confidence": 0.9},
            {"action": "SELL", "confidence": 0.4},
            {"action": "HOLD", "confidence": 0.3},
            {"action": "BUY", "confidence": 0.85},
        ]
        rewards = service.compute_rewards(preds, "BUY", oracle_return=0.03)
        mean = sum(rewards) / len(rewards)
        assert abs(mean) < 1e-6

    def test_train_raises_on_missing_dataset(self):
        """Training should raise FileNotFoundError for missing data."""
        service = GRPOTrainingService()
        with pytest.raises((FileNotFoundError, RuntimeError)):
            service.train(
                dataset_path=Path("/nonexistent/data.jsonl"),
                output_dir=Path("/tmp/grpo_test"),
            )

    def test_parse_prediction_valid_json(self):
        raw = '{"action": "BUY", "confidence": 0.85}'
        result = GRPOTrainingService._parse_prediction(raw)
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.85

    def test_parse_prediction_malformed(self):
        result = GRPOTrainingService._parse_prediction("random text")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.5
