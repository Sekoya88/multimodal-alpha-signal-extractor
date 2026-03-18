"""Tests for DPO alignment service and utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpha_signal.application.dpo_alignment_service import (
    DPOAlignmentService,
    _parse_signal_json,
    _synthetic_rejected,
)


# ============================================================================
# Unit: _parse_signal_json
# ============================================================================


class TestParseSignalJson:
    """Tests for _parse_signal_json."""

    def test_plain_json(self):
        raw = '{"action": "BUY", "confidence": 0.85}'
        out = _parse_signal_json(raw)
        assert out["action"] == "BUY"
        assert out["confidence"] == 0.85

    def test_json_in_markdown(self):
        raw = 'Some text ```json\n{"action": "SELL", "confidence": 0.7}\n```'
        out = _parse_signal_json(raw)
        assert out["action"] == "SELL"
        assert out["confidence"] == 0.7

    def test_malformed_returns_hold(self):
        out = _parse_signal_json("not valid json at all")
        assert out["action"] == "HOLD"
        assert out["confidence"] == 0.5


# ============================================================================
# Unit: _synthetic_rejected
# ============================================================================


class TestSyntheticRejected:
    """Tests for _synthetic_rejected."""

    def test_buy_flips_to_sell(self):
        oracle = {
            "action": "BUY",
            "confidence": 0.9,
            "entry_price": 180.0,
            "stop_loss": 175.0,
            "take_profit": 190.0,
            "reasoning": "RSI oversold.",
        }
        rejected = _synthetic_rejected(oracle, "BUY")
        obj = json.loads(rejected)
        assert obj["action"] == "SELL"

    def test_sell_flips_to_buy(self):
        oracle = {"action": "SELL", "confidence": 0.8}
        rejected = _synthetic_rejected(oracle, "SELL")
        obj = json.loads(rejected)
        assert obj["action"] == "BUY"

    def test_hold_flips_to_buy(self):
        oracle = {"action": "HOLD", "confidence": 0.6}
        rejected = _synthetic_rejected(oracle, "HOLD")
        obj = json.loads(rejected)
        assert obj["action"] == "BUY"


# ============================================================================
# Unit: DPOAlignmentService
# ============================================================================


class TestDPOAlignmentService:
    """Tests for DPOAlignmentService."""

    def test_service_implements_port(self):
        from alpha_signal.application.ports import DPOAlignmentPort

        service = DPOAlignmentService()
        assert isinstance(service, DPOAlignmentPort)

    def test_build_preference_pairs_raises_on_missing_file(self):
        service = DPOAlignmentService()
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            service.build_preference_pairs(
                jsonl_path="/nonexistent/training_data.jsonl",
                model=MagicMock(),
                processor=MagicMock(),
                device="cpu",
                max_samples=1,
            )


# ============================================================================
# Integration: DPO pairs (no GPU required)
# ============================================================================


def test_build_pairs_empty_file_returns_empty(tmp_path: Path):
    """Empty or no-image samples yield empty pairs."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    service = DPOAlignmentService()
    mock_model = MagicMock()
    mock_processor = MagicMock()
    # With 0 samples, loop never runs inference
    pairs = service.build_preference_pairs(
        jsonl_path=empty,
        model=mock_model,
        processor=mock_processor,
        device="cpu",
        max_samples=10,
    )
    assert pairs == []


def test_build_pairs_skips_samples_without_image(tmp_path: Path):
    """Samples with no image block are skipped."""
    no_img = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are an analyst."}]},
            {"role": "user", "content": [{"type": "text", "text": "Analyze"}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"action":"HOLD"}'}]},
        ]
    }
    path = tmp_path / "noimg.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(no_img, ensure_ascii=False) + "\n")
    service = DPOAlignmentService()
    pairs = service.build_preference_pairs(
        jsonl_path=path,
        model=MagicMock(),
        processor=MagicMock(),
        device="cpu",
    )
    assert pairs == []
