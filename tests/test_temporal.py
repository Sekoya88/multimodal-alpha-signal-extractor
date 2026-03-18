"""Tests for Temporal Multi-Frame Extension (Sprint 4).

Tests cover:
1. Frame window generation
2. Position embeddings
3. Multi-image message format
4. Config validation
5. Port compliance
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from alpha_signal.application.ports import TemporalSignalPort
from alpha_signal.domain.temporal_signal_extractor import (
    TemporalPositionEmbedding,
    TemporalSignalExtractor,
    build_temporal_messages,
    generate_frame_windows,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with enough rows."""
    np.random.seed(42)
    n_rows = 200
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n_rows) * 0.5,
            "High": close + abs(np.random.randn(n_rows) * 1.5),
            "Low": close - abs(np.random.randn(n_rows) * 1.5),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n_rows),
        },
        index=dates,
    )


# ============================================================================
# Unit: generate_frame_windows
# ============================================================================


class TestGenerateFrameWindows:
    """Tests for sliding window frame generation."""

    def test_correct_number_of_frames(self, sample_df):
        frames = generate_frame_windows(sample_df, n_frames=8, window_size=60, stride=5)
        assert len(frames) == 8

    def test_each_frame_correct_size(self, sample_df):
        frames = generate_frame_windows(sample_df, n_frames=4, window_size=60, stride=5)
        for frame in frames:
            assert len(frame) == 60

    def test_frames_are_consecutive(self, sample_df):
        """Each frame should start `stride` rows after the previous."""
        frames = generate_frame_windows(sample_df, n_frames=4, window_size=60, stride=5)
        for i in range(1, len(frames)):
            prev_start = sample_df.index.get_loc(frames[i - 1].index[0])
            curr_start = sample_df.index.get_loc(frames[i].index[0])
            assert curr_start - prev_start == 5

    def test_insufficient_data_raises(self, sample_df):
        """Should raise ValueError when not enough data for all frames."""
        short_df = sample_df.iloc[:50]
        with pytest.raises(ValueError, match="needs"):
            generate_frame_windows(short_df, n_frames=8, window_size=60, stride=5)

    def test_single_frame(self, sample_df):
        frames = generate_frame_windows(sample_df, n_frames=1, window_size=60, stride=5)
        assert len(frames) == 1
        assert len(frames[0]) == 60

    def test_different_strides(self, sample_df):
        """Different strides should produce different starting points."""
        frames_s5 = generate_frame_windows(sample_df, n_frames=3, window_size=60, stride=5)
        frames_s10 = generate_frame_windows(sample_df, n_frames=3, window_size=60, stride=10)
        # Third frame should start at different positions
        assert frames_s5[2].index[0] != frames_s10[2].index[0]


# ============================================================================
# Unit: TemporalPositionEmbedding
# ============================================================================


class TestTemporalPositionEmbedding:
    """Tests for learned temporal position embeddings."""

    def test_output_shape(self):
        """Position embeddings should have correct shape."""
        emb = TemporalPositionEmbedding(max_frames=16, embedding_dim=64)
        indices = torch.arange(8)
        out = emb(indices)
        assert out.shape == (8, 64)

    def test_different_positions_different_embeddings(self):
        """Different frame positions should get different embeddings."""
        emb = TemporalPositionEmbedding(max_frames=16, embedding_dim=64)
        indices = torch.arange(4)
        out = emb(indices)
        # Each position should be unique
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(out[i], out[j], atol=1e-5)

    def test_weights_sum_to_one(self):
        """Position weights should be softmax-normalized (sum ≈ 1)."""
        emb = TemporalPositionEmbedding(max_frames=16, embedding_dim=64)
        weights = emb.get_position_weights(8)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_single_frame_weight(self):
        """Single frame should get weight 1.0."""
        emb = TemporalPositionEmbedding(max_frames=16, embedding_dim=64)
        weights = emb.get_position_weights(1)
        assert torch.allclose(weights, torch.tensor([1.0]), atol=1e-5)


# ============================================================================
# Unit: build_temporal_messages
# ============================================================================


class TestBuildTemporalMessages:
    """Tests for multi-image message builder."""

    def test_message_structure(self, tmp_path):
        """Messages should have system + user roles."""
        paths = [tmp_path / f"frame_{i}.png" for i in range(4)]
        for p in paths:
            p.touch()

        msgs = build_temporal_messages(paths)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_correct_number_of_images(self, tmp_path):
        """User message should contain N image blocks."""
        paths = [tmp_path / f"frame_{i}.png" for i in range(4)]
        for p in paths:
            p.touch()

        msgs = build_temporal_messages(paths)
        user_content = msgs[1]["content"]
        image_blocks = [c for c in user_content if c.get("type") == "image"]
        assert len(image_blocks) == 4

    def test_labels_present(self, tmp_path):
        """Each image should be followed by a text label."""
        paths = [tmp_path / f"frame_{i}.png" for i in range(3)]
        for p in paths:
            p.touch()

        msgs = build_temporal_messages(paths)
        user_content = msgs[1]["content"]
        text_blocks = [c for c in user_content if c.get("type") == "text"]
        # N labels + 1 final prompt = N+1 text blocks
        assert len(text_blocks) == 4  # 3 labels + 1 final prompt


# ============================================================================
# Unit: TemporalSignalExtractor
# ============================================================================


class TestTemporalSignalExtractor:
    """Tests for the domain service."""

    def test_parse_valid_json(self):
        raw = '{"action": "BUY", "confidence": 0.85, "trend": "UPTREND", "reasoning": "test"}'
        result = TemporalSignalExtractor.parse_temporal_signal(raw)
        assert result["action"] == "BUY"
        assert result["trend"] == "UPTREND"

    def test_parse_malformed(self):
        result = TemporalSignalExtractor.parse_temporal_signal("random text")
        assert result["action"] == "HOLD"
        assert result["trend"] == "SIDEWAYS"

    def test_temporal_weights_correct_length(self):
        extractor = TemporalSignalExtractor(position_embedding_dim=64)
        weights = extractor.get_temporal_weights(8)
        assert len(weights) == 8

    def test_temporal_weights_sum_to_one(self):
        extractor = TemporalSignalExtractor(position_embedding_dim=64)
        weights = extractor.get_temporal_weights(8)
        assert abs(sum(weights) - 1.0) < 1e-5


# ============================================================================
# Config validation
# ============================================================================


class TestTemporalConfig:
    """Tests for TemporalConfig dataclass."""

    def test_default_values(self):
        from config import temporal_cfg
        assert temporal_cfg.n_frames == 8
        assert temporal_cfg.window_size == 60
        assert temporal_cfg.stride == 5
        assert temporal_cfg.position_embedding_dim == 64

    def test_total_coverage(self):
        """Total temporal coverage = window_size + (n_frames - 1) * stride."""
        from config import temporal_cfg
        coverage = temporal_cfg.window_size + (temporal_cfg.n_frames - 1) * temporal_cfg.stride
        # 60 + 7 * 5 = 95 trading days
        assert coverage == 95
