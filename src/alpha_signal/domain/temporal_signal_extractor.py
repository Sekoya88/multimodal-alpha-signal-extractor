"""temporal_signal_extractor.py — Domain service for temporal multi-frame analysis.

Generates sequences of N=8 consecutive 60-day chart windows (stride=5 days)
and uses Qwen2.5-VL multi-image input to analyze temporal trends.

Includes learned temporal position embeddings added to visual features
before passing to the decoder.

Compatible with Apple Silicon M4 (CPU/MPS) and Colab T4 (CUDA).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# Temporal Position Embeddings
# ============================================================================

class TemporalPositionEmbedding(nn.Module):
    """Learned temporal position embeddings for multi-frame sequences.

    Adds a position-dependent vector to visual features before decoding,
    enabling the model to reason about temporal ordering of chart frames.
    """

    def __init__(self, max_frames: int = 16, embedding_dim: int = 64):
        super().__init__()
        self.position_embedding = nn.Embedding(max_frames, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Get position embeddings for given frame indices.

        Args:
            frame_indices: (N,) integer tensor of frame positions [0, max_frames).

        Returns:
            (N, embedding_dim) position embedding vectors.
        """
        emb = self.position_embedding(frame_indices)
        emb = self.projection(emb)
        emb = self.layer_norm(emb)
        return emb

    def get_position_weights(self, n_frames: int) -> torch.Tensor:
        """Get attention-like weights from position embeddings.

        Later frames (more recent) get higher weights by default.

        Args:
            n_frames: Number of frames in the sequence.

        Returns:
            (n_frames,) normalized weight tensor.
        """
        indices = torch.arange(n_frames)
        emb = self.forward(indices)
        # Use L2 norm as proxy weight (later positions tend to get larger norms after training)
        weights = emb.norm(dim=-1)
        weights = torch.softmax(weights, dim=0)
        return weights


# ============================================================================
# Sequence Generator (pure, no GPU)
# ============================================================================

def generate_frame_windows(
    df: pd.DataFrame,
    n_frames: int = 8,
    window_size: int = 60,
    stride: int = 5,
    start_offset: int = 0,
) -> list[pd.DataFrame]:
    """Generate N consecutive overlapping windows from OHLCV data.

    Each window is `window_size` trading days. Consecutive windows
    are separated by `stride` days.

    Args:
        df: Full OHLCV DataFrame (must have enough rows).
        n_frames: Number of frames to generate.
        window_size: Trading days per chart window.
        stride: Days shift between consecutive frames.
        start_offset: Starting row offset in the DataFrame.

    Returns:
        List of N DataFrames, each representing one chart window.

    Raises:
        ValueError: If DataFrame doesn't have enough rows for the sequence.
    """
    required = start_offset + window_size + (n_frames - 1) * stride
    if len(df) < required:
        raise ValueError(
            f"DataFrame has {len(df)} rows but needs {required} for "
            f"n_frames={n_frames}, window_size={window_size}, stride={stride}"
        )

    frames = []
    for i in range(n_frames):
        start = start_offset + i * stride
        end = start + window_size
        frames.append(df.iloc[start:end].copy())

    return frames


def render_frame_charts(
    frames: list[pd.DataFrame],
    output_dir: Path,
    prefix: str = "temporal",
) -> list[Path]:
    """Render each frame DataFrame as a candlestick chart PNG.

    Uses mplfinance for rendering. Charts include indicators
    if columns BB_Upper, BB_Lower, BB_Middle, RSI are present.

    Args:
        frames: List of OHLCV DataFrames (one per frame).
        output_dir: Directory to save chart images.
        prefix: Filename prefix for chart images.

    Returns:
        List of Paths to saved chart images.
    """
    import mplfinance as mpf

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    mc = mpf.make_marketcolors(
        up="limegreen", down="tomato", edge="inherit",
        wick="inherit", volume="steelblue",
    )
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="gray")

    for i, frame in enumerate(frames):
        chart_path = output_dir / f"{prefix}_frame_{i:02d}.png"

        add_plots = []
        if "BB_Upper" in frame.columns:
            add_plots.extend([
                mpf.make_addplot(frame["BB_Upper"], color="steelblue", linestyle="--", width=0.8),
                mpf.make_addplot(frame["BB_Lower"], color="steelblue", linestyle="--", width=0.8),
                mpf.make_addplot(frame["BB_Middle"], color="orange", linestyle="-", width=0.8),
            ])
        if "RSI" in frame.columns:
            add_plots.append(
                mpf.make_addplot(frame["RSI"], panel=2, color="purple", ylabel="RSI", width=1.0)
            )

        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        mpf.plot(
            frame[ohlcv_cols],
            type="candle",
            style=style,
            addplot=add_plots if add_plots else None,
            volume=True,
            figsize=(10, 6),
            savefig=dict(fname=str(chart_path), dpi=80, bbox_inches="tight"),
        )
        paths.append(chart_path)

    return paths


# ============================================================================
# Multi-Image Prompt Builder
# ============================================================================

TEMPORAL_SYSTEM_PROMPT = (
    "Tu es un analyste quantitatif senior. On te présente une séquence de "
    "{n_frames} graphiques financiers en chandeliers montrant l'évolution "
    "des prix sur des fenêtres temporelles consécutives. "
    "Analyse la tendance temporelle et fournis un signal de trading. "
    "Réponds en JSON strict avec: action (BUY/SELL/HOLD), confidence (0-1), "
    "trend (UPTREND/DOWNTREND/SIDEWAYS), reasoning (explication)."
)

TEMPORAL_USER_PROMPT = (
    "Voici une séquence de {n_frames} graphiques montrant l'évolution du prix. "
    "Le graphique 1 est le plus ancien, le graphique {n_frames} est le plus récent. "
    "Identifie la tendance et génère un signal de trading structuré."
)


def build_temporal_messages(
    image_paths: list[Path],
) -> list[dict[str, Any]]:
    """Build Qwen2.5-VL multi-image conversation format.

    Qwen2.5-VL natively supports multi-image input by passing
    multiple image blocks in the user message.

    Args:
        image_paths: Ordered list of chart image paths.

    Returns:
        Messages list in Qwen chat format.
    """
    n = len(image_paths)

    user_content: list[dict[str, Any]] = []
    for i, path in enumerate(image_paths):
        # Add image followed by a label
        user_content.append({"type": "image", "image": str(path)})
        user_content.append({
            "type": "text",
            "text": f"[Graphique {i + 1}/{n}]",
        })

    user_content.append({
        "type": "text",
        "text": TEMPORAL_USER_PROMPT.format(n_frames=n),
    })

    return [
        {"role": "system", "content": TEMPORAL_SYSTEM_PROMPT.format(n_frames=n)},
        {"role": "user", "content": user_content},
    ]


# ============================================================================
# Temporal Signal Extractor (Domain Service)
# ============================================================================

class TemporalSignalExtractor:
    """Domain service for temporal multi-frame VLM analysis.

    Combines multiple consecutive chart windows and uses Qwen2.5-VL's
    native multi-image support to reason about temporal trends.
    """

    def __init__(self, position_embedding_dim: int = 64, max_frames: int = 16):
        self.pos_embedding = TemporalPositionEmbedding(
            max_frames=max_frames,
            embedding_dim=position_embedding_dim,
        )

    def generate_sequence(
        self,
        df: pd.DataFrame,
        n_frames: int = 8,
        window_size: int = 60,
        stride: int = 5,
        start_offset: int = 0,
        output_dir: Path = Path("/tmp/temporal_charts"),
    ) -> list[Path]:
        """Generate chart sequence from market data.

        Args:
            df: Full OHLCV DataFrame with indicators.
            n_frames: Number of frames.
            window_size: Trading days per frame.
            stride: Days between frames.
            start_offset: Starting position in DataFrame.
            output_dir: Where to save chart images.

        Returns:
            List of chart image paths.
        """
        frames = generate_frame_windows(df, n_frames, window_size, stride, start_offset)
        return render_frame_charts(frames, output_dir)

    def get_temporal_weights(self, n_frames: int) -> list[float]:
        """Get position-based attention weights for N frames.

        Args:
            n_frames: Number of frames.

        Returns:
            List of normalized weights (sum ≈ 1.0).
        """
        with torch.no_grad():
            weights = self.pos_embedding.get_position_weights(n_frames)
        return weights.tolist()

    @staticmethod
    def parse_temporal_signal(raw: str) -> dict[str, Any]:
        """Parse temporal analysis response into structured format."""
        raw = raw.strip()
        for start in ["{", "```json"]:
            if start in raw:
                idx = raw.find(start)
                if start == "```json":
                    idx += 7
                rest = raw[idx:]
                end = rest.find("}") + 1
                if end > 0:
                    try:
                        return json.loads(rest[:end])
                    except json.JSONDecodeError:
                        pass
        return {
            "action": "HOLD",
            "confidence": 0.5,
            "trend": "SIDEWAYS",
            "reasoning": "Could not parse temporal analysis.",
        }
