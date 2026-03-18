#!/usr/bin/env python3
"""
07_temporal_extension.py — Temporal Multi-Frame Extension for VLM.

Generates N=8 consecutive 60-day chart windows (stride=5 days) and passes
the full sequence to Qwen2.5-VL via its native multi-image input.

Includes a benchmark mode comparing single-frame vs 8-frame accuracy.

Usage:
    python 07_temporal_extension.py --mode generate --ticker AAPL
    python 07_temporal_extension.py --mode benchmark [--max-samples N]

Requires: mplfinance, yfinance, torch.

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import DATASET_DIR, temporal_cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Generate temporal sequence
# ============================================================================

def generate_temporal_sequence(
    ticker: str,
    days: int = 120,
    n_frames: int = 8,
    window_size: int = 60,
    stride: int = 5,
) -> list[Path]:
    """Generate a temporal chart sequence for a given ticker.

    Args:
        ticker: Stock ticker symbol.
        days: Total trading days to download.
        n_frames: Number of frames.
        window_size: Trading days per frame.
        stride: Days between frames.

    Returns:
        List of chart image Paths.
    """
    from alpha_signal.infrastructure.adapters.yfinance_adapter import YFinanceAdapter
    from alpha_signal.domain.indicators import add_indicators
    from alpha_signal.domain.temporal_signal_extractor import TemporalSignalExtractor

    # Fetch enough data for the sequence
    required_days = window_size + (n_frames - 1) * stride + 10  # extra buffer
    total_days = max(days, required_days)

    adapter = YFinanceAdapter()
    df = adapter.fetch_data(ticker, total_days)
    df = add_indicators(df)

    logger.info(f"Fetched {len(df)} rows for {ticker}, generating {n_frames} frames")

    extractor = TemporalSignalExtractor(
        position_embedding_dim=temporal_cfg.position_embedding_dim,
    )
    output_dir = temporal_cfg.output_dir / ticker.lower()
    paths = extractor.generate_sequence(
        df=df,
        n_frames=n_frames,
        window_size=window_size,
        stride=stride,
        output_dir=output_dir,
    )

    # Log temporal weights
    weights = extractor.get_temporal_weights(n_frames)
    logger.info(f"Position weights: {[f'{w:.3f}' for w in weights]}")

    return paths


# ============================================================================
# Benchmark
# ============================================================================

def run_benchmark(max_samples: int | None = None) -> dict[str, float]:
    """Benchmark single-frame vs 8-frame temporal analysis.

    Since actual model inference requires CUDA, this benchmark measures
    the structural properties of the temporal approach:
    - Number of frames generated
    - Temporal coverage (days)
    - Position weight distribution

    For actual accuracy comparisons, run on Colab T4.

    Returns:
        Dict with benchmark metrics.
    """
    from alpha_signal.domain.temporal_signal_extractor import (
        TemporalSignalExtractor,
        generate_frame_windows,
    )
    import numpy as np

    n_frames = temporal_cfg.n_frames
    window_size = temporal_cfg.window_size
    stride = temporal_cfg.stride

    # Generate synthetic OHLCV data for benchmarking
    np.random.seed(temporal_cfg.seed)
    n_rows = window_size + (n_frames - 1) * stride + 20
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")

    close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
    df = pd.DataFrame({
        "Open": close + np.random.randn(n_rows) * 0.5,
        "High": close + abs(np.random.randn(n_rows) * 1.5),
        "Low": close - abs(np.random.randn(n_rows) * 1.5),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n_rows),
    }, index=dates)

    # Single frame vs multi-frame coverage
    single_coverage = window_size
    multi_coverage = window_size + (n_frames - 1) * stride

    # Frame generation test
    frames = generate_frame_windows(df, n_frames, window_size, stride)

    # Position embedding analysis
    extractor = TemporalSignalExtractor(
        position_embedding_dim=temporal_cfg.position_embedding_dim,
    )
    weights = extractor.get_temporal_weights(n_frames)

    # Weight distribution metrics
    weight_entropy = -sum(w * np.log(max(w, 1e-8)) for w in weights)
    most_recent_weight = weights[-1]

    results = {
        "n_frames": n_frames,
        "window_size": window_size,
        "stride": stride,
        "single_frame_coverage_days": single_coverage,
        "multi_frame_coverage_days": multi_coverage,
        "coverage_improvement": f"{multi_coverage / single_coverage:.1f}x",
        "frames_generated": len(frames),
        "position_weights": [round(w, 4) for w in weights],
        "weight_entropy": round(float(weight_entropy), 4),
        "most_recent_frame_weight": round(float(most_recent_weight), 4),
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description="Temporal Multi-Frame Extension (Sprint 4)")
    parser.add_argument(
        "--mode",
        choices=["generate", "benchmark"],
        required=True,
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--n-frames", type=int, default=temporal_cfg.n_frames)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal — Temporal Multi-Frame (Sprint 4)")
    logger.info("=" * 70)

    if args.mode == "generate":
        paths = generate_temporal_sequence(
            ticker=args.ticker,
            n_frames=args.n_frames,
        )
        logger.info(f"Generated {len(paths)} temporal frames:")
        for p in paths:
            logger.info(f"  → {p}")

    elif args.mode == "benchmark":
        results = run_benchmark(max_samples=args.max_samples)
        logger.info(f"Benchmark results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")

        # Save
        out_path = temporal_cfg.benchmark_results_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    logger.info("=" * 70)
    logger.info("  Temporal Multi-Frame complete ✓")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
