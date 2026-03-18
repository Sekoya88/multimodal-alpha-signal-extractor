#!/usr/bin/env python3
"""
06_grpo_training.py — GRPO (Group Relative Policy Optimization) Training.

Implements GRPO training loop for the Alpha-Signal VLM:
1. Generate N=8 predictions per chart using temperature sampling
2. Reward: 0.6 * directional_accuracy + 0.4 * calibration_error
3. Normalize rewards within group (subtract mean, divide by std)
4. PPO-style clipped policy gradient update

Usage:
    python 06_grpo_training.py [--max-samples N]

Requires: CUDA GPU, unsloth, trl, transformers, accelerate.

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from config import grpo_cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO Training (Sprint 3)")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap number of training samples for faster iteration",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal Extractor — GRPO Training (Sprint 3)")
    logger.info("=" * 70)

    if not torch.cuda.is_available():
        logger.error("CUDA required for GRPO training. Run on Colab T4 or local GPU.")
        return 1

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Group size: {grpo_cfg.group_size}")
    logger.info(f"Epsilon: {grpo_cfg.epsilon}")
    logger.info(f"Reward weights: direction={grpo_cfg.reward_weight_direction}, "
                f"calibration={grpo_cfg.reward_weight_calibration}")

    from alpha_signal.application.grpo_training_service import GRPOTrainingService

    service = GRPOTrainingService()

    # Optionally cap dataset
    dataset_path = grpo_cfg.dataset_path
    if args.max_samples:
        logger.info(f"Capping to {args.max_samples} samples")
        tmp_path = Path("/tmp/grpo_subset.jsonl")
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(line)
                    if len(samples) >= args.max_samples:
                        break
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(samples)
        dataset_path = tmp_path

    # Train
    output_dir = grpo_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = service.train(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    # Log results
    logger.info(f"✓ avg_reward: {metrics.get('avg_reward', 0):.4f}")
    logger.info(f"✓ avg_loss: {metrics.get('avg_loss', 0):.4f}")
    logger.info(f"✓ total_steps: {metrics.get('total_steps', 0)}")
    logger.info(f"✓ CSV rewards: {metrics.get('csv_path', 'N/A')}")

    # Save metrics
    metrics_path = output_dir / "grpo_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    logger.info("=" * 70)
    logger.info("  GRPO Training complete ✓")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
