#!/usr/bin/env python3
"""
04_dpo_alignment.py — DPO Alignment for Alpha-Signal VLM.

Uses training_data.jsonl + chart images as base. Builds chosen/rejected pairs by:
- Running model inference on each chart
- Comparing predicted action vs actual (oracle from forward_return label)
- correct prediction → chosen, incorrect → rejected (or synthetic wrong when model right)

Implements full DPOTrainer pipeline with trl on Qwen2.5-VL-3B.
Uses existing QLoRA config (r=16, alpha=16, NF4).
Saves adapter to models/dpo-adapter/.

Usage:
    python 04_dpo_alignment.py [--max-samples N] [--skip-pairs]

Requires: CUDA GPU, unsloth, trl, transformers.

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

from config import DATASET_DIR, MODELS_DIR, dpo_cfg

from alpha_signal.application.dpo_alignment_service import DPOAlignmentService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_or_build_pairs(service: DPOAlignmentService, max_samples: int | None) -> list:
    """Load existing dpo_pairs or build new ones."""
    pairs_path = dpo_cfg.dpo_pairs_path
    if pairs_path.exists():
        logger.info(f"Loading pairs from {pairs_path}...")
        pairs = []
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs

    logger.info("Building preference pairs (model inference vs oracle)...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

    # Use standard Transformers for generation to avoid Unsloth/bitsandbytes RecursionError
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=dpo_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        dpo_cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=256*28*28, max_pixels=512*28*28)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = service.build_preference_pairs(
        jsonl_path=dpo_cfg.dataset_path,
        model=model,
        processor=processor,
        device=device,
        max_samples=max_samples,
    )

    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for p in pairs:
        from PIL import Image
        import base64
        import io
        imgs = p["images"]
        imgs_b64 = []
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            imgs_b64.append(base64.b64encode(buf.getvalue()).decode())
        serializable.append({
            "images_b64": imgs_b64,
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    with open(pairs_path, "w", encoding="utf-8") as f:
        for rec in serializable:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(pairs)} pairs to {pairs_path}")

    return pairs


def _normalize_prompt(r: dict) -> dict:
    """Unify prompt so content is always list (PyArrow 'cannot mix list and non-list').
    User messages need image placeholder for Qwen2.5-VL; system can be text-only.
    """
    prompt = r["prompt"]
    out = []
    for msg in prompt:
        c = msg.get("content")
        if isinstance(c, str):
            if msg.get("role") == "user":
                out.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": c}]})
            else:
                out.append({"role": msg["role"], "content": [{"type": "text", "text": c}]})
        else:
            out.append(msg)
    return {**r, "prompt": out}


def _dataset_from_pairs_file(pairs_path: Path):
    """Load pairs from JSONL and rebuild PIL images."""
    from PIL import Image
    import base64
    import io
    from datasets import Dataset

    records = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    def _decode(rec):
        imgs = []
        for b64 in rec["images_b64"]:
            imgs.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
        p = rec["prompt"]
        return {
            "images": imgs,
            # Keep base64 strings: Arrow-safe, used by _patched_process_row to reliably
            # re-decode the image regardless of how Arrow/HF-datasets serialises numpy/PIL.
            "images_b64": rec["images_b64"],
            "prompt": p,
            "prompt_raw": p,  # Preserve raw format; process_row uses it to avoid TRL's wrong placeholder count
            "chosen": rec["chosen"],
            "rejected": rec["rejected"],
        }

    decoded = [_decode(r) for r in records]
    # Normalize prompt so all message content is list (avoids PyArrow "cannot mix list and non-list")
    decoded = [_normalize_prompt(d) for d in decoded]
    for d in decoded:
        d["prompt_raw"] = d["prompt"]  # Normalized list for process_row (avoids TRL's wrong placeholder count)

    try:
        return Dataset.from_list(decoded)
    except Exception as e:
        err_str = str(e).lower()
        if "mix list and non-list" in err_str or "arrowinvalid" in err_str or "arrow" in err_str:
            # Fallback: numpy arrays play nicer with Arrow than PIL; processor accepts both
            import numpy as np
            decoded_np = []
            for d in decoded:
                decoded_np.append({
                    "images": [np.array(img) for img in d["images"]],
                    "prompt": d["prompt"],
                    "prompt_raw": d.get("prompt_raw", d["prompt"]),
                    "chosen": d["chosen"],
                    "rejected": d["rejected"],
                })
            return Dataset.from_list(decoded_np)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="DPO Alignment for Alpha-Signal VLM")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of samples for faster iteration",
    )
    parser.add_argument(
        "--skip-pairs",
        action="store_true",
        help="Skip building pairs; use existing dpo_pairs.jsonl",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal Extractor — DPO Alignment (Sprint 1)")
    logger.info("=" * 70)

    if not torch.cuda.is_available():
        logger.error("CUDA required. Run on Colab T4 or local GPU.")
        return 1

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Base model: {dpo_cfg.base_model}")
    logger.info(f"Output: {dpo_cfg.output_dir}")

    service = DPOAlignmentService()

    # 1. Build or load pairs
    if args.skip_pairs:
        if not dpo_cfg.dpo_pairs_path.exists():
            logger.error("--skip-pairs but dpo_pairs.jsonl not found. Run without --skip-pairs first.")
            return 1
        dataset = _dataset_from_pairs_file(dpo_cfg.dpo_pairs_path)
    else:
        pairs = _load_or_build_pairs(service, args.max_samples)
        dataset = _dataset_from_pairs_file(dpo_cfg.dpo_pairs_path)

    if len(dataset) == 0:
        logger.error("No preference pairs. Generate training_data.jsonl first (01_generate_dataset.py).")
        return 1

    logger.info(f"Training on {len(dataset)} preference pairs")

    # 2. Train
    dpo_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = service.train(
        pairs=list(dataset),
        output_dir=dpo_cfg.output_dir,
    )

    # 3. Log and plot
    logger.info(f"✓ train_loss: {metrics.get('train_loss', 0):.4f}")
    logger.info(f"✓ calibration_improvement: {metrics.get('calibration_improvement', 0):.4f}")

    # Save metrics for plotting
    metrics_path = dpo_cfg.output_dir / "dpo_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    logger.info("=" * 70)
    logger.info("  DPO Alignment complete ✓")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
