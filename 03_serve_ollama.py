#!/usr/bin/env python3
"""
03_serve_ollama.py — Ollama Model Server Utilities for Apple Silicon (M4).

Since vLLM requires CUDA (not available on Apple Silicon), this module
provides utilities to manage Ollama-based serving for both:
  - Vision-Language Model: llama3.2-vision:11b (multimodal)
  - Text Sentiment LLM:   llama3:8b (text-only)

Ollama runs natively on Apple Silicon M-series chips with Metal acceleration,
making it the ideal inference backend for MacBook M4 (24 GB unified memory).

Usage:
    python 03_serve_ollama.py              # Health check + list models
    python 03_serve_ollama.py --pull       # Pull required models
    python 03_serve_ollama.py --test       # Run a test inference
    python 03_serve_ollama.py --test-vision # Run a vision inference test

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import httpx

from config import pipeline_cfg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_BASE = pipeline_cfg.ollama_base_url  # http://localhost:11434
REQUIRED_MODELS = [
    "llama3.2-vision:11b",  # VLM for chart analysis
    "llama3:8b",             # Text sentiment extraction
]


# ---------------------------------------------------------------------------
# Health Check & Model Management
# ---------------------------------------------------------------------------

def health_check() -> bool:
    """Check if the Ollama server is running and responsive.

    Returns:
        True if server is healthy.
    """
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            logger.info("✓ Ollama server is healthy")
            return True
    except httpx.ConnectError:
        logger.error("✗ Ollama server is not running. Start it with: ollama serve")
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
    return False


def list_models() -> list[str]:
    """List all locally available Ollama models.

    Returns:
        List of model names (e.g., ['llama3:8b', 'llama3.2-vision:11b']).
    """
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


def check_required_models() -> dict[str, bool]:
    """Verify which required models are available locally.

    Returns:
        Dict mapping model name → is_available.
    """
    available = list_models()
    status = {}
    for model in REQUIRED_MODELS:
        # Ollama may add ":latest" suffix
        is_present = any(
            model in m or model.split(":")[0] in m
            for m in available
        )
        status[model] = is_present
        icon = "✓" if is_present else "✗"
        logger.info(f"  {icon} {model}")
    return status


def pull_models() -> None:
    """Pull all required models via Ollama API.

    This may take several minutes depending on network speed.
    """
    for model in REQUIRED_MODELS:
        logger.info(f"Pulling {model}...")
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/pull",
                json={"name": model, "stream": False},
                timeout=600.0,  # 10 min timeout for large models
            )
            if resp.status_code == 200:
                logger.info(f"  ✓ {model} pulled successfully")
            else:
                logger.warning(f"  ✗ Pull failed for {model}: {resp.text}")
        except Exception as e:
            logger.error(f"  ✗ Pull error for {model}: {e}")


# ---------------------------------------------------------------------------
# Test Inference
# ---------------------------------------------------------------------------

def test_text_inference() -> None:
    """Run a simple text inference test against the sentiment model."""
    logger.info("Running text inference test (llama3:8b)...")

    payload = {
        "model": "llama3:8b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un analyste de sentiment financier. "
                    "Réponds uniquement en JSON avec les champs: "
                    "sentiment (BULLISH/BEARISH/NEUTRAL), intensity (0-1), summary."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Apple annonce des résultats trimestriels record avec "
                    "une hausse de 12% du chiffre d'affaires."
                ),
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result.get("message", {}).get("content", "")
        logger.info(f"Response:\n{content}")
    except Exception as e:
        logger.error(f"Text inference failed: {e}")


def test_vision_inference(image_path: str | None = None) -> None:
    """Run a vision inference test against the VLM.

    Args:
        image_path: Optional path to a test image. If None, uses a
                   chart from the dataset directory.
    """
    logger.info("Running vision inference test (llama3.2-vision:11b)...")

    # Find a test image
    if image_path is None:
        dataset_dir = Path(__file__).parent / "dataset" / "charts"
        images = list(dataset_dir.glob("*.png"))
        if not images:
            # Generate a simple test from the JSONL
            jsonl_path = Path(__file__).parent / "dataset" / "training_data.jsonl"
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    first_line = json.loads(f.readline())
                # Extract the base64 image from the training data
                for block in first_line["messages"][1]["content"]:
                    if block.get("type") == "image":
                        # Remove data URI prefix
                        img_data = block["image"].split(",", 1)[1]
                        img_bytes = base64.b64decode(img_data)
                        test_img_path = Path(__file__).parent / "dataset" / "test_chart.png"
                        test_img_path.write_bytes(img_bytes)
                        image_path = str(test_img_path)
                        logger.info(f"  Extracted test chart from JSONL → {test_img_path}")
                        break
            if image_path is None:
                logger.error("No test image found. Run 01_generate_dataset.py first.")
                return
        else:
            image_path = str(images[0])

    # Read and encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "llama3.2-vision:11b",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Analyse ce graphique financier en chandeliers japonais. "
                    "Identifie le RSI, les bandes de Bollinger, et génère "
                    "un signal de trading (BUY/SELL/HOLD) en JSON avec les "
                    "champs: action, confidence, reasoning."
                ),
                "images": [img_b64],
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result.get("message", {}).get("content", "")
        logger.info(f"Vision Response:\n{content}")
    except Exception as e:
        logger.error(f"Vision inference failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the appropriate action."""
    parser = argparse.ArgumentParser(
        description="Ollama Model Server Utilities for M4"
    )
    parser.add_argument("--pull", action="store_true", help="Pull required models")
    parser.add_argument("--test", action="store_true", help="Test text inference")
    parser.add_argument("--test-vision", action="store_true", help="Test vision inference")
    parser.add_argument("--image", type=str, default=None, help="Image for vision test")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Ollama Server Utilities — Apple Silicon M4")
    logger.info("=" * 60)

    if not health_check():
        sys.exit(1)

    logger.info("Available models:")
    available = list_models()
    for m in available:
        logger.info(f"  • {m}")

    logger.info("\nRequired models status:")
    model_status = check_required_models()

    if args.pull:
        pull_models()
        return

    missing = [m for m, ok in model_status.items() if not ok]
    if missing:
        logger.warning(f"\nMissing models: {missing}")
        logger.warning("Run: python 03_serve_ollama.py --pull")

    if args.test:
        test_text_inference()

    if args.test_vision:
        test_vision_inference(args.image)

    if not (args.pull or args.test or args.test_vision):
        logger.info("\nAll checks complete. Use --test or --test-vision to run inference.")


if __name__ == "__main__":
    main()
