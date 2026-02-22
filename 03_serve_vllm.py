#!/usr/bin/env python3
"""
03_serve_vllm.py — vLLM Inference Server for the Fine-Tuned VLM.

Starts an OpenAI-compatible API server using vLLM, optimized for
vision-language model inference with paged attention, KV-cache management,
and optional tensor parallelism.

Can be used as:
  1. A Python launcher:     python 03_serve_vllm.py
  2. A CLI reference:       see the CLI_COMMAND constant below

Requirements:
    - NVIDIA GPU with ≥16 GB VRAM
    - pip install vllm>=0.6.0

Usage:
    python 03_serve_vllm.py [--dry-run]

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from typing import NoReturn

import httpx

from config import vllm_cfg

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
# CLI Command Reference
# ---------------------------------------------------------------------------

# The equivalent CLI command to start the vLLM server manually.
# This is provided for documentation — the Python launcher below builds
# the same command programmatically from VLLMConfig.
CLI_COMMAND = """
# ============================================================================
# vLLM OpenAI-Compatible Server — CLI Command Reference
# ============================================================================
#
# Basic launch:
#   python -m vllm.entrypoints.openai.api_server \\
#       --model ./models/qwen2-vl-alpha-signal \\
#       --host 0.0.0.0 \\
#       --port 8000 \\
#       --tensor-parallel-size 1 \\
#       --gpu-memory-utilization 0.90 \\
#       --max-model-len 4096 \\
#       --dtype auto \\
#       --trust-remote-code \\
#       --api-key alpha-signal-key \\
#       --max-num-seqs 64 \\
#       --enable-prefix-caching
#
# Multi-GPU (2x A100):
#   python -m vllm.entrypoints.openai.api_server \\
#       --model ./models/qwen2-vl-alpha-signal \\
#       --tensor-parallel-size 2 \\
#       --gpu-memory-utilization 0.92 \\
#       --max-model-len 8192
#
# Environment variables for advanced tuning:
#   export VLLM_ATTENTION_BACKEND=FLASH_ATTN    # Use FlashAttention-2
#   export VLLM_USE_MODELSCOPE=false             # Disable ModelScope fallback
#   export CUDA_VISIBLE_DEVICES=0,1              # Pin specific GPUs
#   export VLLM_WORKER_MULTIPROC_METHOD=spawn    # Multi-process method
#
# ============================================================================
"""


# ---------------------------------------------------------------------------
# Server Launcher
# ---------------------------------------------------------------------------

def build_server_command() -> list[str]:
    """Build the vLLM server CLI command from config.

    Returns:
        List of command-line arguments for subprocess.
    """
    cmd: list[str] = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", vllm_cfg.model_path,
        "--host", vllm_cfg.host,
        "--port", str(vllm_cfg.port),
        "--tensor-parallel-size", str(vllm_cfg.tensor_parallel_size),
        "--gpu-memory-utilization", str(vllm_cfg.gpu_memory_utilization),
        "--max-model-len", str(vllm_cfg.max_model_len),
        "--dtype", vllm_cfg.dtype,
        "--max-num-seqs", str(vllm_cfg.max_num_seqs),
        "--served-model-name", "qwen2-vl-alpha-signal",
    ]

    if vllm_cfg.trust_remote_code:
        cmd.append("--trust-remote-code")

    if vllm_cfg.api_key:
        cmd.extend(["--api-key", vllm_cfg.api_key])

    if vllm_cfg.quantization:
        cmd.extend(["--quantization", vllm_cfg.quantization])

    if vllm_cfg.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    return cmd


def health_check(
    base_url: str | None = None,
    timeout: float = 5.0,
    max_retries: int = 30,
    retry_interval: float = 2.0,
) -> bool:
    """Poll the vLLM server health endpoint until it responds.

    Args:
        base_url: Base URL of the vLLM server (default: from config).
        timeout: HTTP request timeout per attempt (seconds).
        max_retries: Maximum number of polling attempts.
        retry_interval: Seconds to wait between attempts.

    Returns:
        True if the server is healthy, False if max retries exceeded.
    """
    if base_url is None:
        base_url = f"http://localhost:{vllm_cfg.port}"

    health_url = f"{base_url}/health"
    models_url = f"{base_url}/v1/models"

    logger.info(f"Polling server health at {health_url}...")

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.get(health_url, timeout=timeout)
            if resp.status_code == 200:
                logger.info(f"✓ Server is healthy (attempt {attempt})")

                # Also verify model is loaded
                try:
                    models_resp = httpx.get(
                        models_url,
                        headers={"Authorization": f"Bearer {vllm_cfg.api_key}"},
                        timeout=timeout,
                    )
                    if models_resp.status_code == 200:
                        data = models_resp.json()
                        model_ids = [m["id"] for m in data.get("data", [])]
                        logger.info(f"  Loaded models: {model_ids}")
                except Exception:
                    pass

                return True
        except httpx.ConnectError:
            pass
        except Exception as e:
            logger.debug(f"  Health check attempt {attempt} failed: {e}")

        if attempt < max_retries:
            time.sleep(retry_interval)

    logger.error(f"Server did not become healthy after {max_retries} attempts")
    return False


def start_server() -> NoReturn:
    """Launch the vLLM server as a subprocess and monitor it.

    This function blocks indefinitely, forwarding stdout/stderr from
    the vLLM process.

    Raises:
        SystemExit: If the server process exits unexpectedly.
    """
    cmd = build_server_command()

    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal Extractor — vLLM Server Launcher")
    logger.info("=" * 70)
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Model  : {vllm_cfg.model_path}")
    logger.info(f"Endpoint: http://{vllm_cfg.host}:{vllm_cfg.port}/v1")
    logger.info(f"API Key : {vllm_cfg.api_key}")
    logger.info("=" * 70)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=None,  # Inherit current environment
        )
        # Wait for server to become healthy
        time.sleep(5)  # Initial startup delay
        if health_check():
            logger.info("Server is ready to accept requests.")
        else:
            logger.warning("Server may not be fully ready — check logs above.")

        # Block until process exits
        exit_code = process.wait()
        logger.info(f"vLLM server exited with code {exit_code}")
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Shutting down vLLM server (Ctrl+C)...")
        process.terminate()
        process.wait(timeout=10)
        logger.info("Server stopped.")
        sys.exit(0)

    except FileNotFoundError:
        logger.error(
            "vLLM is not installed. Install it with: pip install vllm"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Inference Client Example
# ---------------------------------------------------------------------------

def example_inference() -> None:
    """Demonstrate a simple inference call against the running vLLM server.

    This function is provided as a reference and is only called with --example.
    """
    import json

    base_url = f"http://localhost:{vllm_cfg.port}/v1"

    logger.info("Running example inference against vLLM server...")

    # Verify server is up
    if not health_check(max_retries=3, retry_interval=1.0):
        logger.error("Server is not running. Start it first with: python 03_serve_vllm.py")
        return

    # Example: text-only request (no image for simplicity)
    payload = {
        "model": "qwen2-vl-alpha-signal",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un analyste quantitatif. Réponds en JSON strict "
                    "avec les champs: action, confidence, reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Apple a annoncé des résultats supérieurs aux attentes. "
                    "Le RSI est à 35 et le prix touche la bande de Bollinger "
                    "inférieure. Quel est ton signal ?"
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {vllm_cfg.api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        logger.info(f"Model response:\n{json.dumps(json.loads(content), indent=2, ensure_ascii=False)}")
    except json.JSONDecodeError:
        logger.info(f"Model response (raw): {content}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the appropriate action."""
    parser = argparse.ArgumentParser(
        description="vLLM Server Launcher for the fine-tuned VLM"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the CLI command without executing it",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Only check server health, don't start a new server",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run an example inference call against a running server",
    )

    args = parser.parse_args()

    if args.dry_run:
        cmd = build_server_command()
        print("\n  " + " \\\n    ".join(cmd) + "\n")
        return

    if args.health:
        is_healthy = health_check(max_retries=3, retry_interval=1.0)
        sys.exit(0 if is_healthy else 1)

    if args.example:
        example_inference()
        return

    # Default: start the server
    start_server()


if __name__ == "__main__":
    main()
