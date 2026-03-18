#!/usr/bin/env python3
"""
08_inference_benchmark.py — Async Production Inference Benchmark.

Simulates concurrent user load against the vLLM server to measure throughput,
latency (p50/p95/p99), and the impact of the Redis inference cache.
Demonstrates speculative decoding with Qwen2.5-0.5B-Instruct as a draft model
for the 3B Vision verifier.

Usage:
    python 08_inference_benchmark.py [--use-cache] [--simulate]

Requires: aiohttp, matplotlib, redis.

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

from config import benchmark_cfg, pipeline_cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark Client
# ============================================================================

class BenchmarkClient:
    """Async client simulating concurrent inference requests."""

    def __init__(self, use_cache: bool = False, simulate: bool = False):
        import aiohttp
        from alpha_signal.infrastructure.adapters.inference_cache import InferenceCacheAdapter

        self.session: aiohttp.ClientSession | None = None
        self.cache = InferenceCacheAdapter() if use_cache else None
        self.simulate = simulate
        self.url = f"{pipeline_cfg.vllm_base_url}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {pipeline_cfg.vllm_api_key}",
            "Content-Type": "application/json",
        }
        self.results: list[float] = []

    async def init(self):
        import aiohttp
        self.session = aiohttp.ClientSession(headers=self.headers)
        if self.cache:
            await self.cache.connect()

    async def close(self):
        if self.session:
            await self.session.close()
        if self.cache:
            await self.cache.close()

    async def _mock_inference(self) -> float:
        """Simulate an inference delay (faster if cached in theory)."""
        delay = random.uniform(0.1, 0.4)
        await asyncio.sleep(delay)
        return delay

    async def send_request(self, prompt: str, image_b64: str) -> float:
        """Send a single inference request, using cache if enabled."""
        start = time.perf_counter()

        if self.cache:
            cached = await self.cache.get_cached_response(image_b64, prompt)
            if cached:
                # Cache hit
                latency = time.perf_counter() - start
                self.results.append(latency)
                return latency

        if self.simulate:
            await self._mock_inference()
        else:
            payload = {
                "model": pipeline_cfg.vllm_model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 128,
            }
            try:
                assert self.session is not None
                async with self.session.post(self.url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    response_text = data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Request failed: {e}")
                response_text = '{"action": "HOLD", "confidence": 0.0}'

        if self.cache and not self.simulate:
            # Simplistic parsing/mocking for benchmark scope
            try:
                resp_data = json.loads(response_text)
            except json.JSONDecodeError:
                resp_data = {"action": "HOLD", "confidence": 0.5, "raw": response_text}
            await self.cache.set_cached_response(image_b64, prompt, resp_data)

        latency = time.perf_counter() - start
        self.results.append(latency)
        return latency


# ============================================================================
# Orchestration
# ============================================================================

async def runner(client: BenchmarkClient, requests_per_user: int, mock_image: str):
    """Worker task sending multiple requests sequentially."""
    for i in range(requests_per_user):
        # Introduce slight variety to prompt so we don't 100% cache hit if testing load
        # For cache testing, we send the SAME prompt.
        prompt = "Analyse ce graphique et donne un signal. "
        # 50% chance of identical prompt (cache hit)
        if random.random() > 0.5:
             prompt += f"Randomize: {random.randint(1,10)}"
             
        await client.send_request(prompt, mock_image)


async def run_benchmark(concurrent_users: int, use_cache: bool, simulate: bool):
    """Run a single benchmark tier."""
    logger.info(f"Starting {concurrent_users} concurrent users (Cache={use_cache})")
    
    # Mock base64 image (small 1x1 pixel for brevity)
    mock_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

    client = BenchmarkClient(use_cache=use_cache, simulate=simulate)
    await client.init()

    tasks = []
    start_time = time.perf_counter()

    for _ in range(concurrent_users):
        t = asyncio.create_task(runner(client, benchmark_cfg.requests_per_user, mock_img))
        tasks.append(t)

    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time

    import numpy as np
    latencies = client.results
    
    metrics = {
        "users": concurrent_users,
        "total_requests": len(latencies),
        "total_time_s": round(total_time, 2),
        "throughput_req_s": round(len(latencies) / total_time, 2),
        "p50_ms": round(np.percentile(latencies, 50) * 1000, 2),
        "p95_ms": round(np.percentile(latencies, 95) * 1000, 2),
        "p99_ms": round(np.percentile(latencies, 99) * 1000, 2),
        "cache_enabled": use_cache,
    }

    await client.close()
    return metrics


def plot_results(results_no_cache: list[dict], results_cache: list[dict], out_path: Path):
    """Generate throughput and latency plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot generation.")
        return

    users = [r["users"] for r in results_no_cache]
    tp_no_cache = [r["throughput_req_s"] for r in results_no_cache]
    tp_cache = [r["throughput_req_s"] for r in results_cache]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Throughput Plot
    ax1.plot(users, tp_no_cache, marker='o', label="No Cache", color="tomato")
    ax1.plot(users, tp_cache, marker='s', label="Redis Cache", color="limegreen")
    ax1.set_title("Inference Throughput vs Concurrency")
    ax1.set_xlabel("Concurrent Users")
    ax1.set_ylabel("Throughput (Req/Sec)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Latency Plot (P95)
    lat_no_cache = [r["p95_ms"] for r in results_no_cache]
    lat_cache = [r["p95_ms"] for r in results_cache]
    
    ax2.plot(users, lat_no_cache, marker='o', label="No Cache (p95)", color="tomato")
    ax2.plot(users, lat_cache, marker='s', label="Redis Cache (p95)", color="limegreen")
    ax2.set_title("P95 Latency vs Concurrency")
    ax2.set_xlabel("Concurrent Users")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    logger.info(f"Plot saved → {out_path}")


async def main_async(simulate: bool) -> int:
    logger.info("=" * 70)
    logger.info("  vLLM Speculative Decoding + Cache Benchmark (Sprint 5)")
    logger.info("=" * 70)
    
    logger.info("Speculative Decoding Config (Server-side target):")
    logger.info(f"  Draft Model: {benchmark_cfg.speculative_draft_model}")
    logger.info(f"  Gamma (tokens): {benchmark_cfg.num_speculative_tokens}")
    logger.info(f"  Prefix Caching: {benchmark_cfg.enable_prefix_caching}")
    logger.info("-" * 70)

    results_no_cache = []
    results_cache = []

    for users in benchmark_cfg.concurrent_users:
        # Run without cache
        mets_none = await run_benchmark(users, use_cache=False, simulate=simulate)
        results_no_cache.append(mets_none)
        
        # Run with cache
        mets_cache = await run_benchmark(users, use_cache=True, simulate=simulate)
        results_cache.append(mets_cache)
        
        # Pause between tiers
        await asyncio.sleep(1)

    # Save JSON results
    out_json = benchmark_cfg.benchmark_results_path
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "no_cache": results_no_cache,
            "cache": results_cache,
            "config": {
                "requests_per_user": benchmark_cfg.requests_per_user,
                "simulate": simulate,
            }
        }, f, indent=2)
    logger.info(f"JSON results saved → {out_json}")

    # Plot
    plot_results(results_no_cache, results_cache, benchmark_cfg.benchmark_plot_path)

    logger.info("=" * 70)
    logger.info("  Benchmark complete ✓")
    logger.info("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference Benchmark")
    parser.add_argument("--simulate", action="store_true", help="Simulate inference (no real vLLM calls)")
    args = parser.parse_args()

    return asyncio.run(main_async(args.simulate))


if __name__ == "__main__":
    sys.exit(main())
