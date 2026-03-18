"""Tests for Production Inference Benchmark (Sprint 5).

Tests cover:
1. BenchmarkConfig validation
2. Redis cache adapter (mocked connection)
3. Cache key deterministic hashing
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from alpha_signal.infrastructure.adapters.inference_cache import InferenceCacheAdapter


# ============================================================================
# Unit: BenchmarkConfig
# ============================================================================

class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self):
        from config import benchmark_cfg
        assert benchmark_cfg.enable_prefix_caching is True
        assert benchmark_cfg.num_speculative_tokens == 4
        assert benchmark_cfg.speculative_draft_model == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_redis_url(self):
        from config import benchmark_cfg
        assert "redis://" in benchmark_cfg.redis_url


# ============================================================================
# Unit: InferenceCacheAdapter
# ============================================================================

class TestInferenceCacheAdapter:
    """Tests for InferenceCacheAdapter."""

    def test_generate_key_deterministic(self):
        """SHA-256 keys should map identical inputs to identical hashes."""
        adapter = InferenceCacheAdapter()
        img = "mock_base64_data"
        prompt = "test prompt"
        key1 = adapter._generate_key(img, prompt)
        key2 = adapter._generate_key(img, prompt)

        assert key1 == key2
        assert key1.startswith("vlm:inference:")

    def test_different_inputs_different_keys(self):
        """Different prompts/images should yield different keys."""
        adapter = InferenceCacheAdapter()
        key1 = adapter._generate_key("img1", "prompt")
        key2 = adapter._generate_key("img2", "prompt")
        key3 = adapter._generate_key("img1", "prompt2")

        assert key1 != key2
        assert key1 != key3

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_get_cached_response_not_connected(self, mock_redis):
        """Should return None if Redis is not connected."""
        adapter = InferenceCacheAdapter()
        # Not calling connect()
        result = await adapter.get_cached_response("img", "prompt")
        assert result is None

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_set_cached_response_not_connected(self, mock_redis):
        """Should safely ignore sets if Redis is not connected."""
        adapter = InferenceCacheAdapter()
        # Not calling connect()
        # Should not raise
        await adapter.set_cached_response("img", "prompt", {"action": "BUY"})

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_cache_hit(self, mock_redis_func):
        """Should parse JSON and return dictionary on cache hit."""
        # Setup mock Redis client
        mock_client = AsyncMock()
        mock_client.get.return_value = '{"action": "SELL", "confidence": 0.9}'
        mock_redis_func.return_value = mock_client

        adapter = InferenceCacheAdapter()
        adapter.client = mock_client
        adapter._is_connected = True  # Simulate successful connect

        result = await adapter.get_cached_response("img", "prompt")
        
        assert result is not None
        assert result["action"] == "SELL"
        assert result["confidence"] == 0.9
        
        # Verify get was called with correct key format
        mock_client.get.assert_called_once()
        key_used = mock_client.get.call_args[0][0]
        assert key_used.startswith("vlm:inference:")

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_cache_write(self, mock_redis_func):
        """Should serialize and set with TTL."""
        mock_client = AsyncMock()
        mock_redis_func.return_value = mock_client

        adapter = InferenceCacheAdapter(ttl_seconds=3600)
        adapter.client = mock_client
        adapter._is_connected = True

        payload = {"action": "BUY", "confidence": 0.8}
        await adapter.set_cached_response("img", "prompt", payload)

        mock_client.set.assert_called_once()
        args, kwargs = mock_client.set.call_args
        
        assert args[0].startswith("vlm:inference:")
        assert json.loads(args[1]) == payload
        assert kwargs["ex"] == 3600
