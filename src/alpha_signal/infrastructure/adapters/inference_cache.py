"""inference_cache.py — Redis cache for production inference.

Caches expensive visual embeddings and final responses to avoid
re-running the vision encoder and LLM for identical charts/prompts.
Optimizes vLLM throughput in concurrent multi-user environments.

Uses an SHA-256 hash of the image bytes + prompt as the cache key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ConnectionError

from config import benchmark_cfg

logger = logging.getLogger(__name__)


class InferenceCacheAdapter:
    """Adapter for caching VLM inference results in Redis."""

    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None):
        """Initialize the cache connection.

        Args:
            redis_url: Overrides default URL in config.
            ttl_seconds: Cache time-to-live in seconds.
        """
        self.url = redis_url or benchmark_cfg.redis_url
        self.ttl = ttl_seconds or benchmark_cfg.cache_ttl_seconds
        self.client = redis.from_url(self.url, decode_responses=True)
        self._is_connected = False

    async def connect(self) -> None:
        """Verify connection to Redis."""
        try:
            await self.client.ping()
            self._is_connected = True
            logger.debug(f"Connected to Redis cache at {self.url}")
        except ConnectionError as e:
            logger.warning(f"Failed to connect to Redis cache: {e}. Falling back to no-cache.")
            self._is_connected = False

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.client.aclose()
        self._is_connected = False

    @staticmethod
    def _generate_key(image_base64: str, prompt: str) -> str:
        """Generate a deterministic SHA-256 cache key.

        Args:
            image_base64: Base64 encoded image string.
            prompt: Text prompt sent to the VLM.

        Returns:
            SHA-256 hash string prefix "vlm:inference:".
        """
        raw = f"{image_base64}:{prompt}".encode("utf-8")
        hash_hex = hashlib.sha256(raw).hexdigest()
        return f"vlm:inference:{hash_hex}"

    async def get_cached_response(self, image_base64: str, prompt: str) -> dict[str, Any] | None:
        """Retrieve a cached VLM response if available.

        Args:
            image_base64: Base64 encoded chart image.
            prompt: Text prompt.

        Returns:
            Parsed JSON response dict if found, else None.
        """
        if not self._is_connected:
            return None

        key = self._generate_key(image_base64, prompt)
        try:
            val = await self.client.get(key)
            if val:
                logger.debug("Cache hit")
                return json.loads(val)
        except Exception as e:
            logger.error(f"Read cache error: {e}")

        return None

    async def set_cached_response(self, image_base64: str, prompt: str, response: dict[str, Any]) -> None:
        """Cache a VLM response.

        Args:
            image_base64: Base64 encoded chart image.
            prompt: Text prompt.
            response: Dict to cache (action, confidence, reasoning, etc.).
        """
        if not self._is_connected:
            return

        key = self._generate_key(image_base64, prompt)
        try:
            val = json.dumps(response, ensure_ascii=False)
            await self.client.set(key, val, ex=self.ttl)
        except Exception as e:
            logger.error(f"Write cache error: {e}")
