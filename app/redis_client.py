"""
redis_client.py — Shared async Redis client singleton.

The client is created once at import time and reused across the entire
application.  redis.asyncio.Redis is fully thread-safe and coroutine-safe.

Fail-open contract
------------------
If Redis is unreachable at startup the client object is still created;
individual commands will raise RedisError when first called.  Callers in
cache.py and rate_limiter.py catch that error and fail open (warn + continue).
"""
import logging

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
# decode_responses=True → all keys/values are str (not bytes), simplifying
# JSON serialisation in cache.py without manual .decode() calls.
#
# socket_connect_timeout / socket_timeout guard against a Redis container
# that is slow to come up during development or a network partition in prod.
# ---------------------------------------------------------------------------

_redis: aioredis.Redis = aioredis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
    socket_connect_timeout=0.25,
    socket_timeout=0.25,
    retry_on_timeout=False,
    health_check_interval=0,
)


def get_redis() -> aioredis.Redis:
    """Return the shared Redis client.

    Call this in every module instead of importing ``_redis`` directly
    so that tests can monkeypatch ``redis_client.get_redis`` easily.
    """
    return _redis


async def close_redis() -> None:
    """Cleanly close all connections in the pool (called from lifespan shutdown)."""
    await _redis.aclose()
    logger.info("Redis connection pool closed.")
