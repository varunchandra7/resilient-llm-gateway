"""
cache.py — Redis response cache for POST /v1/chat.

Keys
----
  cache:{prompt_hash}   →  JSON-encoded response dict (str, TTL = CACHE_TTL_SECONDS)

Fail-open contract
------------------
Every Redis call is wrapped in try/except RedisError.  On any Redis outage:
  - get_cached_response  returns None  (treated as a cache miss)
  - set_cached_response  returns None  (silently skipped)
Both log a WARNING so the outage is visible in logs without crashing requests.
"""
import json
import logging
from typing import Any

from redis.exceptions import RedisError

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "cache:"


def _cache_key(prompt_hash: str) -> str:
    return f"{_KEY_PREFIX}{prompt_hash}"


async def get_cached_response(prompt_hash: str) -> dict[str, Any] | None:
    """Return the cached response dict for *prompt_hash*, or None on miss/error.

    Args:
        prompt_hash: Hex SHA-256 of the raw prompt (from pii.hash_prompt).

    Returns:
        Parsed response dict on a cache hit, None on a miss or Redis error.
    """
    try:
        raw = await get_redis().get(_cache_key(prompt_hash))
        if raw is None:
            return None
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Cache value for %s is corrupt — treating as miss.", prompt_hash)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis cache GET failed (%s) — treating as cache miss.", exc)
        return None


async def set_cached_response(
    prompt_hash: str,
    payload: dict[str, Any],
    ttl: int,
) -> None:
    """Store *payload* under *prompt_hash* with a TTL of *ttl* seconds.

    Args:
        prompt_hash: Hex SHA-256 of the raw prompt.
        payload:     Dict to serialise and cache (must be JSON-serialisable).
        ttl:         Time-to-live in seconds (from settings.CACHE_TTL_SECONDS).
    """
    try:
        await get_redis().setex(
            _cache_key(prompt_hash),
            ttl,
            json.dumps(payload),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis cache SET failed (%s) — response not cached.", exc)
