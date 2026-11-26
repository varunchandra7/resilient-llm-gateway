"""
rate_limiter.py — Atomic token-bucket rate limiting + daily token quota.

Token-bucket rate limiter
-------------------------
Key:  rl:{api_key}         (Redis Hash)
Fields: tokens (float), last_refill (unix timestamp float)

The Lua script runs atomically on the Redis server, eliminating any TOCTOU
race between concurrent requests on the same api_key.

Algorithm (runs inside Redis via EVAL):
  1. Load current bucket state (seed to full capacity on first call).
  2. Compute elapsed seconds since last refill.
  3. Refill: tokens = min(capacity, stored_tokens + elapsed * refill_per_sec)
  4. If tokens >= 1: consume one token → allowed; else denied.
  5. Persist new state; set TTL to 2 × drain-time so idle keys expire.
  6. Return [allowed (0/1), floor(remaining_tokens)].

Daily token-quota check
-----------------------
Key:  quota:{api_key}:{YYYY-MM-DD}   (Redis String, integer counter)
TTL:  set via EXPIREAT to next midnight UTC so the counter always resets
      at a consistent wall-clock boundary regardless of first-request time.

Fail-open contract
------------------
Both functions catch RedisError and return a permissive result
(allowed=True, remaining=-1) so a Redis outage never blocks users.
"""
import logging
import time
from datetime import datetime, timezone

from redis.exceptions import RedisError

from app.config import settings
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-bucket Lua script
# ---------------------------------------------------------------------------
# Runs atomically on the Redis server.  Arguments:
#   KEYS[1] = "rl:{api_key}"
#   ARGV[1] = current unix timestamp (float, as string)
#   ARGV[2] = bucket capacity (int)
#   ARGV[3] = refill_per_sec (float)
# Returns: {allowed, remaining}  (both integers; remaining = floor of tokens)

_TOKEN_BUCKET_LUA = """
local key          = KEYS[1]
local now          = tonumber(ARGV[1])
local capacity     = tonumber(ARGV[2])
local refill_rate  = tonumber(ARGV[3])

-- Load current state (or seed a full bucket on first call)
local stored = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens      = tonumber(stored[1]) or capacity
local last_refill = tonumber(stored[2]) or now

-- Refill based on elapsed time
local elapsed = math.max(0, now - last_refill)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

-- Try to consume one token
local allowed   = 0
local remaining = math.floor(tokens)

if tokens >= 1 then
    tokens  = tokens - 1
    allowed = 1
    remaining = math.floor(tokens)
end

-- Persist new state
redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)

-- TTL: twice the time it takes to drain a full bucket from empty
local ttl = math.ceil(capacity / refill_rate) * 2
redis.call('EXPIRE', key, ttl)

return {allowed, remaining}
"""


async def check_rate_limit(api_key: str) -> tuple[bool, int]:
    """Check the token-bucket rate limit for *api_key*.

    Returns:
        (allowed, remaining) where:
          allowed   — True if the request may proceed.
          remaining — Tokens left after this call (-1 on Redis error → fail open).
    """
    redis = get_redis()
    key = f"rl:{api_key}"
    now = time.time()

    try:
        result = await redis.eval(
            _TOKEN_BUCKET_LUA,
            1,                                          # numkeys
            key,                                        # KEYS[1]
            str(now),                                   # ARGV[1]
            str(settings.RATE_LIMIT_CAPACITY),          # ARGV[2]
            str(settings.RATE_LIMIT_REFILL_PER_SEC),    # ARGV[3]
        )
        allowed, remaining = int(result[0]), int(result[1])
        return bool(allowed), remaining

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis rate-limit check failed for key=%s (%s) — failing open.",
            api_key,
            exc,
        )
        return True, -1  # fail open: allow the request


# ---------------------------------------------------------------------------
# Daily token quota
# ---------------------------------------------------------------------------

def _quota_key(api_key: str) -> str:
    """Return today's quota key (UTC date so it resets at midnight UTC)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"quota:{api_key}:{today}"


def _next_midnight_utc() -> int:
    """Unix timestamp of the next midnight UTC (for EXPIREAT)."""
    now = datetime.now(timezone.utc)
    next_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Advance by one day
    import datetime as dt
    next_day = next_day + dt.timedelta(days=1)
    return int(next_day.timestamp())


async def check_and_increment_quota(api_key: str, tokens: int) -> bool:
    """Atomically increment the daily token counter and check the quota.

    Uses INCRBY + EXPIREAT.  The EXPIREAT is set only on the first write
    (NX option) so we don't reset the TTL on every request.

    Args:
        api_key: Identifier for the caller.
        tokens:  Number of tokens consumed by this request (prompt + completion).

    Returns:
        True if quota is not exceeded (request may proceed).
        True also if Redis is unavailable (fail open).
        False if the daily quota has been reached.
    """
    redis = get_redis()
    key = _quota_key(api_key)

    try:
        # Increment counter atomically; get new total
        new_total = await redis.incrby(key, tokens)

        # Set expiry to next midnight UTC only on the first write to this key.
        # EXPIREAT with NX (available in Redis 7+) sets TTL only if none exists.
        await redis.expireat(key, _next_midnight_utc(), nx=True)

        if new_total > settings.DAILY_TOKEN_QUOTA:
            logger.info(
                "Daily quota exceeded for api_key=%s (total=%d, quota=%d).",
                api_key, new_total, settings.DAILY_TOKEN_QUOTA,
            )
            return False
        return True

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis quota check failed for key=%s (%s) — failing open.",
            api_key,
            exc,
        )
        return True  # fail open: allow the request
