"""
simulator.py — Deterministic discrete-event replay simulator.

Models:
  - Per-key token-bucket rate limiting
  - Per-key daily token quota
  - TTL response cache (cache hits at 2.0ms, $0.0)
  - Upstream latency + retry/fallback overhead when primary_error_rate > 0
  - Token cost savings vs. a zero-cache baseline
"""
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RequestLog
from app.pricing import estimate_cost_usd


# ---------------------------------------------------------------------------
# Pydantic request / policy models
# ---------------------------------------------------------------------------


class SimulationPolicy(BaseModel):
    name: str = "candidate"
    cache_ttl_seconds: int = Field(300, ge=0)
    rate_limit_capacity: int = Field(10, ge=1)
    rate_limit_refill_per_sec: float = Field(2.0, gt=0.0)
    daily_token_quota: int = Field(100_000, ge=1)
    primary_error_rate: float = Field(0.0, ge=0.0, le=1.0)


class SimulationRequest(BaseModel):
    source: Literal["postgres", "synthetic"] = "synthetic"
    log_limit: int = Field(500, ge=1, le=5000)
    synthetic_count: int = Field(200, ge=1, le=10000)
    unique_prompts: int = Field(30, ge=1, le=1000)
    rps: float = Field(5.0, gt=0.0)
    seed: int = 42
    policies: list[SimulationPolicy] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Normalized event representation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SimEvent:
    timestamp: float
    api_key: str
    prompt: str
    prompt_hash: str
    prompt_tokens: int
    completion_tokens: int
    provider: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Trace generators / loaders
# ---------------------------------------------------------------------------


def generate_synthetic_events(
    count: int,
    unique_prompts: int,
    rps: float,
    seed: int = 42,
    prompt_prefix: str = "sim-prompt",
    api_key: str = "sim-key-1",
    base_latency_ms: float = 150.0,
) -> list[SimEvent]:
    """Generate a deterministic Zipf-distributed synthetic event trace."""
    rng = random.Random(seed)

    # Zipf weights: 1 / (rank + 1) creates ~80/20 hot-prompt skew
    weights = [1.0 / (rank + 1) for rank in range(unique_prompts)]

    catalog: list[tuple[str, str, int, int]] = []
    for idx in range(unique_prompts):
        prompt_text = f"{prompt_prefix}-{idx}: explain resilient llm gateway pattern {idx}"
        p_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        completion_text = f"This is a mock response to: '{prompt_text}'"
        p_tokens = len(prompt_text.split())
        c_tokens = len(completion_text.split())
        catalog.append((prompt_text, p_hash, p_tokens, c_tokens))

    step = 1.0 / rps
    events: list[SimEvent] = []
    for i in range(count):
        prompt_text, p_hash, p_tokens, c_tokens = rng.choices(catalog, weights=weights, k=1)[0]
        events.append(
            SimEvent(
                timestamp=i * step,
                api_key=api_key,
                prompt=prompt_text,
                prompt_hash=p_hash,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                provider="mock-primary",
                latency_ms=base_latency_ms,
            )
        )
    return events


async def load_postgres_events(session: AsyncSession, limit: int) -> list[SimEvent]:
    """Load recent RequestLog rows ordered chronologically (created_at ASC)."""
    result = await session.execute(
        select(RequestLog)
        .order_by(RequestLog.created_at.desc())
        .limit(limit)
    )
    rows = list(reversed(result.scalars().all()))
    if not rows:
        return []

    t_base = rows[0].created_at
    events: list[SimEvent] = []
    for row in rows:
        ts = max(0.0, (row.created_at - t_base).total_seconds())
        p_tokens = row.prompt_tokens if row.prompt_tokens is not None else max(1, len(row.redacted_prompt.split()))
        c_tokens = row.completion_tokens if row.completion_tokens is not None else 12
        provider = row.provider if row.provider in ("gemini", "mock-primary", "mock-fallback") else "mock-primary"
        base_lat = row.latency_ms if not row.cache_hit and row.status_code == 200 else 150.0
        events.append(
            SimEvent(
                timestamp=ts,
                api_key="replay-key",
                prompt=row.redacted_prompt,
                prompt_hash=row.prompt_hash,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                provider=provider,
                latency_ms=base_lat,
            )
        )
    return events


# ---------------------------------------------------------------------------
# Discrete-event simulation engine
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile from an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    idx = (pct / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(sorted_values[lo], 2)
    frac = idx - lo
    val = sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
    return round(val, 2)


def simulate_policy(
    events: list[SimEvent],
    policy: SimulationPolicy,
    seed: int = 42,
) -> dict:
    """Replay *events* chronologically under *policy* using a virtual clock."""
    if not events:
        return {
            "policy_name": policy.name,
            "total_requests": 0,
            "cache_hits": 0,
            "cache_hit_rate_pct": 0.0,
            "rate_limited_count": 0,
            "fallback_count": 0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "cost_savings_pct": 0.0,
        }

    rng = random.Random(seed)
    buckets: dict[str, tuple[float, float]] = {}
    daily_quotas: dict[tuple[str, int], int] = {}
    cache: dict[str, float] = {}

    latencies: list[float] = []
    cache_hits = 0
    rate_limited_count = 0
    fallback_count = 0
    total_cost = 0.0
    baseline_cost = 0.0

    for ev in events:
        # 1. Token-bucket rate limit
        tokens, last_refill = buckets.get(ev.api_key, (float(policy.rate_limit_capacity), ev.timestamp))
        elapsed = max(0.0, ev.timestamp - last_refill)
        tokens = min(float(policy.rate_limit_capacity), tokens + elapsed * policy.rate_limit_refill_per_sec)

        if tokens < 1.0:
            buckets[ev.api_key] = (tokens, ev.timestamp)
            rate_limited_count += 1
            latencies.append(1.0)
            continue

        tokens -= 1.0
        buckets[ev.api_key] = (tokens, ev.timestamp)

        # 2. Daily token quota check
        day_idx = int(ev.timestamp // 86400)
        q_key = (ev.api_key, day_idx)
        used_tokens = daily_quotas.get(q_key, 0)
        if used_tokens + ev.prompt_tokens > policy.daily_token_quota:
            rate_limited_count += 1
            latencies.append(1.0)
            continue

        # Baseline cost for any admitted request (no cache)
        baseline_cost += estimate_cost_usd(
            ev.provider, ev.prompt_tokens, ev.completion_tokens, cache_hit=False
        )

        # 3. TTL cache lookup
        expires_at = cache.get(ev.prompt_hash)
        if policy.cache_ttl_seconds > 0 and expires_at is not None and ev.timestamp < expires_at:
            cache_hits += 1
            latencies.append(2.0)
            daily_quotas[q_key] = used_tokens + ev.prompt_tokens
            continue

        # 4. Cache miss -> upstream provider (+ modeled retry/fallback if primary fails)
        used_provider = ev.provider
        req_latency = ev.latency_ms

        if policy.primary_error_rate > 0.0 and rng.random() < policy.primary_error_rate:
            # Model 2 retries with jitter + fallback to mock-fallback (120ms)
            retry_overhead = rng.uniform(0, 100) + rng.uniform(0, 200)
            req_latency = ev.latency_ms + retry_overhead + 120.0
            used_provider = "mock-fallback"
            fallback_count += 1

        req_cost = estimate_cost_usd(
            used_provider, ev.prompt_tokens, ev.completion_tokens, cache_hit=False
        )
        total_cost += req_cost
        latencies.append(req_latency)

        if policy.cache_ttl_seconds > 0:
            cache[ev.prompt_hash] = ev.timestamp + policy.cache_ttl_seconds
        daily_quotas[q_key] = used_tokens + ev.prompt_tokens + ev.completion_tokens

    total_requests = len(events)
    admitted = total_requests - rate_limited_count
    sorted_lat = sorted(latencies)
    cache_hit_rate_pct = round((cache_hits / admitted) * 100.0, 2) if admitted > 0 else 0.0
    cost_savings_pct = (
        round(((baseline_cost - total_cost) / baseline_cost) * 100.0, 2)
        if baseline_cost > 0.0
        else 0.0
    )

    return {
        "policy_name": policy.name,
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_hit_rate_pct": cache_hit_rate_pct,
        "rate_limited_count": rate_limited_count,
        "fallback_count": fallback_count,
        "p50_latency_ms": _percentile(sorted_lat, 50.0),
        "p95_latency_ms": _percentile(sorted_lat, 95.0),
        "p99_latency_ms": _percentile(sorted_lat, 99.0),
        "total_cost_usd": round(total_cost, 8),
        "cost_savings_pct": cost_savings_pct,
    }
