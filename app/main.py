"""
main.py — Resilient LLM Gateway (Step 5: Gemini SDK + SSE streaming).

Routes
------
GET  /health          → liveness probe
POST /v1/chat         → rate-limit → quota → cache → resilience engine (or SSE stream) → log
GET  /v1/logs         → tail recent request_logs rows (?limit=20)
GET  /v1/circuits     → live circuit-breaker state for all providers
"""
import json
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, text

from app.cache import get_cached_response, set_cached_response
from app.circuit_breaker import get_circuit_states
from app.config import settings
from app.db import AsyncSessionLocal, Base, engine
from app.models import RequestLog
from app.pii import hash_prompt, redact_pii
from app.pricing import estimate_cost_usd
from app.rate_limiter import check_and_increment_quota, check_rate_limit
from app.redis_client import close_redis
from app.resilience import call_with_resilience, stream_with_resilience
from app.simulator import (
    SimulationPolicy,
    SimulationRequest,
    _percentile,
    generate_synthetic_events,
    load_postgres_events,
    simulate_policy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

_MIGRATE_SQL = text(
    "ALTER TABLE request_logs "
    "ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables + migrate schema.  Shutdown: close Redis + engine."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(_MIGRATE_SQL)
    logger.info("Database tables created / schema migrated.")
    yield
    await close_redis()
    await engine.dispose()
    logger.info("Engine and Redis pool disposed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Resilient LLM Gateway",
    version="0.5.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user prompt")
    latency_ms: int = Field(150, ge=0, description="Simulated provider latency in ms")
    error_rate: float = Field(0.0, ge=0.0, le=1.0, description="Provider error probability")
    stream: bool = Field(False, description="Stream response via Server-Sent Events")
    use_mock: bool = Field(False, description="Force mock provider instead of Gemini")


class ChatResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cache_hit: bool
    provider: str
    retry_count: int


class LogEntry(BaseModel):
    id: str
    created_at: str
    prompt_hash: str
    redacted_prompt: str
    provider: str
    status_code: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None
    cache_hit: bool
    retry_count: int


# ---------------------------------------------------------------------------
# Background DB logger — never raises; swallows all DB exceptions
# ---------------------------------------------------------------------------


async def _log_request(
    prompt_hash: str,
    redacted_prompt: str,
    status_code: int,
    latency_ms: float,
    provider: str = "gateway",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
    cache_hit: bool = False,
    retry_count: int = 0,
) -> None:
    """Write a RequestLog row via a standalone async session."""
    try:
        async with AsyncSessionLocal() as session:
            row = RequestLog(
                prompt_hash=prompt_hash,
                redacted_prompt=redacted_prompt,
                provider=provider,
                status_code=status_code,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost_usd,
                cache_hit=cache_hit,
                retry_count=retry_count,
            )
            session.add(row)
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("Background DB log failed — user response unaffected.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    x_api_key: str = Header(default="dev-key-123", alias="X-API-Key"),
) -> Any:
    """
    Full gateway pipeline: rate-limit → quota → cache → resilience engine → log.
    Supports both standard JSON responses and Server-Sent Events (stream=True).
    """
    t0 = time.perf_counter()

    # ① Pre-compute hash and redacted prompt
    p_hash = hash_prompt(request.prompt)
    safe_prompt = redact_pii(request.prompt)
    est_tokens = len(request.prompt.split())

    # ------------------------------------------------------------------ #
    # ② Token-bucket rate limit                                           #
    # ------------------------------------------------------------------ #
    allowed, remaining = await check_rate_limit(x_api_key)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    if not allowed:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        retry_after = math.ceil(1.0 / settings.RATE_LIMIT_REFILL_PER_SEC)
        response.headers["Retry-After"] = str(retry_after)
        response.headers["X-RateLimit-Remaining"] = "0"
        background_tasks.add_task(
            _log_request,
            prompt_hash=p_hash,
            redacted_prompt=safe_prompt,
            status_code=429,
            latency_ms=elapsed_ms,
            provider="gateway",
            estimated_cost_usd=0.0,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
        )

    # ------------------------------------------------------------------ #
    # ③ Daily token quota                                                 #
    # ------------------------------------------------------------------ #
    quota_ok = await check_and_increment_quota(x_api_key, est_tokens)
    if not quota_ok:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        background_tasks.add_task(
            _log_request,
            prompt_hash=p_hash,
            redacted_prompt=safe_prompt,
            status_code=429,
            latency_ms=elapsed_ms,
            provider="gateway",
            estimated_cost_usd=0.0,
        )
        raise HTTPException(status_code=429, detail="Daily token quota exceeded.")

    # ------------------------------------------------------------------ #
    # ④ Cache lookup                                                      #
    # ------------------------------------------------------------------ #
    cached = await get_cached_response(p_hash)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        provider_from_cache = cached.get("provider", "cache")
        cached_p_tok = cached.get("prompt_tokens") or 0
        cached_c_tok = cached.get("completion_tokens") or 0
        cost_usd = estimate_cost_usd(
            provider_from_cache, cached_p_tok, cached_c_tok, cache_hit=True
        )
        background_tasks.add_task(
            _log_request,
            prompt_hash=p_hash,
            redacted_prompt=safe_prompt,
            status_code=200,
            latency_ms=elapsed_ms,
            provider=provider_from_cache,
            prompt_tokens=cached.get("prompt_tokens"),
            completion_tokens=cached.get("completion_tokens"),
            estimated_cost_usd=cost_usd,
            cache_hit=True,
            retry_count=0,
        )

        if request.stream:
            async def cached_sse_generator():
                frame = json.dumps({"chunk": cached["text"], "cached": True})
                yield f"data: {frame}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                cached_sse_generator(),
                media_type="text/event-stream",
                headers={"X-RateLimit-Remaining": str(remaining)},
            )

        return ChatResponse(
            text=cached["text"],
            prompt_tokens=cached["prompt_tokens"],
            completion_tokens=cached["completion_tokens"],
            latency_ms=round(elapsed_ms, 2),
            cache_hit=True,
            provider=provider_from_cache,
            retry_count=0,
        )

    # ------------------------------------------------------------------ #
    # ⑤ Streaming cache-miss path (SSE with stream-teeing)                #
    # ------------------------------------------------------------------ #
    if request.stream:
        async def live_sse_generator():
            chunks: list[str] = []
            provider_used = "unknown"
            p_tokens: int | None = None
            c_tokens: int | None = None
            stream_status = 200

            try:
                async for chunk, prov in stream_with_resilience(
                    prompt=request.prompt,
                    latency_ms=request.latency_ms,
                    error_rate=request.error_rate,
                    use_mock=request.use_mock,
                ):
                    provider_used = prov
                    chunk_text = chunk.get("text") or ""
                    if chunk.get("prompt_tokens") is not None:
                        p_tokens = chunk["prompt_tokens"]
                    if chunk.get("completion_tokens") is not None:
                        c_tokens = chunk["completion_tokens"]

                    if chunk_text:
                        chunks.append(chunk_text)
                        frame = json.dumps(
                            {
                                "chunk": chunk_text,
                                "cached": False,
                                "provider": provider_used,
                            }
                        )
                        yield f"data: {frame}\n\n"

                full_text = "".join(chunks)
                if p_tokens is None:
                    p_tokens = len(request.prompt.split())
                if c_tokens is None:
                    c_tokens = len(full_text.split())

                # Store full response in Redis cache
                cacheable = {
                    "text": full_text,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "provider": provider_used,
                }
                await set_cached_response(p_hash, cacheable, settings.CACHE_TTL_SECONDS)

                # Account for remaining streamed tokens in daily quota (est_tokens already counted pre-stream)
                remaining_tokens = max(0, (p_tokens + c_tokens) - est_tokens)
                if remaining_tokens > 0:
                    await check_and_increment_quota(x_api_key, remaining_tokens)

                yield "data: [DONE]\n\n"

            except HTTPException as exc:
                stream_status = exc.status_code
                err_frame = json.dumps({"error": exc.detail, "status_code": stream_status})
                yield f"data: {err_frame}\n\n"
                yield "data: [DONE]\n\n"
            except httpx.HTTPStatusError as exc:
                stream_status = exc.response.status_code
                err_frame = json.dumps({"error": str(exc), "status_code": stream_status})
                yield f"data: {err_frame}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                stream_status = 500
                err_frame = json.dumps({"error": str(exc), "status_code": 500})
                yield f"data: {err_frame}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                stream_cost = (
                    estimate_cost_usd(provider_used, p_tokens or 0, c_tokens or 0, cache_hit=False)
                    if stream_status == 200
                    else 0.0
                )
                await _log_request(
                    prompt_hash=p_hash,
                    redacted_prompt=safe_prompt,
                    status_code=stream_status,
                    latency_ms=elapsed_ms,
                    provider=provider_used,
                    prompt_tokens=p_tokens if stream_status == 200 else None,
                    completion_tokens=c_tokens if stream_status == 200 else None,
                    estimated_cost_usd=stream_cost,
                    cache_hit=False,
                    retry_count=0,
                )

        return StreamingResponse(
            live_sse_generator(),
            media_type="text/event-stream",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )

    # ------------------------------------------------------------------ #
    # ⑥ Non-streaming resilience engine (cache miss path)                 #
    # ------------------------------------------------------------------ #
    status_code = 200
    result: dict | None = None
    provider_used = "unknown"
    retry_count = 0

    try:
        result, provider_used, retry_count = await call_with_resilience(
            prompt=request.prompt,
            latency_ms=request.latency_ms,
            error_rate=request.error_rate,
            use_mock=request.use_mock,
        )
    except HTTPException as exc:
        status_code = exc.status_code
        elapsed_ms = (time.perf_counter() - t0) * 1000
        background_tasks.add_task(
            _log_request,
            prompt_hash=p_hash,
            redacted_prompt=safe_prompt,
            status_code=status_code,
            latency_ms=elapsed_ms,
            provider=provider_used,
            estimated_cost_usd=0.0,
            retry_count=retry_count,
        )
        raise
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        elapsed_ms = (time.perf_counter() - t0) * 1000
        background_tasks.add_task(
            _log_request,
            prompt_hash=p_hash,
            redacted_prompt=safe_prompt,
            status_code=status_code,
            latency_ms=elapsed_ms,
            provider=provider_used,
            estimated_cost_usd=0.0,
            retry_count=retry_count,
        )
        raise HTTPException(status_code=status_code, detail=str(exc))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ⑦ Cache the successful result
    cacheable = {**result, "provider": provider_used}
    await set_cached_response(p_hash, cacheable, settings.CACHE_TTL_SECONDS)

    # ⑧ Log to Postgres in the background
    cost_usd = estimate_cost_usd(
        provider_used,
        result["prompt_tokens"],
        result["completion_tokens"],
        cache_hit=False,
    )
    background_tasks.add_task(
        _log_request,
        prompt_hash=p_hash,
        redacted_prompt=safe_prompt,
        status_code=200,
        latency_ms=elapsed_ms,
        provider=provider_used,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        estimated_cost_usd=cost_usd,
        cache_hit=False,
        retry_count=retry_count,
    )

    return ChatResponse(
        text=result["text"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        latency_ms=round(elapsed_ms, 2),
        cache_hit=False,
        provider=provider_used,
        retry_count=retry_count,
    )


@app.get("/v1/logs", response_model=list[LogEntry])
async def get_logs(limit: int = Query(20, ge=1, le=200)) -> Any:
    """Return the *limit* most recent request_log rows, newest first."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(RequestLog)
            .order_by(RequestLog.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    return [
        LogEntry(
            id=str(row.id),
            created_at=row.created_at.isoformat(),
            prompt_hash=row.prompt_hash,
            redacted_prompt=row.redacted_prompt,
            provider=row.provider,
            status_code=row.status_code,
            latency_ms=row.latency_ms,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            estimated_cost_usd=row.estimated_cost_usd,
            cache_hit=row.cache_hit,
            retry_count=row.retry_count,
        )
        for row in rows
    ]


@app.get("/v1/circuits")
async def circuit_states() -> Any:
    """Return the live state of all registered circuit breakers."""
    return get_circuit_states()


@app.post("/v1/simulate")
async def simulate(request: SimulationRequest) -> Any:
    """Run discrete-event simulation across candidate policies."""
    if request.source == "postgres":
        async with AsyncSessionLocal() as session:
            events = await load_postgres_events(session, request.log_limit)
    else:
        events = generate_synthetic_events(
            count=request.synthetic_count,
            unique_prompts=request.unique_prompts,
            rps=request.rps,
            seed=request.seed,
        )

    results = [
        simulate_policy(events, policy, seed=request.seed)
        for policy in request.policies
    ]

    return {
        "source": request.source,
        "event_count": len(events),
        "results": results,
    }


@app.post("/v1/simulate/validate")
async def simulate_validate() -> Any:
    """Validate simulator predictions against live POST /v1/chat execution."""
    run_id = uuid.uuid4().hex[:8]
    prompt_prefix = f"validate-{run_id}"
    base_key = f"validate-key-{run_id}"

    events = generate_synthetic_events(
        count=100,
        unique_prompts=20,
        rps=10.0,
        seed=42,
        prompt_prefix=prompt_prefix,
        api_key=base_key,
    )

    policy = SimulationPolicy(
        name="live-validation",
        cache_ttl_seconds=settings.CACHE_TTL_SECONDS,
        rate_limit_capacity=1000,
        rate_limit_refill_per_sec=1000.0,
        daily_token_quota=10_000_000,
        primary_error_rate=0.0,
    )
    sim_result = simulate_policy(events, policy, seed=42)

    live_latencies: list[float] = []
    live_cache_hits = 0
    live_total_cost = 0.0
    live_baseline_cost = 0.0

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for idx, ev in enumerate(events):
            # Rotate fresh key every 8 requests so default rate limit (capacity=10) does not 429
            req_key = f"{base_key}-{idx // 8}"
            resp = await client.post(
                "/v1/chat",
                headers={"X-API-Key": req_key},
                json={
                    "prompt": ev.prompt,
                    "latency_ms": 150,
                    "error_rate": 0.0,
                    "use_mock": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            live_latencies.append(float(data["latency_ms"]))
            if data["cache_hit"]:
                live_cache_hits += 1

            p_tok = int(data["prompt_tokens"])
            c_tok = int(data["completion_tokens"])
            prov = data["provider"]
            live_baseline_cost += estimate_cost_usd(prov, p_tok, c_tok, cache_hit=False)
            live_total_cost += estimate_cost_usd(prov, p_tok, c_tok, cache_hit=bool(data["cache_hit"]))

    total = len(events)
    live_cache_hit_rate_pct = round((live_cache_hits / total) * 100.0, 2) if total else 0.0
    live_cost_savings_pct = (
        round(((live_baseline_cost - live_total_cost) / live_baseline_cost) * 100.0, 2)
        if live_baseline_cost > 0
        else 0.0
    )
    live_p95_latency_ms = _percentile(sorted(live_latencies), 95.0)

    predicted = {
        "cache_hit_rate_pct": sim_result["cache_hit_rate_pct"],
        "cost_savings_pct": sim_result["cost_savings_pct"],
        "p95_latency_ms": sim_result["p95_latency_ms"],
    }
    live_actual = {
        "cache_hit_rate_pct": live_cache_hit_rate_pct,
        "cost_savings_pct": live_cost_savings_pct,
        "p95_latency_ms": live_p95_latency_ms,
    }

    def _rel_or_abs_err(pred: float, actual: float) -> float:
        if actual == 0.0:
            return round(abs(pred - actual), 2)
        return round(abs(pred - actual) / abs(actual) * 100.0, 2)

    prediction_error_pct = {
        "cache_hit_rate_pct": round(abs(predicted["cache_hit_rate_pct"] - live_actual["cache_hit_rate_pct"]), 2),
        "cost_savings_pct": round(abs(predicted["cost_savings_pct"] - live_actual["cost_savings_pct"]), 2),
        "p95_latency_ms": _rel_or_abs_err(predicted["p95_latency_ms"], live_actual["p95_latency_ms"]),
    }

    return {
        "run_id": run_id,
        "total_requests": total,
        "predicted": predicted,
        "live_actual": live_actual,
        "prediction_error_pct": prediction_error_pct,
    }
