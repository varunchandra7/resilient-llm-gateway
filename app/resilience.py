"""
resilience.py — Retry, full-jitter backoff, circuit-breaker, and provider fallback.

Provider chain (tried in order)
--------------------------------
  When GEMINI_API_KEY is set and use_mock=False:
    1. "gemini"         — real Google Gemini API via google-genai SDK
    2. "mock-fallback"  — reliable local backup (latency=120ms, error_rate=0.0)

  When GEMINI_API_KEY is unset or use_mock=True:
    1. "mock-primary"   — uses caller-supplied latency_ms and error_rate
    2. "mock-fallback"  — reliable local backup (latency=120ms, error_rate=0.0)

Retry policy (per provider)
-----------------------------
  - Retryable errors: asyncio.TimeoutError, HTTP 429, HTTP 5xx
  - Non-retryable:    HTTP 4xx (excluding 429) — raised immediately
  - Attempts:         MAX_RETRIES + 1 total (attempt 0 … MAX_RETRIES)
  - Sleep:            full-jitter = random.uniform(0, min(MAX_DELAY, BASE * 2^attempt))
  - Hard deadline:    REQUEST_DEADLINE_SECONDS per attempt via asyncio.wait_for()
"""
import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator

import httpx
from fastapi import HTTPException

from app.circuit_breaker import CircuitOpenError, get_breaker
from app.config import settings
from app.gemini_provider import generate_gemini_response, stream_gemini_response
from app.mock_provider import generate_mock_response, stream_mock_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """True for transient errors that are worth retrying."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


def _jitter_sleep_seconds(attempt: int) -> float:
    """Full-jitter backoff: uniform(0, min(MAX_DELAY, BASE * 2^attempt))."""
    cap_ms = min(
        settings.RETRY_MAX_DELAY_MS,
        settings.RETRY_BASE_DELAY_MS * (2 ** attempt),
    )
    return random.uniform(0, cap_ms) / 1000.0


def _build_provider_chain(use_mock: bool = False) -> list[dict]:
    """Return the ordered provider chain for the current request."""
    if settings.GEMINI_API_KEY and not use_mock:
        return [
            {
                "name": "gemini",
                "fixed_latency_ms": None,
                "fixed_error_rate": None,
            },
            {
                "name": "mock-fallback",
                "fixed_latency_ms": 120,
                "fixed_error_rate": 0.0,
            },
        ]

    return [
        {
            "name": "mock-primary",
            "fixed_latency_ms": None,
            "fixed_error_rate": None,
        },
        {
            "name": "mock-fallback",
            "fixed_latency_ms": 120,
            "fixed_error_rate": 0.0,
        },
    ]


async def _invoke_provider(
    name: str,
    prompt: str,
    latency_ms: int,
    error_rate: float,
) -> dict:
    """Dispatch a single non-streaming call to the named provider."""
    if name == "gemini":
        return await generate_gemini_response(prompt=prompt)
    return await generate_mock_response(
        prompt=prompt,
        latency_ms=latency_ms,
        error_rate=error_rate,
    )


def _stream_provider(
    name: str,
    prompt: str,
    latency_ms: int,
    error_rate: float,
) -> AsyncGenerator[dict, None]:
    """Return the async chunk generator for the named provider."""
    if name == "gemini":
        return stream_gemini_response(prompt=prompt)
    return stream_mock_response(
        prompt=prompt,
        latency_ms=latency_ms,
        error_rate=error_rate,
    )


# ---------------------------------------------------------------------------
# Non-streaming entry point
# ---------------------------------------------------------------------------

async def call_with_resilience(
    prompt: str,
    latency_ms: int,
    error_rate: float,
    use_mock: bool = False,
) -> tuple[dict, str, int]:
    """Attempt providers in order with retries, jitter, and circuit breaking.

    Returns:
        (response_data, provider_name, total_retry_count)

    Raises:
        httpx.HTTPStatusError: for non-retryable 4xx errors.
        HTTPException(503):    if every provider in the chain fails.
    """
    total_retries = 0
    providers = _build_provider_chain(use_mock=use_mock)

    for provider_cfg in providers:
        name: str = provider_cfg["name"]
        p_latency = (
            provider_cfg["fixed_latency_ms"]
            if provider_cfg["fixed_latency_ms"] is not None
            else latency_ms
        )
        p_error_rate = (
            provider_cfg["fixed_error_rate"]
            if provider_cfg["fixed_error_rate"] is not None
            else error_rate
        )

        cb = get_breaker(name)

        # Fast-fail if the circuit is OPEN and recovery window hasn't elapsed
        if cb.state.value == "OPEN":
            if cb.opened_at is not None and (
                time.monotonic() - cb.opened_at < settings.CB_RECOVERY_TIMEOUT_SECONDS
            ):
                logger.info(
                    "Skipping provider '%s' — circuit is OPEN, falling over.", name
                )
                continue

        last_exc: Exception | None = None

        for attempt in range(settings.MAX_RETRIES + 1):
            if attempt > 0:
                sleep_s = _jitter_sleep_seconds(attempt - 1)
                logger.info(
                    "Provider '%s' attempt %d/%d — sleeping %.0fms before retry.",
                    name, attempt + 1, settings.MAX_RETRIES + 1, sleep_s * 1000,
                )
                await asyncio.sleep(sleep_s)
                total_retries += 1

            try:
                async with cb:
                    result = await asyncio.wait_for(
                        _invoke_provider(
                            name=name,
                            prompt=prompt,
                            latency_ms=p_latency,
                            error_rate=p_error_rate,
                        ),
                        timeout=settings.REQUEST_DEADLINE_SECONDS,
                    )
                logger.info(
                    "Provider '%s' succeeded on attempt %d (total_retries=%d).",
                    name, attempt + 1, total_retries,
                )
                return result, name, total_retries

            except CircuitOpenError:
                logger.info(
                    "Provider '%s' circuit opened mid-flight — falling over.", name
                )
                last_exc = CircuitOpenError(name)
                break

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await cb.record_failure()

                if not _is_retryable(exc):
                    logger.warning(
                        "Provider '%s' returned non-retryable error: %s", name, exc
                    )
                    raise

                logger.warning(
                    "Provider '%s' attempt %d failed (%s) — %s.",
                    name,
                    attempt + 1,
                    type(exc).__name__,
                    "retrying" if attempt < settings.MAX_RETRIES else "giving up",
                )

        logger.warning(
            "Provider '%s' exhausted all %d attempt(s). Last error: %s. Trying fallback.",
            name, settings.MAX_RETRIES + 1, last_exc,
        )

    raise HTTPException(
        status_code=503,
        detail="All providers are unavailable. Please try again later.",
    )


# ---------------------------------------------------------------------------
# Streaming entry point (circuit-breaker + fallback aware)
# ---------------------------------------------------------------------------

async def stream_with_resilience(
    prompt: str,
    latency_ms: int,
    error_rate: float,
    use_mock: bool = False,
) -> AsyncGenerator[tuple[dict, str], None]:
    """Stream chunks from the ordered provider chain with circuit breaking.

    Yields:
        (chunk_dict, provider_name)
    """
    providers = _build_provider_chain(use_mock=use_mock)

    for provider_cfg in providers:
        name: str = provider_cfg["name"]
        p_latency = (
            provider_cfg["fixed_latency_ms"]
            if provider_cfg["fixed_latency_ms"] is not None
            else latency_ms
        )
        p_error_rate = (
            provider_cfg["fixed_error_rate"]
            if provider_cfg["fixed_error_rate"] is not None
            else error_rate
        )

        cb = get_breaker(name)

        if cb.state.value == "OPEN":
            if cb.opened_at is not None and (
                time.monotonic() - cb.opened_at < settings.CB_RECOVERY_TIMEOUT_SECONDS
            ):
                logger.info(
                    "Skipping streaming provider '%s' — circuit is OPEN, falling over.",
                    name,
                )
                continue

        yielded_any = False
        try:
            async with cb:
                async for chunk in _stream_provider(
                    name=name,
                    prompt=prompt,
                    latency_ms=p_latency,
                    error_rate=p_error_rate,
                ):
                    yielded_any = True
                    yield chunk, name
            return
        except CircuitOpenError:
            logger.info("Streaming provider '%s' circuit OPEN — falling over.", name)
            continue
        except Exception as exc:  # noqa: BLE001
            await cb.record_failure()
            if not _is_retryable(exc) or yielded_any:
                raise
            logger.warning(
                "Streaming provider '%s' failed before first chunk (%s) — falling over.",
                name, exc,
            )

    raise HTTPException(
        status_code=503,
        detail="All providers are unavailable. Please try again later.",
    )
