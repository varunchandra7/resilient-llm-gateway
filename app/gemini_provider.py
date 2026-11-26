"""
gemini_provider.py — Real Gemini LLM integration via the official google-genai SDK.

Provides:
  - generate_gemini_response(prompt)  → dict {text, prompt_tokens, completion_tokens}
  - stream_gemini_response(prompt)    → async generator yielding chunk dicts

Error normalization:
  Upstream google-genai APIError / transport exceptions are converted into
  httpx.HTTPStatusError with HTTP-style status codes (429, 500, 503, etc.)
  so resilience.py retries and circuit breakers handle them uniformly.
"""
import logging
from collections.abc import AsyncGenerator

import httpx
from google import genai
from google.genai import errors as genai_errors

from app.config import settings

logger = logging.getLogger(__name__)

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com"


def _get_client() -> genai.Client:
    """Create a Gemini async-capable client using the configured API key."""
    if not settings.GEMINI_API_KEY:
        raise httpx.HTTPStatusError(
            message="GEMINI_API_KEY is not configured.",
            request=httpx.Request("POST", _GEMINI_ENDPOINT),
            response=httpx.Response(503),
        )
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _normalize_gemini_error(exc: Exception) -> httpx.HTTPStatusError:
    """Convert a google-genai SDK or network exception into httpx.HTTPStatusError."""
    status_code = 503

    if isinstance(exc, genai_errors.APIError):
        raw_code = getattr(exc, "code", None)
        if isinstance(raw_code, int) and 400 <= raw_code <= 599:
            status_code = raw_code
        else:
            msg = str(exc).lower()
            if "429" in msg or "resource_exhausted" in msg or "quota" in msg:
                status_code = 429
            elif "500" in msg or "internal" in msg:
                status_code = 500
            else:
                status_code = 503
    else:
        raw_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(raw_code, int) and 400 <= raw_code <= 599:
            status_code = raw_code

    return httpx.HTTPStatusError(
        message=f"Gemini provider error ({status_code}): {exc}",
        request=httpx.Request("POST", _GEMINI_ENDPOINT),
        response=httpx.Response(status_code),
    )


def _extract_usage(response_obj: object) -> tuple[int | None, int | None]:
    """Extract (prompt_tokens, completion_tokens) from usage_metadata if available."""
    usage = getattr(response_obj, "usage_metadata", None)
    if usage is None:
        return None, None
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    completion_tokens = getattr(usage, "candidates_token_count", None)
    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )


async def generate_gemini_response(prompt: str) -> dict:
    """Call Gemini non-streaming API and return normalized response + token counts."""
    client = _get_client()
    try:
        response = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
    except httpx.HTTPStatusError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _normalize_gemini_error(exc) from exc

    text = getattr(response, "text", None) or ""
    prompt_tokens, completion_tokens = _extract_usage(response)

    if prompt_tokens is None:
        prompt_tokens = len(prompt.split())
    if completion_tokens is None:
        completion_tokens = len(text.split())

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def stream_gemini_response(prompt: str) -> AsyncGenerator[dict, None]:
    """Stream chunks from Gemini's async streaming API.

    Yields dicts of the form:
      {"text": str, "prompt_tokens": int | None, "completion_tokens": int | None}
    """
    client = _get_client()
    try:
        stream = await client.aio.models.generate_content_stream(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        async for chunk in stream:
            chunk_text = getattr(chunk, "text", None) or ""
            prompt_tokens, completion_tokens = _extract_usage(chunk)
            if chunk_text or prompt_tokens is not None or completion_tokens is not None:
                yield {
                    "text": chunk_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
    except httpx.HTTPStatusError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _normalize_gemini_error(exc) from exc
