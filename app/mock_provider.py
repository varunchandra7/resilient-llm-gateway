"""
mock_provider.py — simulates an upstream LLM provider.

Mimics:
  - variable latency (latency_ms)
  - random HTTP 429 (rate-limit) or 500 (server error) failures (error_rate)
  - token counting (approximated by word-split)
  - streaming word-by-word chunks (stream_mock_response)
"""
import asyncio
import random
from collections.abc import AsyncGenerator

import httpx


async def generate_mock_response(
    prompt: str,
    latency_ms: int = 150,
    error_rate: float = 0.0,
) -> dict:
    """Simulate an async LLM provider call.

    Args:
        prompt:      The user prompt string.
        latency_ms:  Artificial delay in milliseconds before returning.
        error_rate:  Probability [0.0, 1.0] that the call raises an error.
                     Half the errors are 429s, half are 500s.

    Returns:
        dict with keys: text, prompt_tokens, completion_tokens

    Raises:
        httpx.HTTPStatusError: on simulated 429 or 500 responses.
    """
    # Simulate network / model latency
    await asyncio.sleep(latency_ms / 1000)

    # Simulate provider errors
    if error_rate > 0.0 and random.random() < error_rate:
        status_code = 429 if random.random() < 0.5 else 500
        raise httpx.HTTPStatusError(
            message=f"Mock provider error {status_code}",
            request=httpx.Request("POST", "http://mock-provider/v1/chat"),
            response=httpx.Response(status_code),
        )

    # Build a fake response
    completion_text = f"This is a mock response to: '{prompt}'"

    # Approximate token counts (1 token ≈ 1 word for the mock)
    prompt_tokens = len(prompt.split())
    completion_tokens = len(completion_text.split())

    return {
        "text": completion_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def stream_mock_response(
    prompt: str,
    latency_ms: int = 150,
    error_rate: float = 0.0,
) -> AsyncGenerator[dict, None]:
    """Simulate an async streaming LLM provider call yielding word chunks."""
    await asyncio.sleep(latency_ms / 1000)

    if error_rate > 0.0 and random.random() < error_rate:
        status_code = 429 if random.random() < 0.5 else 500
        raise httpx.HTTPStatusError(
            message=f"Mock provider error {status_code}",
            request=httpx.Request("POST", "http://mock-provider/v1/chat"),
            response=httpx.Response(status_code),
        )

    completion_text = f"This is a mock response to: '{prompt}'"
    words = completion_text.split()
    prompt_tokens = len(prompt.split())
    completion_tokens = len(words)

    for idx, word in enumerate(words):
        piece = word if idx == 0 else f" {word}"
        is_last = idx == len(words) - 1
        yield {
            "text": piece,
            "prompt_tokens": prompt_tokens if is_last else None,
            "completion_tokens": completion_tokens if is_last else None,
        }
        if not is_last:
            await asyncio.sleep(0.02)
