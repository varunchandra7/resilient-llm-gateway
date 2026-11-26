"""
pii.py — PII detection, redaction, and prompt hashing.

Redaction rules (applied in order, non-overlapping):
  - Email addresses            → [REDACTED_EMAIL]
  - API keys: sk-...           → [REDACTED_API_KEY]
  - API keys: AIza...          → [REDACTED_API_KEY]
  - 16-digit card numbers      → [REDACTED_CARD]
  - Phone numbers (US + intl)  → [REDACTED_PHONE]

hash_prompt returns a hex SHA-256 of the *raw* text (before redaction),
so identical prompts are deduplicatable without exposing PII in the DB.
"""
import hashlib
import re

# ---------------------------------------------------------------------------
# Compiled patterns — compiled once at import time for performance
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email addresses
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
        "[REDACTED_EMAIL]",
    ),
    # OpenAI-style API keys: sk- followed by 20+ alphanumeric/dash chars
    (
        re.compile(r"\bsk-[A-Za-z0-9\-_]{20,}", re.IGNORECASE),
        "[REDACTED_API_KEY]",
    ),
    # Google API keys: AIza followed by 35 alphanumeric chars
    (
        re.compile(r"\bAIza[A-Za-z0-9\-_]{35}", re.IGNORECASE),
        "[REDACTED_API_KEY]",
    ),
    # 16-digit credit/debit card numbers (optionally space- or dash-separated)
    (
        re.compile(r"\b(?:\d[ \-]?){15}\d\b"),
        "[REDACTED_CARD]",
    ),
    # Phone numbers: international (+1 …) and common US formats
    # Matches: +1-800-555-1234, (800) 555-1234, 800.555.1234, 8005551234
    (
        re.compile(
            r"(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{4}\b"
        ),
        "[REDACTED_PHONE]",
    ),
]


def redact_pii(text: str) -> str:
    """Return *text* with all recognised PII replaced by [REDACTED_*] tokens.

    Each pattern is applied to the result of the previous one so that
    partial matches left by one rule are caught by the next.

    Args:
        text: Raw input string (prompt, completion, etc.)

    Returns:
        Redacted string safe for structured logging / DB storage.
    """
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def hash_prompt(text: str) -> str:
    """Return a hex-encoded SHA-256 digest of *text*.

    The hash is computed on the **raw** (pre-redaction) text so that
    semantically identical prompts map to the same hash regardless of
    when redaction runs.

    Args:
        text: Raw prompt string.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
