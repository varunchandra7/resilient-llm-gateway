"""
models.py — SQLAlchemy ORM models.

Table: request_logs
  Stores every gateway request, with PII already redacted.
  Indexed on created_at (range scans) and prompt_hash (deduplication lookups).

Step 4 addition: retry_count column (zero for cache hits and rate-limit blocks).
  The column is added to an *existing* table via ALTER TABLE ... ADD COLUMN IF NOT EXISTS
  executed in main.py lifespan, so no manual SQL or migration tool is needed.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class RequestLog(Base):
    """One row per request handled by POST /v1/chat."""

    __tablename__ = "request_logs"

    # Primary key — UUID v4 generated in Python (no DB round-trip needed)
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Timestamp — stored with timezone, indexed for range queries
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # SHA-256 of the *raw* prompt — indexed for deduplication lookups
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Redacted version of the prompt — safe to store
    redacted_prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # Which upstream provider handled the request (e.g. "mock-primary", "mock-fallback")
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")

    # HTTP status code returned to the caller (200, 429, 500, 503, 504, …)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)

    # Wall-clock latency measured in main.py (float for sub-ms precision)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # Token counts from the provider (NULL on error responses)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Rough cost estimate — populate in later steps when real pricing is known
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cache hit flag
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Number of retries consumed before a successful (or final failed) response.
    # 0 for cache hits, rate-limit blocks, and first-attempt successes.
    # Added in Step 4 — ALTER TABLE migration runs in lifespan startup.
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Composite indexes defined at table level for clarity
    __table_args__ = (
        Index("ix_request_logs_created_at", "created_at"),
        Index("ix_request_logs_prompt_hash", "prompt_hash"),
    )

    def __repr__(self) -> str:
        return (
            f"<RequestLog id={self.id} status={self.status_code} "
            f"provider={self.provider} retries={self.retry_count} "
            f"latency={self.latency_ms:.1f}ms>"
        )
