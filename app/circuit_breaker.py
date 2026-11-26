"""
circuit_breaker.py — Async-safe 3-state circuit breaker per provider.

States
------
  CLOSED     Normal operation. Failures are counted; success resets the counter.
  OPEN       All calls fail fast (CircuitOpenError) without touching the provider.
             Transitions to HALF_OPEN after CB_RECOVERY_TIMEOUT_SECONDS.
  HALF_OPEN  One probe call is allowed through (the asyncio.Lock is held).
             Probe success  → CLOSED (failure counter reset).
             Probe failure  → back to OPEN (opened_at refreshed).

Concurrency
-----------
Each CircuitBreaker has its own asyncio.Lock.  All state reads/writes happen
inside that lock.  While the HALF_OPEN probe is running the lock is held, so
any concurrent request sees the breaker as OPEN and fails fast — preventing a
thundering herd of probe calls.

Usage
-----
    cb = get_breaker("mock-primary")

    async with cb:                      # acquires permission or raises CircuitOpenError
        result = await some_provider()
        # on success the context manager calls record_success automatically

    # on exception the caller is responsible for calling cb.record_failure()
    # (resilience.py does this in its except block)

API
---
    get_breaker(name) → CircuitBreaker     singleton per name
    get_circuit_states() → dict            snapshot of all breakers (for /v1/circuits)
"""
import asyncio
import logging
import time
from enum import Enum

from app.config import settings

logger = logging.getLogger(__name__)


class CBState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit breaker is OPEN."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Circuit breaker OPEN for provider '{provider}'")


class CircuitBreaker:
    """Async-safe 3-state circuit breaker for a single named provider."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._state = CBState.CLOSED
        self._failure_count: int = 0
        self._opened_at: float | None = None   # monotonic timestamp when tripped
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public state accessors (lock-free reads — snapshot only)
    # ------------------------------------------------------------------

    @property
    def state(self) -> CBState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def opened_at(self) -> float | None:
        return self._opened_at

    # ------------------------------------------------------------------
    # Core logic (always called with self._lock held)
    # ------------------------------------------------------------------

    def _should_allow_probe(self) -> bool:
        """True when enough time has passed to transition OPEN → HALF_OPEN."""
        if self._opened_at is None:
            return False
        return (time.monotonic() - self._opened_at) >= settings.CB_RECOVERY_TIMEOUT_SECONDS

    async def _acquire(self) -> None:
        """Check state and block / raise as appropriate.

        Called at the top of __aenter__ while holding self._lock.
        """
        if self._state == CBState.CLOSED:
            return  # normal path

        if self._state == CBState.OPEN:
            if self._should_allow_probe():
                self._state = CBState.HALF_OPEN
                logger.info(
                    "CircuitBreaker[%s]: OPEN → HALF_OPEN (sending probe).", self.name
                )
                return  # allow the probe; lock stays held until __aexit__
            raise CircuitOpenError(self.name)

        if self._state == CBState.HALF_OPEN:
            # Another coroutine is already probing — fail fast
            raise CircuitOpenError(self.name)

    # ------------------------------------------------------------------
    # Async context manager — wrap a single provider call
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CircuitBreaker":
        await self._lock.acquire()
        try:
            await self._acquire()
        except Exception:
            self._lock.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is None:
                # Clean exit → record success
                self._record_success_locked()
            # Failures are recorded explicitly by the caller via record_failure()
            # so the breaker knows the HTTP status code context.
        finally:
            self._lock.release()
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Success / failure recording (called from resilience.py)
    # ------------------------------------------------------------------

    def _record_success_locked(self) -> None:
        """Reset breaker to CLOSED.  Called inside __aexit__ with lock held."""
        if self._state != CBState.CLOSED:
            logger.info(
                "CircuitBreaker[%s]: %s → CLOSED (probe succeeded).",
                self.name, self._state,
            )
        self._state = CBState.CLOSED
        self._failure_count = 0
        self._opened_at = None

    async def record_failure(self) -> None:
        """Increment failure count and open the circuit if threshold reached.

        Called by resilience.py in the except block after a provider error.
        Acquires the lock independently (not inside __aexit__).
        """
        async with self._lock:
            self._failure_count += 1

            if self._state == CBState.HALF_OPEN:
                # Probe failed → return to OPEN immediately
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "CircuitBreaker[%s]: HALF_OPEN → OPEN (probe failed, failures=%d).",
                    self.name, self._failure_count,
                )
            elif self._failure_count >= settings.CB_FAILURE_THRESHOLD:
                prev = self._state
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "CircuitBreaker[%s]: %s → OPEN (failures=%d >= threshold=%d).",
                    self.name, prev, self._failure_count, settings.CB_FAILURE_THRESHOLD,
                )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serialisable state snapshot (no lock — best-effort)."""
        seconds_open: float | None = None
        if self._opened_at is not None:
            seconds_open = round(time.monotonic() - self._opened_at, 1)

        return {
            "provider": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "seconds_open": seconds_open,
            "recovery_timeout_seconds": settings.CB_RECOVERY_TIMEOUT_SECONDS,
        }


# ---------------------------------------------------------------------------
# Registry — one breaker per provider name, created on first access
# ---------------------------------------------------------------------------

_registry: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    """Return (creating if necessary) the CircuitBreaker for *name*."""
    if name not in _registry:
        _registry[name] = CircuitBreaker(name)
    return _registry[name]


def get_circuit_states() -> dict[str, list[dict]]:
    """Return a snapshot of every registered circuit breaker."""
    return {"circuits": [cb.snapshot() for cb in _registry.values()]}
