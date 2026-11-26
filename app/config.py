"""
config.py — Application settings via pydantic-settings.

Values are read from environment variables or a .env file in the project root.
Add a .env file (git-ignored) to override defaults in development:

    DATABASE_URL=postgresql+asyncpg://gateway:gateway@localhost:5432/gateway
    REDIS_URL=redis://localhost:6379/0
    ENV=dev
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # Postgres                                                             #
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = (
        "postgresql+asyncpg://gateway:gateway@localhost:5432/gateway"
    )

    # ------------------------------------------------------------------ #
    # Redis                                                                #
    # ------------------------------------------------------------------ #
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 300

    # ------------------------------------------------------------------ #
    # Token-bucket rate limiting (per API key)                            #
    # ------------------------------------------------------------------ #
    RATE_LIMIT_CAPACITY: int = 10
    RATE_LIMIT_REFILL_PER_SEC: float = 2.0

    # ------------------------------------------------------------------ #
    # Daily token quota (per API key, resets at midnight UTC)             #
    # ------------------------------------------------------------------ #
    DAILY_TOKEN_QUOTA: int = 100_000

    # ------------------------------------------------------------------ #
    # Retry / timeout policy (per provider attempt)                       #
    # ------------------------------------------------------------------ #
    # Maximum number of retries on the *same* provider before falling over
    MAX_RETRIES: int = 2

    # Base delay for exponential-backoff full-jitter (milliseconds)
    RETRY_BASE_DELAY_MS: int = 100

    # Upper cap on the jitter window (milliseconds)
    RETRY_MAX_DELAY_MS: int = 1000

    # Hard per-attempt deadline passed to asyncio.wait_for()
    REQUEST_DEADLINE_SECONDS: float = 4.0

    # ------------------------------------------------------------------ #
    # Circuit breaker                                                      #
    # ------------------------------------------------------------------ #
    # Consecutive failures that trip a circuit from CLOSED → OPEN
    CB_FAILURE_THRESHOLD: int = 5

    # Seconds an OPEN circuit waits before allowing a single probe attempt
    CB_RECOVERY_TIMEOUT_SECONDS: float = 30.0

    # ------------------------------------------------------------------ #
    # Gemini LLM Provider                                                  #
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # ------------------------------------------------------------------ #
    # General                                                              #
    # ------------------------------------------------------------------ #
    ENV: str = "dev"


# Module-level singleton — import this everywhere instead of re-instantiating
settings = Settings()
