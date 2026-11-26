# Resilient LLM Gateway

A production-grade, fault-tolerant LLM API gateway built with **FastAPI**, **PostgreSQL 16**, **Redis 7**, and the **Google Gemini SDK (`google-genai`)**. Designed for high availability with atomic token-bucket rate limiting, fail-open caching, 3-state circuit breakers, SSE stream-teeing, PII redaction, and a discrete-event replay simulator.

## Architecture

```
                         ┌───────────────────────────────────────────┐
   Client                │        FastAPI Gateway (app/main.py)      │
   POST /v1/chat ───────►│                                           │
   (stream=true/false)   │  1. SHA-256 Hash + PII Redaction          │
                         │  2. Lua Token-Bucket & Daily Quota ──────►├──► Redis 7 (Fail-Open <250ms)
                         │  3. Prompt-Hash Cache Lookup ────────────►│
                         │  4. Resilience Engine:                    │
                         │     ├─ Retries + Full Jitter              │
                         │     ├─ 3-State Circuit Breaker            │
                         │     └─ Ordered Provider Fallback:         │
                         │         ├─ Primary: Gemini / Mock-Primary │
                         │         └─ Backup:  Mock-Fallback         │
                         │  5. Background Audit + Cost Log ─────────►├──► PostgreSQL 16
                         └───────────────────────────────────────────┘
```

## Architectural Trade-Offs

| Decision | Trade-Off | Rationale |
|---|---|---|
| **Fail-Open Redis (`0.25s` timeout)** | Temporarily bypasses cache & rate limits during a Redis outage | Availability over strict enforcement; requests never stall for 6s+ or return 5xx when Redis restarts |
| **Atomic Redis Lua Token Bucket** | Slightly more complex than `INCR` + `EXPIRE` | Eliminates TOCTOU race conditions under high concurrency |
| **Hash Before PII Redaction** | Raw prompt is hashed in-memory; only redacted text is persisted | Preserves exact cache-key deduplication while ensuring emails, cards, phones, and API keys never hit Postgres |
| **Async `BackgroundTasks` Logging** | Minimal risk of losing a log row if the container is `SIGKILL`ed mid-flush | Keeps Postgres I/O off the critical path so DB hiccups never inflate caller p99 latency |
| **Full-Jitter Exponential Backoff** | Variable retry wait times (`uniform(0, cap)`) | Prevents synchronized thundering-herd retry spikes against recovering upstream providers |
| **SSE Stream-Teeing** | Buffers streamed chunks in-memory until `[DONE]` | Enables real-time token streaming to the client while still populating Redis cache and Postgres cost logs |

---

## Quick Start (Docker Compose)

1. **Configure optional `.env`** (if testing live Gemini; omit to use the mock provider automatically):

   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-3.8-flash
   ```

2. **Start Postgres, Redis, and the hot-reloading API**:

   ```bash
   docker compose up -d --build
   ```

3. **Verify health**:

   ```bash
   curl http://localhost:8000/health
   # {"status":"ok"}
   ```

---

## Reproducible Benchmarks & Chaos Testing

### Run the Automated Benchmark Suite

Executes 3 suites (300-req Zipf cache concurrency, 30%/100% outage failover, and simulator-vs-live validation):

```bash
# Inside the API container
docker compose exec api python scripts/benchmark.py

# Or from host (if httpx is installed locally)
python scripts/benchmark.py --url http://localhost:8000
```

### Redis Fail-Open Chaos Test

Verify that stopping Redis fails open in `<250ms` with `200 OK` and zero unhandled errors:

```bash
docker compose stop redis
curl -w "\ntime_total: %{time_total}s\n" -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Chaos test during Redis outage", "use_mock": true}'
docker compose start redis
```

---

## API Reference

| Method & Path | Description |
|---|---|
| `GET /health` | Liveness probe |
| `POST /v1/chat` | Main gateway endpoint (`prompt`, `stream`, `use_mock`, `latency_ms`, `error_rate`, `X-API-Key` header) |
| `GET /v1/logs?limit=20` | Inspect recent PII-redacted Postgres `request_logs` with token counts & `estimated_cost_usd` |
| `GET /v1/circuits` | Live state (`CLOSED`, `OPEN`, `HALF_OPEN`) of all registered circuit breakers |
| `POST /v1/simulate` | Run discrete-event replay across candidate `SimulationPolicy` configs (`synthetic` or `postgres` trace) |
| `POST /v1/simulate/validate` | Run a 100-request synthetic trace through `simulate_policy()` and live `/v1/chat` to measure prediction error |
