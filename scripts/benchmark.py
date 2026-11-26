"""
scripts/benchmark.py — Automated benchmark & resilience verification runner.

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --url http://localhost:8000
"""
import argparse
import asyncio
import math
import random
import time
import uuid

import httpx


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(sorted_vals[lo], 2)
    return round(sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo]), 2)


async def run_suite_1_cache_concurrency(client: httpx.AsyncClient, run_id: str) -> dict:
    """Suite 1: 300 concurrent requests (concurrency=25) across 30 Zipf prompts."""
    total_requests = 300
    concurrency = 25
    unique_prompts = 30

    rng = random.Random(42)
    weights = [1.0 / (i + 1) for i in range(unique_prompts)]
    prompts = [
        f"bench-{run_id}-s1-{i}: summarize distributed gateway design pattern {i}"
        for i in range(unique_prompts)
    ]
    trace = rng.choices(prompts, weights=weights, k=total_requests)

    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    cache_hits = 0
    errors_5xx = 0
    errors_429 = 0

    async def _worker(idx: int, prompt: str) -> None:
        nonlocal cache_hits, errors_5xx, errors_429
        # Rotate fresh API key every 8 requests so default bucket capacity (10) does not 429
        api_key = f"bench-{run_id}-s1-key-{idx // 8}"
        async with sem:
            t0 = time.perf_counter()
            resp = await client.post(
                "/v1/chat",
                headers={"X-API-Key": api_key},
                json={
                    "prompt": prompt,
                    "latency_ms": 120,
                    "error_rate": 0.0,
                    "use_mock": True,
                },
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            if resp.status_code >= 500:
                errors_5xx += 1
            elif resp.status_code == 429:
                errors_429 += 1
            elif resp.status_code == 200:
                data = resp.json()
                if data.get("cache_hit"):
                    cache_hits += 1

    t_start = time.perf_counter()
    await asyncio.gather(*[_worker(i, p) for i, p in enumerate(trace)])
    total_s = time.perf_counter() - t_start

    sorted_lat = sorted(latencies)
    rps = round(total_requests / total_s, 2) if total_s > 0 else 0.0
    hit_rate = round((cache_hits / total_requests) * 100, 2)
    err_5xx_pct = round((errors_5xx / total_requests) * 100, 2)

    return {
        "name": "Suite 1: Steady-State & Cache",
        "requests": total_requests,
        "concurrency": concurrency,
        "rps": rps,
        "cache_hit_pct": hit_rate,
        "p50_ms": _percentile(sorted_lat, 50.0),
        "p95_ms": _percentile(sorted_lat, 95.0),
        "p99_ms": _percentile(sorted_lat, 99.0),
        "err_5xx_pct": err_5xx_pct,
        "err_429": errors_429,
    }


async def run_suite_2_outage(
    client: httpx.AsyncClient, run_id: str, error_rate: float
) -> dict:
    """Suite 2: 100 concurrent requests (concurrency=15) with primary error injection."""
    total_requests = 100
    concurrency = 15
    sem = asyncio.Semaphore(concurrency)

    latencies: list[float] = []
    fallback_count = 0
    errors_5xx = 0
    tag = f"err{int(error_rate * 100)}"

    async def _worker(idx: int) -> None:
        nonlocal fallback_count, errors_5xx
        api_key = f"bench-{run_id}-s2-{tag}-key-{idx // 8}"
        # Unique prompt per request forces cache miss so resilience engine is exercised
        prompt = f"bench-{run_id}-s2-{tag}-prompt-{idx}"
        async with sem:
            t0 = time.perf_counter()
            resp = await client.post(
                "/v1/chat",
                headers={"X-API-Key": api_key},
                json={
                    "prompt": prompt,
                    "latency_ms": 100,
                    "error_rate": error_rate,
                    "use_mock": True,
                },
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            if resp.status_code >= 500:
                errors_5xx += 1
            elif resp.status_code == 200:
                data = resp.json()
                if data.get("provider") == "mock-fallback":
                    fallback_count += 1

    t_start = time.perf_counter()
    await asyncio.gather(*[_worker(i) for i in range(total_requests)])
    total_s = time.perf_counter() - t_start

    circuits_resp = await client.get("/v1/circuits")
    circuits = circuits_resp.json().get("circuits", []) if circuits_resp.status_code == 200 else []
    primary_state = next(
        (c["state"] for c in circuits if c["provider"] == "mock-primary"),
        "UNKNOWN",
    )

    sorted_lat = sorted(latencies)
    return {
        "name": f"Suite 2: Outage ({int(error_rate * 100)}% Primary Err)",
        "requests": total_requests,
        "concurrency": concurrency,
        "rps": round(total_requests / total_s, 2) if total_s > 0 else 0.0,
        "fallback_pct": round((fallback_count / total_requests) * 100, 2),
        "primary_cb_state": primary_state,
        "p50_ms": _percentile(sorted_lat, 50.0),
        "p95_ms": _percentile(sorted_lat, 95.0),
        "p99_ms": _percentile(sorted_lat, 99.0),
        "err_5xx_pct": round((errors_5xx / total_requests) * 100, 2),
    }


async def run_suite_3_simulator(client: httpx.AsyncClient) -> dict:
    """Suite 3: Call POST /v1/simulate/validate and capture accuracy."""
    resp = await client.post("/v1/simulate/validate", timeout=60.0)
    resp.raise_for_status()
    return resp.json()


async def main(base_url: str) -> None:
    run_id = uuid.uuid4().hex[:8]
    print(f"# Resilient LLM Gateway Benchmark (`run_id={run_id}`, target=`{base_url}`)\n")

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        s1 = await run_suite_1_cache_concurrency(client, run_id)
        s2_30 = await run_suite_2_outage(client, run_id, error_rate=0.3)
        s2_100 = await run_suite_2_outage(client, run_id, error_rate=1.0)
        s3 = await run_suite_3_simulator(client)

    print("## Suite 1 & 2 — Load, Caching & Outage Resilience\n")
    print(
        "| Suite | Requests | Concurrency | Throughput (req/s) | Cache Hit / Fallback | "
        "p50 (ms) | p95 (ms) | p99 (ms) | 5xx Error % |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {s1['name']} | {s1['requests']} | {s1['concurrency']} | {s1['rps']} | "
        f"{s1['cache_hit_pct']}% cache hit | {s1['p50_ms']} | {s1['p95_ms']} | "
        f"{s1['p99_ms']} | {s1['err_5xx_pct']}% |"
    )
    print(
        f"| {s2_30['name']} | {s2_30['requests']} | {s2_30['concurrency']} | {s2_30['rps']} | "
        f"{s2_30['fallback_pct']}% fallback (CB: {s2_30['primary_cb_state']}) | "
        f"{s2_30['p50_ms']} | {s2_30['p95_ms']} | {s2_30['p99_ms']} | {s2_30['err_5xx_pct']}% |"
    )
    print(
        f"| {s2_100['name']} | {s2_100['requests']} | {s2_100['concurrency']} | {s2_100['rps']} | "
        f"{s2_100['fallback_pct']}% fallback (CB: {s2_100['primary_cb_state']}) | "
        f"{s2_100['p50_ms']} | {s2_100['p95_ms']} | {s2_100['p99_ms']} | {s2_100['err_5xx_pct']}% |"
    )

    pred = s3["predicted"]
    live = s3["live_actual"]
    err = s3["prediction_error_pct"]

    print("\n## Suite 3 — Discrete-Event Simulator vs. Live Execution\n")
    print("| Metric | Predicted | Live Actual | Error |")
    print("|---|---:|---:|---:|")
    print(
        f"| Cache Hit Rate (%) | {pred['cache_hit_rate_pct']}% | "
        f"{live['cache_hit_rate_pct']}% | {err['cache_hit_rate_pct']} pp |"
    )
    print(
        f"| Cost Savings (%) | {pred['cost_savings_pct']}% | "
        f"{live['cost_savings_pct']}% | {err['cost_savings_pct']} pp |"
    )
    print(
        f"| p95 Latency (ms) | {pred['p95_latency_ms']} ms | "
        f"{live['p95_latency_ms']} ms | {err['p95_latency_ms']}% |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Resilient LLM Gateway benchmark suites.")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway base URL")
    args = parser.parse_args()
    asyncio.run(main(args.url))
