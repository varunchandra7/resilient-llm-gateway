"""
pricing.py — Token cost accounting per provider.

Rates (per 1M tokens):
  - gemini:                        $0.75 input  / $3.75 output
  - mock-primary / mock-fallback:  $0.30 input  / $1.50 output
  - cache hit:                     $0.00
"""

_PROVIDER_RATES_PER_1M: dict[str, tuple[float, float]] = {
    "gemini": (0.75, 3.75),
    "mock-primary": (0.30, 1.50),
    "mock-fallback": (0.30, 1.50),
    "mock": (0.30, 1.50),
}


def estimate_cost_usd(
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit: bool,
) -> float:
    """Return estimated USD cost for a request, rounded to 8 decimal places."""
    if cache_hit:
        return 0.0

    rates = _PROVIDER_RATES_PER_1M.get(provider)
    if rates is None:
        return 0.0

    in_rate, out_rate = rates
    cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000
    return round(cost, 8)
