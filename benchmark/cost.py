"""Cost comparison — self-hosted vs cloud alternatives.

Uses performance numbers to compute $/1M tokens on different infrastructures.
"""
from __future__ import annotations

# Reference cloud pricing (as of April 2026)
PROVIDER_PRICING = {
    # $/1M tokens (input, output)
    "openai_gpt-4o":       {"input": 2.50,  "output": 10.00},
    "openai_gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "anthropic_sonnet_4.6":{"input": 3.00,  "output": 15.00},
    "anthropic_haiku_4.5": {"input": 1.00,  "output": 5.00},
    "together_llama_70b":  {"input": 0.88,  "output": 0.88},
    "together_llama_8b":   {"input": 0.20,  "output": 0.20},
    "fireworks_llama_70b": {"input": 0.90,  "output": 0.90},
    "fireworks_llama_8b":  {"input": 0.20,  "output": 0.20},
}

# DGX Spark cost model
SPARK_COST_PER_HOUR = 0.05   # $/hr electricity amortized
# (~400W × $0.13/kWh Montreal = $0.052/hr)


def compute_cost_comparison(perf: dict) -> dict:
    """Given performance metrics, compute $/1M tokens on various providers."""
    if not perf or "throughput_tok_s" not in perf:
        return {"error": "no performance data"}

    tok_per_sec = perf["throughput_tok_s"].get("single_mean") or 0
    if tok_per_sec <= 0:
        return {"error": "no throughput data"}

    # Self-hosted cost per 1M tokens:
    # $/hr / (3600 × tok/sec) × 1_000_000
    self_hosted_per_1m = (SPARK_COST_PER_HOUR / (3600 * tok_per_sec)) * 1_000_000

    comparison = []
    for provider, pricing in PROVIDER_PRICING.items():
        # Use output pricing as the fair comparison (fine-tune produces output)
        cloud_per_1m = pricing["output"]
        savings_factor = cloud_per_1m / self_hosted_per_1m if self_hosted_per_1m > 0 else float("inf")
        comparison.append({
            "provider": provider,
            "input_per_1m": pricing["input"],
            "output_per_1m": pricing["output"],
            "vs_self_hosted_factor": round(savings_factor, 1),
        })

    comparison.sort(key=lambda x: -x["vs_self_hosted_factor"])

    return {
        "self_hosted_per_1m_usd": round(self_hosted_per_1m, 4),
        "self_hosted_tok_per_sec": tok_per_sec,
        "hourly_infra_cost_usd": SPARK_COST_PER_HOUR,
        "cloud_comparison": comparison,
        "notes": (
            "Self-hosted cost assumes DGX Spark electricity at $0.05/hr "
            "(Montreal hydro pricing). Your actual compute cost approaches $0 "
            "once hardware is amortized."
        ),
    }


if __name__ == "__main__":
    import json
    # Demo calculation
    demo = compute_cost_comparison({
        "throughput_tok_s": {"single_mean": 85.0},
    })
    print(json.dumps(demo, indent=2))
