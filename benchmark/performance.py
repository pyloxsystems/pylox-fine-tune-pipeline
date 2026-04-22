"""Measure throughput, TTFT, latency, concurrent capacity against vLLM endpoint."""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

STANDARD_PROMPT = (
    "Write a clear two-paragraph explanation of what fine-tuning a language model "
    "means and why a business might want to do it. Focus on practical benefits."
)


async def _single_request(endpoint: str, model: str, prompt: str, max_tokens: int = 200) -> dict:
    """Fire one request, time TTFT + total, count tokens."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    t0 = time.time()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        t_done = time.time()
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "body": r.text[:200]}
    data = r.json()

    content = data["choices"][0]["message"].get("content", "") or ""
    usage = data.get("usage", {})
    total_ms = (t_done - t0) * 1000
    # No true TTFT without streaming, use total as proxy + completion length
    return {
        "total_ms": total_ms,
        "output_tokens": usage.get("completion_tokens", len(content.split())),
        "output_chars": len(content),
    }


async def _ttft_request(endpoint: str, model: str, prompt: str, max_tokens: int = 50) -> dict:
    """TTFT-focused: streaming, time to first non-empty chunk."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    t0 = time.time()
    ttft_ms = None
    tokens_seen = 0
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}"}
            async for line in r.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        d = json.loads(chunk)
                        delta = d["choices"][0].get("delta", {})
                        content = delta.get("content") or ""
                        if content and ttft_ms is None:
                            ttft_ms = (time.time() - t0) * 1000
                        if content:
                            tokens_seen += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
    total_ms = (time.time() - t0) * 1000
    return {"ttft_ms": ttft_ms, "total_ms": total_ms, "tokens": tokens_seen}


async def _concurrent_batch(endpoint: str, model: str, prompts: list[str], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def _wrapped(p: str) -> dict:
        async with sem:
            return await _single_request(endpoint, model, p, max_tokens=150)

    return await asyncio.gather(*[_wrapped(p) for p in prompts])


async def _measure(endpoint: str, model_name: str) -> dict:
    # 1) TTFT sweep — 10 single requests, streaming
    ttft_samples = []
    for _ in range(10):
        result = await _ttft_request(endpoint, model_name, STANDARD_PROMPT, max_tokens=50)
        if result.get("ttft_ms") is not None:
            ttft_samples.append(result["ttft_ms"])

    # 2) Latency sweep — 30 single requests, non-streaming
    latencies = []
    throughputs = []
    for _ in range(30):
        result = await _single_request(endpoint, model_name, STANDARD_PROMPT, max_tokens=150)
        if "error" not in result:
            latencies.append(result["total_ms"])
            if result["output_tokens"] > 0:
                throughputs.append(result["output_tokens"] / (result["total_ms"] / 1000))

    # 3) Concurrency — measure effective throughput at concurrency=8
    prompts = [STANDARD_PROMPT] * 16
    t0 = time.time()
    batch_results = await _concurrent_batch(endpoint, model_name, prompts, concurrency=8)
    batch_wall_s = time.time() - t0
    total_tokens = sum(r.get("output_tokens", 0) for r in batch_results if "error" not in r)
    concurrent_throughput = total_tokens / batch_wall_s if batch_wall_s else 0

    def p(lst, q):
        if not lst:
            return None
        srt = sorted(lst)
        idx = min(int(q * len(srt)), len(srt) - 1)
        return round(srt[idx], 1)

    return {
        "ttft_ms": {
            "p50": p(ttft_samples, 0.5),
            "p95": p(ttft_samples, 0.95),
            "mean": round(statistics.mean(ttft_samples), 1) if ttft_samples else None,
            "samples": len(ttft_samples),
        },
        "latency_ms": {
            "p50": p(latencies, 0.5),
            "p95": p(latencies, 0.95),
            "p99": p(latencies, 0.99),
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "samples": len(latencies),
        },
        "throughput_tok_s": {
            "single_mean": round(statistics.mean(throughputs), 1) if throughputs else None,
            "single_max": round(max(throughputs), 1) if throughputs else None,
            "concurrent_batch_8": round(concurrent_throughput, 1),
        },
        "concurrent_requests_tested": 8,
        "model_name": model_name,
        "endpoint": endpoint,
    }


def measure(endpoint: str, model_name: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(_measure(endpoint, model_name))
    (out_dir / "performance.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(measure(args.endpoint, args.model, args.out), indent=2))
