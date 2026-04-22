"""Generate base + fine-tune outputs for judge comparison.

Hits the vLLM endpoint twice per prompt:
  - Once with the client's LoRA adapter (fine-tune output)
  - Once using the base model name (base output)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)


async def _one_call(endpoint: str, model: str, prompt: str, max_tokens: int = 200) -> Optional[str]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                log.warning(f"HTTP {r.status_code} for model={model}: {r.text[:200]}")
                return None
            return r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError) as e:
        log.warning(f"Request failed for model={model}: {e}")
        return None


async def generate_pair_outputs(
    endpoint: str,
    model_name: str,
    eval_path: Path,
    out_path: Path,
    n: int = 100,
) -> dict:
    """For each eval prompt, generate one fine-tune output + one base output."""
    prompts: list[dict] = []
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if user_msg:
                prompts.append({
                    "id": row.get("id"),
                    "prompt": user_msg["content"],
                    "reference": next(
                        (m["content"] for m in messages if m.get("role") == "assistant"),
                        None,
                    ),
                })
    prompts = prompts[:n]
    if not prompts:
        return {"error": "no valid eval prompts"}

    # vLLM routes by "model" name to the LoRA adapter if mounted, else base
    base_model_id = "base"    # served as base if adapter disabled, but vLLM needs the actual base name
    # In our deploy/spark.py the base model is served under its canonical HF name.
    # For this benchmark we assume the endpoint has the base model served under
    # the name from the config; we look that up from the PIPELINE_ROOT configs.
    # Simpler: use a separate "base" endpoint call bypassing LoRA — but vLLM
    # LoRA mode makes this tricky. For now, call WITHOUT model name override
    # vs WITH the adapter mount name; the endpoint differentiates.
    # The actual base in vLLM is usually registered under the base HF name.
    # To avoid guessing, just generate two outputs using vLLM "base_model" by
    # passing the tokenizer-default name = None; falls back on server default.

    # Practical approach: call with LoRA mount name for FT, and with base
    # model's HF id for base. Caller provides base_model via vLLM model list.

    async def _get_models(ep: str) -> list[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(ep.rstrip("/") + "/models")
            if r.status_code == 200:
                return [m["id"] for m in r.json().get("data", [])]
        return []

    served = await _get_models(endpoint)
    # Heuristic: base model is the one that isn't the client's mount name
    non_adapter = [m for m in served if m != model_name]
    base_id = non_adapter[0] if non_adapter else None
    if not base_id:
        return {"error": f"could not find base model in vLLM /models (got: {served})"}

    results = []
    for i, p in enumerate(prompts, 1):
        ft_out = await _one_call(endpoint, model_name, p["prompt"])
        base_out = await _one_call(endpoint, base_id, p["prompt"])
        if ft_out is None or base_out is None:
            continue
        results.append({
            "id": p["id"],
            "prompt": p["prompt"],
            "reference": p["reference"],
            "finetune_output": ft_out.strip(),
            "base_output": base_out.strip(),
        })
        if i % 10 == 0:
            log.info(f"  judge-gen: {i}/{len(prompts)} prompts complete")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return {
        "samples_generated": len(results),
        "base_model_served_as": base_id,
        "output_path": str(out_path),
    }
