"""Ad-hoc benchmark loader — YAML-configured custom tests dropped in configs/extra_benchmarks/.

Supports custom test sets a user drops in later. See ADHOC_FORMAT at bottom for schema.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

PIPELINE_ROOT = Path(__file__).parent.parent
EXTRA_DIR = PIPELINE_ROOT / "configs" / "extra_benchmarks"


def _discover(client_slug: str, explicit_files: Optional[list[Path]]) -> list[Path]:
    """Find ad-hoc benchmark YAMLs that apply to this client.

    Rules:
      - Explicit --extra files (always apply)
      - Files matching client-{slug}-*.yml (auto-apply per client)
      - Files matching global-*.yml (auto-apply to all clients)
    """
    files: list[Path] = list(explicit_files or [])
    if EXTRA_DIR.exists():
        for f in EXTRA_DIR.glob("global-*.yml"):
            if f.name.upper().startswith("EXAMPLE"):
                continue
            files.append(f)
        for f in EXTRA_DIR.glob(f"client-{client_slug}-*.yml"):
            if f.name.upper().startswith("EXAMPLE"):
                continue
            files.append(f)
    # Dedup preserving order
    seen = set()
    unique = []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


async def _load_dataset(spec: dict) -> list[dict]:
    """Handle `dataset:` in YAML — local path, HF dataset, or inline list."""
    source = spec.get("dataset", "")
    if isinstance(spec.get("dataset"), list):
        return spec["dataset"]

    if source.startswith("huggingface:"):
        from datasets import load_dataset
        name = source[len("huggingface:") :]
        ds = load_dataset(name, split=spec.get("split", "test"))
        return [dict(row) for row in ds]
    if source.startswith("lm-eval:"):
        raise NotImplementedError("lm-eval tasks in ad-hoc not yet implemented")
    if source.endswith(".jsonl") or source.endswith(".json"):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"dataset path {p} missing")
        if p.suffix == ".json":
            return json.loads(p.read_text())
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    raise ValueError(f"Unknown dataset source: {source}")


def _render_prompt(template: str, row: dict) -> str:
    """Simple {field} replacement in prompt template."""
    def repl(m):
        field = m.group(1).strip()
        return str(row.get(field, ""))
    return re.sub(r"\{(\w+)\}", repl, template)


async def _score_exact_match(generated: str, expected: str) -> float:
    return 1.0 if generated.strip().lower() == expected.strip().lower() else 0.0


async def _score_substring(generated: str, expected: str) -> float:
    return 1.0 if expected.strip().lower() in generated.strip().lower() else 0.0


async def _score_llm_judge(generated: str, expected: str, prompt: str, gptoss_client) -> float:
    from enrich.gptoss_client import ChatMessage
    system = (
        "Grade the AI's answer as 1 (correct) or 0 (incorrect) compared to the reference. "
        "Respond with ONLY the number 0 or 1."
    )
    user = f"Question: {prompt}\n\nReference: {expected}\n\nAI answer: {generated}\n\nScore:"
    raw = await gptoss_client.chat([
        ChatMessage("system", system),
        ChatMessage("user", user),
    ], temperature=0.0, max_tokens=4)
    return 1.0 if raw.strip().startswith("1") else 0.0


async def _run_one_benchmark(spec_path: Path, endpoint: str, client_slug: str, out_dir: Path) -> dict:
    spec = yaml.safe_load(spec_path.read_text())

    rows = await _load_dataset(spec)
    sample_size = spec.get("sample_size", 200)
    rows = rows[:sample_size]

    prompt_template = spec.get("prompt_template", "{question}")
    metric = spec.get("metric", {"type": "substring"})
    answer_field = metric.get("answer_field", "answer")

    import httpx
    gptoss_client = None
    if metric.get("type") == "llm_judge":
        from enrich.gptoss_client import GPTOSSClient
        gptoss_client = GPTOSSClient()

    scores = []
    async with httpx.AsyncClient(timeout=120) as http:
        for row in rows:
            prompt = _render_prompt(prompt_template, row)
            expected = str(row.get(answer_field, "") or "")
            if not expected:
                continue
            try:
                r = await http.post(endpoint.rstrip("/") + "/chat/completions", json={
                    "model": client_slug,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": spec.get("max_tokens", 200),
                    "temperature": 0.0,
                })
                if r.status_code != 200:
                    continue
                generated = r.json()["choices"][0]["message"]["content"]
            except Exception:
                continue

            mtype = metric.get("type", "substring")
            if mtype == "exact_match":
                s = await _score_exact_match(generated, expected)
            elif mtype == "substring":
                s = await _score_substring(generated, expected)
            elif mtype == "llm_judge":
                s = await _score_llm_judge(generated, expected, prompt, gptoss_client)
            else:
                s = 0.0
            scores.append(s)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    result = {
        "name": spec.get("name", spec_path.stem),
        "description": spec.get("description"),
        "source": str(spec_path),
        "samples": len(scores),
        "score": round(mean_score, 4),
        "metric_type": metric.get("type"),
    }
    (out_dir / f"{spec.get('name', spec_path.stem)}.json").write_text(json.dumps(result, indent=2))
    return result


def run_adhoc_benchmarks(
    client_slug: str,
    endpoint: str,
    out_dir: Path,
    extra_files: Optional[list[Path]] = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _discover(client_slug, extra_files)
    if not specs:
        log.info("No ad-hoc benchmarks found.")
        return []

    log.info(f"Running {len(specs)} ad-hoc benchmarks")
    results = []
    for spec in specs:
        try:
            r = asyncio.run(_run_one_benchmark(spec, endpoint, client_slug, out_dir))
            log.info(f"  {r['name']}: score {r['score']} ({r['samples']} samples)")
            results.append(r)
        except Exception as e:
            log.warning(f"Ad-hoc benchmark {spec} failed: {e}")
            results.append({"name": spec.stem, "error": str(e)})
    return results


ADHOC_FORMAT = """
# Example: configs/extra_benchmarks/client-acme-legal.yml
name: acme-custom-contract-qa
description: ACME's 200-question contract test set
dataset: /data/acme_test.jsonl
split: test                       # optional, for HF datasets
sample_size: 200                  # optional cap
prompt_template: |
  Review this contract clause:
  {context}
  Question: {question}
max_tokens: 200                   # optional
metric:
  type: substring                 # exact_match | substring | llm_judge
  answer_field: answer             # field in the row to compare against
"""
