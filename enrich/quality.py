"""Quality filtering via local gpt-oss-120b.

For each training example, ask the model to score it 1-5 across multiple
dimensions + a single OVERALL score. Examples below a threshold are dropped
from the training set (but retained in a `quality_rejects.jsonl` for audit).

Rubric favors examples that:
  - Have a clear user intent and a substantive assistant response
  - Use natural conversational language (not template-filled)
  - Are free of obvious errors (factual, grammatical, formatting)
  - Match the target domain (if domain hint provided)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

from enrich.gptoss_client import ChatMessage, GPTOSSClient

log = logging.getLogger(__name__)


RUBRIC_SYSTEM = """You are a training-data quality reviewer. You will see a single conversation (user + assistant turns).

Score the conversation on the 1-5 scale for each rubric item. Respond in this exact JSON shape:
{
  "clarity": <int 1-5>,
  "completeness": <int 1-5>,
  "naturalness": <int 1-5>,
  "correctness": <int 1-5>,
  "domain_fit": <int 1-5>,
  "overall": <int 1-5>,
  "reason": "<brief, 1-sentence reason for overall score>"
}

Rules:
  - Output ONLY the JSON object. No prose before or after.
  - Be strict: mediocre examples should score 3. Only clearly excellent examples earn 5.
  - If the conversation is empty, malformed, or nonsense, score everything 1.
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_scores(raw: str) -> Optional[dict]:
    """Try to extract a JSON object from the model's response."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _format_conversation(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


async def score_row(client: GPTOSSClient, row: dict, domain: Optional[str] = None) -> dict:
    """Score a single conversation. Returns dict with scores + reason."""
    convo = _format_conversation(row.get("messages", []))
    system = RUBRIC_SYSTEM
    if domain:
        system += f"\n\nTarget domain: {domain}"

    messages = [
        ChatMessage("system", system),
        ChatMessage("user", convo),
    ]
    try:
        raw = await client.chat(messages, temperature=0.0, max_tokens=256)
        scores = _parse_scores(raw)
        if scores is None:
            log.warning(f"Failed to parse scores for row {row.get('id')}. Raw: {raw[:200]}")
            return {"overall": 3, "_parse_failed": True, "raw": raw[:200]}
        return scores
    except Exception as e:
        log.warning(f"Quality scoring failed for row {row.get('id')}: {e}")
        return {"overall": 3, "_error": str(e)}


async def filter_file(
    in_path: Path,
    kept_path: Path,
    rejected_path: Path,
    client: GPTOSSClient,
    min_overall: int = 3,
    domain: Optional[str] = None,
    batch_size: int = 16,
) -> dict:
    """Score each row, write kept/rejected files."""
    kept_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    log.info(f"Quality scoring {len(rows)} rows (min_overall={min_overall}, domain={domain!r})")

    scored_rows: list[tuple[dict, dict]] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        scores = await asyncio.gather(*[score_row(client, r, domain) for r in batch])
        for row, score in zip(batch, scores):
            scored_rows.append((row, score))
        if (i // batch_size) % 5 == 0:
            log.info(f"  scored {i + len(batch)}/{len(rows)} rows")

    kept = rejected = 0
    score_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    with kept_path.open("w") as kept_f, rejected_path.open("w") as rej_f:
        for row, score in scored_rows:
            overall = int(score.get("overall", 3))
            score_dist[max(1, min(5, overall))] += 1
            row["_quality"] = score
            if overall >= min_overall:
                kept_f.write(json.dumps(row) + "\n")
                kept += 1
            else:
                rej_f.write(json.dumps(row) + "\n")
                rejected += 1

    log.info(f"Kept {kept}, rejected {rejected}. Distribution: {score_dist}")
    return {
        "total": len(rows),
        "kept": kept,
        "rejected": rejected,
        "min_overall": min_overall,
        "score_distribution": score_dist,
    }


async def _cli_main(in_path: Path, kept_path: Path, rej_path: Path, min_overall: int, domain: Optional[str]) -> None:
    client = GPTOSSClient()
    if not await client.health():
        raise SystemExit(f"gpt-oss-120b not healthy at {client.endpoint}")
    result = await filter_file(in_path, kept_path, rej_path, client, min_overall=min_overall, domain=domain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--min-overall", type=int, default=3)
    parser.add_argument("--domain", default=None)
    args = parser.parse_args()
    asyncio.run(_cli_main(args.input, args.kept, args.rejected, args.min_overall, args.domain))
