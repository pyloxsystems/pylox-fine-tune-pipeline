"""LLM-as-judge: gpt-oss-120b compares base vs fine-tune outputs, picks winner.

Produces the "win rate" metric that appears in the client-facing eval report.
A >60% win rate for the fine-tune is a defensible improvement signal.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from enrich.gptoss_client import ChatMessage, GPTOSSClient

log = logging.getLogger(__name__)


JUDGE_SYSTEM = """You are an impartial judge comparing two AI-generated responses to the same user prompt.

Given the prompt and two candidate responses (A and B), choose which response is better for the following criteria combined:
  - Follows the prompt precisely
  - Is informative, complete, and correct
  - Has a natural conversational tone
  - Is free of obvious errors

Respond in this exact JSON shape, no extra text:
{
  "winner": "A" | "B" | "tie",
  "reason": "<one-sentence reason>"
}

Rules:
  - Do NOT be swayed by response length alone. Favor clarity + relevance.
  - If A and B are roughly equivalent, output "tie".
  - Your choice must be consistent with your stated reason.
"""


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_verdict(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return {"winner": "tie", "reason": "unparseable", "_raw": raw[:300]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"winner": "tie", "reason": "unparseable", "_raw": raw[:300]}


async def judge_pair(client: GPTOSSClient, prompt: str, a_text: str, b_text: str) -> dict:
    user_msg = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"RESPONSE A:\n{a_text}\n\n"
        f"RESPONSE B:\n{b_text}"
    )
    messages = [
        ChatMessage("system", JUDGE_SYSTEM),
        ChatMessage("user", user_msg),
    ]
    try:
        raw = await client.chat(messages, temperature=0.0, max_tokens=256)
    except Exception as e:
        log.warning(f"Judge call failed: {e}")
        return {"winner": "tie", "reason": str(e)[:200]}
    return _parse_verdict(raw)


async def judge_samples_file(
    samples_path: Path,
    out_path: Path,
    client: GPTOSSClient,
    swap_order: bool = True,
) -> dict:
    """For each row in samples_path, judge base vs fine-tune. Return summary stats.

    To mitigate position bias, by default we randomize which is A vs B
    per example (swap_order=True).
    """
    import random
    rng = random.Random(42)

    with samples_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    log.info(f"Judging {len(rows)} base-vs-finetune pairs")

    async def _judge_row(row: dict) -> dict:
        prompt = row["prompt"]
        base = row["base_output"]
        ft = row["finetune_output"]

        if swap_order and rng.random() < 0.5:
            verdict = await judge_pair(client, prompt, ft, base)
            winner_raw = verdict.get("winner", "tie").lower()
            if winner_raw == "a":
                winner = "finetune"
            elif winner_raw == "b":
                winner = "base"
            else:
                winner = "tie"
        else:
            verdict = await judge_pair(client, prompt, base, ft)
            winner_raw = verdict.get("winner", "tie").lower()
            if winner_raw == "a":
                winner = "base"
            elif winner_raw == "b":
                winner = "finetune"
            else:
                winner = "tie"

        return {
            **row,
            "judge_winner": winner,
            "judge_reason": verdict.get("reason", ""),
        }

    batch_size = 8
    judged = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        results = await asyncio.gather(*[_judge_row(r) for r in batch])
        judged.extend(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in judged:
            f.write(json.dumps(row) + "\n")

    # Summarize
    wins = {"base": 0, "finetune": 0, "tie": 0}
    for row in judged:
        wins[row["judge_winner"]] += 1
    total = len(judged) or 1
    summary = {
        "total": total,
        "finetune_wins": wins["finetune"],
        "base_wins": wins["base"],
        "ties": wins["tie"],
        "finetune_win_rate": round(wins["finetune"] / total, 3),
        "base_win_rate": round(wins["base"] / total, 3),
        "tie_rate": round(wins["tie"] / total, 3),
        "output_path": str(out_path),
    }
    log.info(f"Judge results: {summary}")
    return summary


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    async def _main():
        client = GPTOSSClient()
        if not await client.health():
            from ops.gptoss_lifecycle import ensure_running
            ensure_running()
        summary = await judge_samples_file(args.samples, args.out, client)
        print(json.dumps(summary, indent=2))

    asyncio.run(_main())
