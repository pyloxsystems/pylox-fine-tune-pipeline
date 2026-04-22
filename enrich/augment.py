"""Synthetic augmentation via local gpt-oss-120b.

For rare classes or small datasets, generate paraphrase variants of existing
training examples. Controlled augmentation — never invent new facts, only
rephrase user asks and (optionally) expand assistant responses.

Used cautiously. Over-augmentation hurts model quality more than it helps.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from enrich.gptoss_client import ChatMessage, GPTOSSClient

log = logging.getLogger(__name__)


PARAPHRASE_SYSTEM = """You are a training-data augmentation assistant.

Given a user question, produce N alternative phrasings of the SAME question. The variants must:
  - Preserve the original intent exactly — same information request, same constraints
  - Vary surface form: different word choice, sentence structure, politeness register
  - Be natural human phrasings (no overly formal or robotic variants)
  - NOT add new facts, details, or constraints that weren't in the original

Output format: one variant per line, no numbering, no prefix, no explanation.
"""


async def paraphrase_user_turn(
    client: GPTOSSClient, user_text: str, n: int = 3
) -> list[str]:
    messages = [
        ChatMessage("system", PARAPHRASE_SYSTEM),
        ChatMessage("user", f"Original question:\n{user_text}\n\nProduce exactly {n} paraphrases."),
    ]
    try:
        raw = await client.chat(messages, temperature=0.7, max_tokens=1024)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # Filter any accidental numbering like "1." "2)" etc
        cleaned = []
        for ln in lines:
            while ln and (ln[0].isdigit() or ln[0] in ".)-"):
                ln = ln[1:].lstrip()
            if ln:
                cleaned.append(ln)
        return cleaned[:n]
    except Exception as e:
        log.warning(f"Paraphrase failed: {e}")
        return []


async def augment_file(
    in_path: Path,
    out_path: Path,
    client: GPTOSSClient,
    variants_per_row: int = 2,
    target_role: str = "user",
) -> dict:
    """Read JSONL, produce paraphrase variants of user turns, append to output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    log.info(f"Augmenting {len(rows)} rows with {variants_per_row} variants each")

    async def _augment_row(row: dict, idx: int) -> list[dict]:
        results = [row]
        user_msgs = [m for m in row.get("messages", []) if m.get("role") == target_role]
        if not user_msgs:
            return results

        user_text = user_msgs[0]["content"]
        variants = await paraphrase_user_turn(client, user_text, n=variants_per_row)
        for j, v in enumerate(variants):
            new_row = json.loads(json.dumps(row))  # deep copy
            new_row["id"] = f"{row.get('id', idx)}_aug{j}"
            for msg in new_row.get("messages", []):
                if msg.get("role") == target_role:
                    msg["content"] = v
                    break
            new_row["_aug_source"] = row.get("id", idx)
            results.append(new_row)
        return results

    batch_size = 8
    all_rows = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        results = await asyncio.gather(*[_augment_row(r, i + j) for j, r in enumerate(batch)])
        for r_list in results:
            all_rows.extend(r_list)
        if (i // batch_size) % 5 == 0:
            log.info(f"  augmented {i + len(batch)}/{len(rows)}")

    with out_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    return {"input_rows": len(rows), "output_rows": len(all_rows), "variants_per_row": variants_per_row}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=2)
    args = parser.parse_args()

    async def _main():
        client = GPTOSSClient()
        if not await client.health():
            raise SystemExit(f"gpt-oss-120b not healthy at {client.endpoint}")
        result = await augment_file(args.input, args.output, client, variants_per_row=args.variants)
        print(json.dumps(result, indent=2))

    asyncio.run(_main())
