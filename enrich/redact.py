"""Robust PII redaction via local gpt-oss-120b.

The regex sanitization in `intake/sanitize_sample.py` is for quick 20-row samples.
For full-corpus redaction during enrichment, we use the LLM for robust NER
(catches names without salutation cues, addresses, medical record numbers, etc.).

All processing happens on the local Spark — data never leaves the box.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from enrich.gptoss_client import ChatMessage, GPTOSSClient

log = logging.getLogger(__name__)

REDACT_SYSTEM = """You are a strict PII redaction assistant. Given a text message, return the same message with every piece of personally identifiable information replaced by a bracketed tag.

Tags to use:
  [NAME]           — any person name
  [EMAIL]          — email addresses
  [PHONE]          — phone numbers in any format
  [ADDRESS]        — physical addresses
  [SSN]            — social security / tax IDs
  [CREDIT_CARD]    — payment card numbers
  [MRN]            — medical record / patient IDs
  [ACCOUNT]        — any other account/customer number
  [DATE_OF_BIRTH]  — DOB only (not general dates)
  [URL]            — URLs and domain references
  [IP]             — IP addresses

Rules:
  - Only replace tokens that are actual PII. Preserve structure, grammar, and non-PII content exactly.
  - Output MUST be ONLY the redacted text, no explanation, no quotes, no preamble.
  - If no PII is present, return the original text unchanged.
"""


async def redact_text(client: GPTOSSClient, text: str) -> str:
    if not text.strip():
        return text
    messages = [
        ChatMessage("system", REDACT_SYSTEM),
        ChatMessage("user", text),
    ]
    try:
        return await client.chat(messages, temperature=0.0, max_tokens=4096)
    except Exception as e:
        log.warning(f"Redaction failed for chunk (len={len(text)}): {e}. Returning original.")
        return text


async def redact_file(
    in_path: Path,
    out_path: Path,
    client: GPTOSSClient,
    fields: tuple[str, ...] = ("user", "assistant"),
) -> dict:
    """Read JSONL with {messages: [...]}, redact each message.content, write redacted JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    log.info(f"Redacting {len(rows)} rows -> {out_path}")

    async def _redact_row(row: dict) -> dict:
        messages = row.get("messages", [])
        for msg in messages:
            if msg.get("role") in fields:
                msg["content"] = await redact_text(client, msg["content"])
        if "text" in row:
            row["text"] = None  # will be re-rendered after redaction
        return row

    # Process in batches to respect concurrency
    redacted = []
    batch_size = 16
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        results = await asyncio.gather(*[_redact_row(r) for r in batch])
        redacted.extend(results)
        if (i // batch_size) % 5 == 0:
            log.info(f"  redacted {i + len(batch)}/{len(rows)} rows")

    with out_path.open("w") as f:
        for row in redacted:
            f.write(json.dumps(row) + "\n")

    return {"total": len(rows), "output": str(out_path)}


async def _cli_main(in_path: Path, out_path: Path) -> None:
    client = GPTOSSClient()
    if not await client.health():
        raise SystemExit(f"gpt-oss-120b not healthy at {client.endpoint}")
    result = await redact_file(in_path, out_path, client)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(_cli_main(args.input, args.output))
