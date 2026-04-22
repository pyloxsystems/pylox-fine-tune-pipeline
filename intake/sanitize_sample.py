"""Extract N rows from validated data + scrub PII for Claude Code review.

Usage:
    python -m intake.sanitize_sample clients/acme/validated.jsonl

Writes to clients/acme/safe_sample.jsonl — paste this into Claude Code when
designing client-specific preprocessing logic. Real client data NEVER leaves
your Spark.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Conservative regex patterns. Not perfect, but good enough for sample sanitization.
# For production PII redaction over full corpus, use enrich/gptoss_redact.py (LLM-based).

PATTERNS = {
    "email":  re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone":  re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ipv4":   re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url":    re.compile(r"https?://[^\s<>\"]+"),
}

NAME_HINTS = [
    # Simple first-name regex — covers common English names. Use enrich step for robust NER.
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"),
    re.compile(r"\b(?:Hi|Hello|Dear)\s+([A-Z][a-z]+)\b"),
]


def scrub(text: str) -> str:
    for label, pattern in PATTERNS.items():
        text = pattern.sub(f"[{label.upper()}_REDACTED]", text)
    for pattern in NAME_HINTS:
        text = pattern.sub("[NAME_REDACTED]", text)
    return text


def sanitize_file(in_path: Path, out_path: Path, n: int = 20, seed: int = 42) -> None:
    with in_path.open() as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"{in_path} is empty")

    rng = random.Random(seed)
    sample_lines = rng.sample(lines, min(n, len(lines)))

    with out_path.open("w") as out:
        for line in sample_lines:
            row = json.loads(line)
            for msg in row.get("messages", []):
                msg["content"] = scrub(msg["content"])
            out.write(json.dumps(row) + "\n")

    log.info(f"Wrote {len(sample_lines)} sanitized rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Validated JSONL file")
    parser.add_argument("--n", type=int, default=20, help="How many rows to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: alongside input)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = args.out or (args.input.parent / "safe_sample.jsonl")
    sanitize_file(args.input, out, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
