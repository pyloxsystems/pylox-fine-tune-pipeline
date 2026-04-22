"""Apply chat template + split train/eval. Output ready for QLoRA SFTTrainer.

Input:  {out_dir}/validated.jsonl (schema: {id, messages})
Output: {out_dir}/formatted_train.jsonl
        {out_dir}/formatted_eval.jsonl

trl's SFTTrainer with `packing=True` expects a dataset where each row has a
"text" field already rendered through the chat template. We render eagerly
here so downstream steps are tokenizer-agnostic for inspection.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


def _load_tokenizer(base_model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def format_for_chat(validated_path: Path, out_dir: Path, config: dict) -> Path:
    tokenizer = _load_tokenizer(config["base_model"])

    rows = []
    with validated_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(42)
    rng.shuffle(rows)

    eval_ratio = config.get("eval", {}).get("eval_split_ratio", 0.05)
    split = max(1, int(len(rows) * eval_ratio))
    eval_rows = rows[:split]
    train_rows = rows[split:]

    train_path = out_dir / "formatted_train.jsonl"
    eval_path = out_dir / "formatted_eval.jsonl"

    def _write(rows_, path):
        with path.open("w") as f:
            for row in rows_:
                text = _render(tokenizer, row["messages"])
                f.write(json.dumps({"id": row["id"], "text": text, "messages": row["messages"]}) + "\n")

    _write(train_rows, train_path)
    _write(eval_rows, eval_path)

    log.info(f"Formatted: train={len(train_rows)} eval={len(eval_rows)} -> {train_path}")
    (out_dir / "format_report.json").write_text(json.dumps({
        "base_model": config["base_model"],
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_ratio": eval_ratio,
    }, indent=2))

    return train_path


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validated", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("config", type=Path, help="Tier config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    format_for_chat(args.validated, args.out_dir, cfg)
