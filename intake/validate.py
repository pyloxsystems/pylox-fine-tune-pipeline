"""Validate client data: detect format, normalize schema, flag issues.

Accepts: JSONL, JSON array, CSV. Outputs JSONL with normalized schema:
    {"id": str, "messages": [{"role": str, "content": str}, ...]}

Detection heuristics (in order):
    1. Explicit chat format: rows have "messages" key with role/content
    2. Prompt/completion: rows have "prompt" + "completion" (OpenAI legacy)
    3. Instruction/input/output: Alpaca style
    4. Q/A: rows have "question" + "answer" (or "Context" + "Response" — Amod style)
    5. Raw text: treat each line as a single turn
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

MIN_TOKENS_APPROX = 10           # skip rows below ~40 chars
MAX_TOKENS_APPROX = 8192         # flag rows over ~32K chars
CHARS_PER_TOKEN = 4


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"{path.name}:{i} invalid JSON: {e}")


def _iter_json_array(path: Path) -> Iterator[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    yield from data


def _iter_csv(path: Path) -> Iterator[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _iter_text(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield {"text": line, "_line": i}


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in (".txt", ".md"):
        return "text"
    raise ValueError(f"Unknown file format: {path}")


def _normalize_row(row: dict, idx: int) -> dict | None:
    """Attempt to normalize any row into {id, messages[role/content]}. Returns None if unrecognizable."""
    row_id = str(row.get("id") or row.get("doc_id") or idx)

    if "messages" in row and isinstance(row["messages"], list):
        msgs = row["messages"]
        if all("role" in m and "content" in m for m in msgs):
            return {"id": row_id, "messages": msgs}

    if "prompt" in row and "completion" in row:
        return {
            "id": row_id,
            "messages": [
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["completion"])},
            ],
        }

    if "instruction" in row and "output" in row:
        user_content = str(row["instruction"])
        if row.get("input"):
            user_content += f"\n\n{row['input']}"
        return {
            "id": row_id,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": str(row["output"])},
            ],
        }

    qa_user_keys = ("question", "Context", "query", "input_text")
    qa_ast_keys = ("answer", "Response", "response", "output_text")
    user_key = next((k for k in qa_user_keys if k in row), None)
    ast_key = next((k for k in qa_ast_keys if k in row), None)
    if user_key and ast_key:
        return {
            "id": row_id,
            "messages": [
                {"role": "user", "content": str(row[user_key])},
                {"role": "assistant", "content": str(row[ast_key])},
            ],
        }

    if "text" in row:
        return {
            "id": row_id,
            "messages": [{"role": "user", "content": str(row["text"])}],
        }

    return None


def validate_data(path: Path, out_dir: Path) -> Path:
    """Validate + normalize. Writes `{out_dir}/validated.jsonl`. Returns that path."""
    fmt = _detect_format(path)
    iterator = {
        "jsonl": _iter_jsonl,
        "json": _iter_json_array,
        "csv": _iter_csv,
        "text": _iter_text,
    }[fmt](path)

    out_path = out_dir / "validated.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = kept = dropped_schema = dropped_short = dropped_long = 0
    with out_path.open("w") as out:
        for idx, row in enumerate(iterator):
            total += 1
            norm = _normalize_row(row, idx)
            if norm is None:
                dropped_schema += 1
                if dropped_schema <= 3:
                    log.warning(f"Row {idx} has unrecognized schema: {list(row.keys())[:10]}")
                continue

            total_chars = sum(len(m["content"]) for m in norm["messages"])
            if total_chars < MIN_TOKENS_APPROX * CHARS_PER_TOKEN:
                dropped_short += 1
                continue
            if total_chars > MAX_TOKENS_APPROX * CHARS_PER_TOKEN:
                dropped_long += 1
                continue

            out.write(json.dumps(norm) + "\n")
            kept += 1

    log.info(
        f"Validated {path.name}: total={total} kept={kept} "
        f"dropped_schema={dropped_schema} dropped_short={dropped_short} dropped_long={dropped_long}"
    )
    if kept == 0:
        raise ValueError(f"No valid rows found in {path}. Check schema.")
    if dropped_schema / max(total, 1) > 0.2:
        log.warning(f"High schema-drop rate ({dropped_schema}/{total}). Review input format.")

    (out_dir / "validate_report.json").write_text(json.dumps({
        "source": str(path),
        "format": fmt,
        "total": total,
        "kept": kept,
        "dropped_schema": dropped_schema,
        "dropped_short": dropped_short,
        "dropped_long": dropped_long,
    }, indent=2))

    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python -m intake.validate <input> <out_dir>")
        sys.exit(1)
    validate_data(Path(sys.argv[1]), Path(sys.argv[2]))
