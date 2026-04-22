"""Code / SQL benchmarks.

HumanEval (Python), MBPP (basic programming), and for SQL specifically we
run against Spider test split.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

CODE_TASKS = [
    "humaneval",        # OpenAI's canonical Python benchmark
    "mbpp",             # basic Python problems
]


def run_code(endpoint: str, client_slug: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    from benchmark.academic import run_academic
    try:
        return {
            "code_suite": run_academic(
                endpoint=endpoint,
                tasks=CODE_TASKS,
                out_dir=out_dir,
                limit=100,
            ),
        }
    except Exception as e:
        log.warning(f"Code benchmarks failed: {e}")
        return {"error": str(e)}
