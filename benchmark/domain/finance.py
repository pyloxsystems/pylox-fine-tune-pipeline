"""Finance / quant benchmarks.

FinanceBench, FinQA routed via lm-eval-harness.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

FINANCE_TASKS = [
    "financebench",       # financial Q&A from 10-K filings
    "finqa",              # numerical reasoning over financial docs
    "fpb",                # financial phrasebook sentiment
]


def run_finance(endpoint: str, client_slug: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    from benchmark.academic import run_academic
    try:
        return {
            "finance_suite": run_academic(
                endpoint=endpoint,
                tasks=FINANCE_TASKS,
                out_dir=out_dir,
                limit=200,
            ),
        }
    except Exception as e:
        log.warning(f"Finance benchmarks failed: {e}")
        return {"error": str(e)}
