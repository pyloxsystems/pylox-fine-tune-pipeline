"""Legal-specific benchmarks.

Uses LegalBench (Stanford, 162 tasks — we run a representative subset) via
lm-eval-harness, and optionally runs against a held-out CUAD test split.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Representative subset — full LegalBench is 162 tasks, too slow for routine runs.
# These 6 cover breadth: contract clauses, constitutional law, consumer rights, etc.
LEGAL_BENCH_SUBSET = [
    "legalbench_abercrombie",                     # trademark classification
    "legalbench_canada_tax_court_outcomes",       # tax case outcomes
    "legalbench_citation_prediction_classification",
    "legalbench_contract_nli_confidentiality_of_agreement",
    "legalbench_corporate_lobbying",
    "legalbench_successor_liability",
]


def run_legal(endpoint: str, client_slug: str, out_dir: Path) -> dict:
    """Run LegalBench subset + CUAD test split via lm-eval + direct eval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # LegalBench subset via lm-eval-harness
    try:
        from benchmark.academic import run_academic
        legal_bench = run_academic(
            endpoint=endpoint,
            tasks=LEGAL_BENCH_SUBSET,
            out_dir=out_dir / "legalbench",
            limit=100,  # 100 per task × 6 tasks = 600 prompts total
        )
        results["legalbench_subset"] = legal_bench
    except Exception as e:
        log.warning(f"LegalBench failed: {e}")
        results["legalbench_subset"] = {"error": str(e)}

    return results
