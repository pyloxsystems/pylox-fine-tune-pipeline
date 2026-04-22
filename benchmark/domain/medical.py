"""Medical / healthcare benchmarks.

MedQA (USMLE), MedMCQA, PubMedQA — routed via lm-eval-harness.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

MEDICAL_TASKS = [
    "medqa_4options",       # USMLE-style questions (4-option MCQ)
    "medmcqa",              # 194K medical MCQ
    "pubmedqa",             # biomedical research Q&A
    "mmlu_medical_genetics",
    "mmlu_clinical_knowledge",
    "mmlu_college_medicine",
]


def run_medical(endpoint: str, client_slug: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    from benchmark.academic import run_academic
    try:
        return {
            "medical_suite": run_academic(
                endpoint=endpoint,
                tasks=MEDICAL_TASKS,
                out_dir=out_dir,
                limit=200,
            ),
        }
    except Exception as e:
        log.warning(f"Medical benchmarks failed: {e}")
        return {"error": str(e)}
