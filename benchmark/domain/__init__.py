"""Domain-specific benchmark dispatcher.

Routes to the right domain module based on client slug heuristics.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


_DOMAIN_KEYWORDS = {
    "legal":   ["legal", "cuad", "contract", "law", "attorney"],
    "medical": ["med", "health", "clinical", "patient", "counseling", "therapy", "psych"],
    "finance": ["finance", "fin", "bank", "trad", "fintech", "insurance"],
    "code":    ["sql", "code", "spider", "sql-spider", "programming", "dev"],
    "voice":   ["voice", "cs", "support", "bitext", "chat"],
}


def _detect_domain(client_slug: str) -> str | None:
    lower = client_slug.lower()
    for domain, kws in _DOMAIN_KEYWORDS.items():
        if any(k in lower for k in kws):
            return domain
    return None


def dispatch_domain(client_slug: str, endpoint: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = _detect_domain(client_slug)
    if not domain:
        log.info(f"No domain detected for '{client_slug}', skipping domain benchmarks")
        return {"detected_domain": None, "note": "no keyword match"}

    log.info(f"Detected domain: {domain}")

    if domain == "legal":
        from benchmark.domain.legal import run_legal
        return {"detected_domain": "legal", "results": run_legal(endpoint, client_slug, out_dir)}
    if domain == "medical":
        from benchmark.domain.medical import run_medical
        return {"detected_domain": "medical", "results": run_medical(endpoint, client_slug, out_dir)}
    if domain == "finance":
        from benchmark.domain.finance import run_finance
        return {"detected_domain": "finance", "results": run_finance(endpoint, client_slug, out_dir)}
    if domain == "code":
        from benchmark.domain.code import run_code
        return {"detected_domain": "code", "results": run_code(endpoint, client_slug, out_dir)}
    if domain == "voice":
        return {"detected_domain": "voice", "note": "no standard benchmark; using LLM judge + ad-hoc"}
    return {"detected_domain": domain, "note": "unhandled domain"}
