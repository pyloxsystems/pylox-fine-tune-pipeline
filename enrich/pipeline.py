"""Orchestrates the enrichment steps:

    validated.jsonl
        -> (semantic dedup)
        -> (quality filter via gpt-oss-120b)
        -> (PII redaction via gpt-oss-120b)
        -> (optional paraphrase augmentation)
        -> enriched_train.jsonl

Each step is idempotent + writes its artifact to the client directory. Re-running
`run_enrichment` skips completed steps unless `force=True`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from enrich.gptoss_client import GPTOSSClient

log = logging.getLogger(__name__)


async def run_enrichment(
    input_path: Path,
    out_dir: Path,
    config: dict,
    force: bool = False,
) -> Path:
    """Full enrichment. `input_path` is typically `formatted_train.jsonl`. Returns path to final enriched file."""

    client = GPTOSSClient(
        endpoint=config.get("eval", {}).get("llm_judge_endpoint", "http://localhost:8002/v1"),
        model=config.get("eval", {}).get("llm_judge_model", "openai/gpt-oss-120b"),
    )
    if not await client.health():
        # Auto-start gpt-oss-120b. Blocks until healthy (~2-3 min cold load).
        from ops.gptoss_lifecycle import ensure_running
        log.info("gpt-oss-120b not healthy — starting container")
        ensure_running()
        if not await client.health():
            raise RuntimeError(f"gpt-oss-120b failed to come up at {client.endpoint}")

    out_dir.mkdir(parents=True, exist_ok=True)

    deduped_path = out_dir / "step1_deduped.jsonl"
    dups_path = out_dir / "step1_duplicates.jsonl"
    quality_kept_path = out_dir / "step2_quality_kept.jsonl"
    quality_rej_path = out_dir / "step2_quality_rejected.jsonl"
    redacted_path = out_dir / "step3_redacted.jsonl"
    augmented_path = out_dir / "step4_augmented.jsonl"
    final_path = out_dir / "enriched_train.jsonl"
    report_path = out_dir / "enrich_report.json"

    enrich_cfg = config.get("enrich", {})
    do_dedup = enrich_cfg.get("dedup", True)
    do_quality = enrich_cfg.get("quality_filter", True)
    do_redact = enrich_cfg.get("redact", True)
    do_augment = enrich_cfg.get("augment", False)   # off by default — use sparingly

    dedup_threshold = enrich_cfg.get("dedup_threshold", 0.92)
    quality_min = enrich_cfg.get("quality_min_overall", 3)
    quality_domain = enrich_cfg.get("quality_domain")
    augment_variants = enrich_cfg.get("augment_variants_per_row", 2)

    report = {"config": enrich_cfg}
    current_path = input_path

    if do_dedup:
        if force or not deduped_path.exists():
            log.info("Step 1: semantic dedup")
            from enrich.dedup import dedup_file
            result = dedup_file(current_path, deduped_path, dups_path, threshold=dedup_threshold)
            report["dedup"] = result
        else:
            log.info(f"Step 1: skipping (exists: {deduped_path})")
        current_path = deduped_path

    if do_quality:
        if force or not quality_kept_path.exists():
            log.info("Step 2: quality filter (gpt-oss-120b scoring)")
            from enrich.quality import filter_file
            result = await filter_file(
                current_path,
                quality_kept_path,
                quality_rej_path,
                client,
                min_overall=quality_min,
                domain=quality_domain,
            )
            report["quality"] = result
        else:
            log.info(f"Step 2: skipping (exists: {quality_kept_path})")
        current_path = quality_kept_path

    if do_redact:
        if force or not redacted_path.exists():
            log.info("Step 3: PII redaction (gpt-oss-120b)")
            from enrich.redact import redact_file
            result = await redact_file(current_path, redacted_path, client)
            report["redact"] = result
        else:
            log.info(f"Step 3: skipping (exists: {redacted_path})")
        current_path = redacted_path

    if do_augment:
        if force or not augmented_path.exists():
            log.info("Step 4: paraphrase augmentation")
            from enrich.augment import augment_file
            result = await augment_file(current_path, augmented_path, client, variants_per_row=augment_variants)
            report["augment"] = result
        else:
            log.info(f"Step 4: skipping (exists: {augmented_path})")
        current_path = augmented_path

    # Copy final to canonical path
    final_path.write_text(current_path.read_text())
    report["final_path"] = str(final_path)
    report["final_row_count"] = sum(1 for line in final_path.open() if line.strip())
    report_path.write_text(json.dumps(report, indent=2, default=str))

    log.info(f"Enrichment complete: {final_path} ({report['final_row_count']} rows)")
    return final_path


def run_enrichment_sync(input_path: Path, out_dir: Path, config: dict, force: bool = False) -> Path:
    """Sync wrapper for CLI consumers."""
    return asyncio.run(run_enrichment(input_path, out_dir, config, force=force))


# Make the sync wrapper the default export expected by cli.py
def _entry(input_path: Path, out_dir: Path, config: dict) -> Path:
    return run_enrichment_sync(input_path, out_dir, config)


# cli.py imports `run_enrichment`; we expose the sync version under that name
# to avoid forcing cli.py to await.
_async_run_enrichment = run_enrichment

def run_enrichment(input_path: Path, out_dir: Path, config: dict, force: bool = False) -> Path:  # type: ignore[no-redef]
    return asyncio.run(_async_run_enrichment(input_path, out_dir, config, force=force))
