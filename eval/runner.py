"""Orchestrates eval:  generate samples -> LLM judge -> Markdown report.

1. Generate base vs fine-tune outputs (eval/samples.py).
   This unloads gpt-oss first to free memory for the training model.
2. After sample generation, unload training model + restart gpt-oss.
3. LLM judge comparison (eval/llm_judge.py).
4. Render markdown report from template.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from jinja2 import Template

log = logging.getLogger(__name__)


def run_eval(
    adapter_path: Path,
    data_path: Path,
    client_dir: Path,
    cfg: dict,
) -> Path:
    """Full eval pipeline. Returns path to the rendered markdown report."""
    eval_dir = client_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    samples_path = eval_dir / "samples.jsonl"
    judgments_path = eval_dir / "judgments.jsonl"
    report_path = eval_dir / "eval_report.md"

    # Find eval jsonl — look for formatted_eval.jsonl in client_dir
    eval_jsonl = client_dir / "formatted_eval.jsonl"
    if not eval_jsonl.exists():
        eval_jsonl = data_path  # fallback to training data (will use a slice)

    # 1) Generate samples (auto-unloads gpt-oss-120b)
    log.info("Step 1/3: generating base + fine-tune samples")
    from eval.samples import generate_samples
    generate_samples(adapter_path, eval_jsonl, samples_path, cfg)

    # 2) Judge (auto-starts gpt-oss-120b)
    log.info("Step 2/3: LLM judge comparison")
    from enrich.gptoss_client import GPTOSSClient
    from ops.gptoss_lifecycle import ensure_running
    from eval.llm_judge import judge_samples_file

    ensure_running()
    client = GPTOSSClient()
    summary = asyncio.run(judge_samples_file(samples_path, judgments_path, client))

    # 3) Render markdown report
    log.info("Step 3/3: rendering report")
    with (judgments_path).open() as f:
        judged_rows = [json.loads(line) for line in f if line.strip()]

    tpl_path = Path(__file__).parent / "report.md.j2"
    tpl = Template(tpl_path.read_text())

    # Attempt optional perplexity delta if trainer metrics exist
    perplexity_baseline = perplexity_finetune = None
    ppl_delta = None

    client_slug = client_dir.name
    markdown = tpl.render(
        client_slug=client_slug,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        base_model=cfg["base_model"],
        tier=cfg["tier"],
        adapter_path=str(adapter_path),
        samples_path=str(samples_path),
        judgments_path=str(judgments_path),
        total=summary["total"],
        finetune_win_rate=summary["finetune_win_rate"],
        base_win_rate=summary["base_win_rate"],
        tie_rate=summary["tie_rate"],
        perplexity_baseline=perplexity_baseline,
        perplexity_finetune=perplexity_finetune,
        perplexity_delta=ppl_delta,
        samples=judged_rows,
    )
    report_path.write_text(markdown)

    # Also write machine-readable summary
    (eval_dir / "summary.json").write_text(json.dumps({
        **summary,
        "adapter": str(adapter_path),
        "base_model": cfg["base_model"],
        "tier": cfg["tier"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    log.info(f"Report written to {report_path}")
    return report_path


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("client_dir", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    report = run_eval(args.adapter, args.data, args.client_dir, cfg)
    print(f"Report: {report}")
