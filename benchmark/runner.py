"""Benchmark orchestrator — runs all benchmarks for a client + generates report.

Design:
  1. Load client state + tier config
  2. Ensure deployment is live (client's endpoint is serving)
  3. Run 4 benchmark families:
       a) Academic (MMLU, HellaSwag, TruthfulQA)  via lm-eval-harness
       b) Domain-specific (routed by client slug or explicit domain)
       c) Performance (throughput, TTFT, latency)
       d) LLM-judge (vs base on 100 domain prompts)
  4. Load any ad-hoc benchmarks from configs/extra_benchmarks/
  5. Render markdown + HTML report

Output: clients/{client}/benchmark/report.md + report.html + results.json
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PIPELINE_ROOT = Path(__file__).parent.parent
EXTRA_BENCHMARKS_DIR = PIPELINE_ROOT / "configs" / "extra_benchmarks"


def run_all(
    client_slug: str,
    tier_cfg: dict,
    client_dir: Path,
    endpoint: str,
    base_model: str,
    adapter_path: Path,
    extra_benchmark_files: Optional[list[Path]] = None,
    skip_academic: bool = False,
    skip_domain: bool = False,
    skip_performance: bool = False,
    skip_judge: bool = False,
) -> Path:
    """Run the full benchmark suite. Returns path to generated report.md."""
    bench_dir = client_dir / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "client": client_slug,
        "tier": tier_cfg.get("tier"),
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "academic": None,
        "domain": None,
        "performance": None,
        "llm_judge": None,
        "cost": None,
        "extra": [],
    }

    # --- Academic benchmarks ---
    if not skip_academic:
        try:
            from benchmark.academic import run_academic
            results["academic"] = run_academic(
                endpoint=endpoint,
                tasks=tier_cfg.get("benchmarks", {}).get("academic", ["mmlu", "hellaswag", "truthfulqa_mc"]),
                out_dir=bench_dir / "academic",
            )
        except Exception as e:
            log.warning(f"Academic benchmark failed: {e}")
            results["academic"] = {"error": str(e)}

    # --- Domain-specific benchmarks ---
    if not skip_domain:
        try:
            from benchmark.domain import dispatch_domain
            results["domain"] = dispatch_domain(
                client_slug=client_slug,
                endpoint=endpoint,
                out_dir=bench_dir / "domain",
            )
        except Exception as e:
            log.warning(f"Domain benchmark failed: {e}")
            results["domain"] = {"error": str(e)}

    # --- Performance ---
    if not skip_performance:
        try:
            from benchmark.performance import measure
            results["performance"] = measure(
                endpoint=endpoint,
                model_name=client_slug,  # used as vLLM LoRA mount name
                out_dir=bench_dir / "performance",
            )
        except Exception as e:
            log.warning(f"Performance benchmark failed: {e}")
            results["performance"] = {"error": str(e)}

    # --- Cost comparison (derived from performance) ---
    try:
        from benchmark.cost import compute_cost_comparison
        if results["performance"] and not results["performance"].get("error"):
            results["cost"] = compute_cost_comparison(results["performance"])
    except Exception as e:
        log.warning(f"Cost comparison failed: {e}")

    # --- LLM judge vs base ---
    if not skip_judge:
        try:
            from benchmark.judge import run_judge
            results["llm_judge"] = run_judge(
                endpoint=endpoint,
                model_name=client_slug,
                base_model=base_model,
                eval_path=client_dir / "formatted_eval.jsonl",
                out_dir=bench_dir / "judge",
                n_prompts=100,
            )
        except Exception as e:
            log.warning(f"Judge benchmark failed: {e}")
            results["llm_judge"] = {"error": str(e)}

    # --- Ad-hoc benchmarks from configs/extra_benchmarks/ ---
    try:
        from benchmark.adhoc import run_adhoc_benchmarks
        results["extra"] = run_adhoc_benchmarks(
            client_slug=client_slug,
            endpoint=endpoint,
            extra_files=extra_benchmark_files,
            out_dir=bench_dir / "adhoc",
        )
    except Exception as e:
        log.warning(f"Ad-hoc benchmarks failed: {e}")
        results["extra"] = [{"error": str(e)}]

    # --- Save raw results ---
    (bench_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # --- Render report ---
    from benchmark.report import render_report
    report_path = render_report(results, bench_dir)
    log.info(f"Benchmark report written to {report_path}")

    return report_path
