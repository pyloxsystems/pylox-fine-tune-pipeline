"""LLM-as-judge: use gpt-oss-120b to compare fine-tune vs base on domain prompts.

Reuses eval/llm_judge.py under the hood but scales from 10 prompts (eval) to 100.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_judge(
    endpoint: str,
    model_name: str,
    base_model: str,
    eval_path: Path,
    out_dir: Path,
    n_prompts: int = 100,
) -> dict:
    """Compare fine-tune vs base on N prompts using gpt-oss-120b judge."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        return {"error": f"no eval data at {eval_path}"}

    # Ensure gpt-oss-120b is running (needed for judging)
    from ops.gptoss_lifecycle import ensure_running, ensure_stopped
    ensure_running()

    # Generate base + finetune outputs via the deployed vLLM endpoint
    from benchmark.judge_gen import generate_pair_outputs
    samples_path = out_dir / "samples.jsonl"
    perf = asyncio.run(generate_pair_outputs(
        endpoint=endpoint,
        model_name=model_name,
        eval_path=eval_path,
        out_path=samples_path,
        n=n_prompts,
    ))

    if "error" in perf:
        return perf

    # Run judge
    from enrich.gptoss_client import GPTOSSClient
    from eval.llm_judge import judge_samples_file
    client = GPTOSSClient()
    summary = asyncio.run(judge_samples_file(
        samples_path, out_dir / "judgments.jsonl", client,
    ))

    summary["samples_tested"] = perf.get("samples_generated", 0)
    return summary


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    print(json.dumps(
        run_judge(args.endpoint, args.model, args.base, args.eval, args.out, args.n),
        indent=2,
    ))
