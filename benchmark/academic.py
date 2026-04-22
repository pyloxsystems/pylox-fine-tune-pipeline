"""Run standard academic benchmarks via lm-evaluation-harness against vLLM endpoint."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_TASKS = {
    # Task name -> (display name, sample size limit for speed)
    "mmlu": ("MMLU (general knowledge, 57 subjects)", 500),
    "hellaswag": ("HellaSwag (common sense reasoning)", 500),
    "truthfulqa_mc": ("TruthfulQA (hallucination resistance)", 500),
    "gsm8k": ("GSM8K (grade school math)", 300),
    "arc_easy": ("ARC-Easy (science reasoning)", 500),
    "arc_challenge": ("ARC-Challenge (hard science)", 500),
}


def run_academic(endpoint: str, tasks: list[str], out_dir: Path, limit: int = 500) -> dict:
    """Invoke lm-eval CLI against the vLLM endpoint. Returns aggregated scores."""
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_tasks = [t for t in tasks if t in SUPPORTED_TASKS]
    if not valid_tasks:
        log.warning(f"No supported tasks in {tasks}; skipping academic.")
        return {"error": "no supported tasks"}

    task_str = ",".join(valid_tasks)
    results_json = out_dir / "lm_eval_results.json"

    # lm-eval OpenAI-compatible endpoint
    model_args = (
        f"base_url={endpoint.rstrip('/')}/chat/completions"
        f",model=localhost"    # endpoint only serves one base, model name is cosmetic for api
        f",num_concurrent=4"
        f",timeout=120"
    )

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", task_str,
        "--limit", str(limit),
        "--output_path", str(results_json),
        "--batch_size", "auto",
    ]

    log.info(f"Running academic benchmarks: {valid_tasks}")
    log.debug(f"CMD: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            return {
                "error": f"lm-eval failed (rc={result.returncode})",
                "stderr": result.stderr[-2000:],
                "tasks_attempted": valid_tasks,
            }
    except subprocess.TimeoutExpired:
        return {"error": "lm-eval exceeded 1 hour timeout", "tasks_attempted": valid_tasks}

    # Parse results
    if not results_json.exists():
        # lm-eval sometimes writes to a nested path
        matches = list(out_dir.rglob("*results*.json"))
        if matches:
            results_json = matches[0]
        else:
            return {"error": "no results.json produced", "stdout": result.stdout[-1000:]}

    data = json.loads(results_json.read_text())
    raw_results = data.get("results", {})

    summary = {}
    for task in valid_tasks:
        if task in raw_results:
            task_results = raw_results[task]
            # Typical metrics: acc, acc_stderr, acc_norm, etc.
            score_keys = [k for k in task_results if "," in k and "acc" in k.split(",")[0]]
            primary_score = None
            for k in score_keys:
                if "acc_norm" in k:
                    primary_score = task_results[k]
                    break
            if primary_score is None and score_keys:
                primary_score = task_results[score_keys[0]]
            summary[task] = {
                "display_name": SUPPORTED_TASKS[task][0],
                "score": round(primary_score, 4) if primary_score is not None else None,
                "sample_size": limit,
            }

    return {
        "raw_path": str(results_json),
        "tasks": summary,
        "model": data.get("model_name", "via endpoint"),
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--tasks", default="mmlu,hellaswag,truthfulqa_mc")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()
    result = run_academic(args.endpoint, args.tasks.split(","), args.out, args.limit)
    print(json.dumps(result, indent=2))
