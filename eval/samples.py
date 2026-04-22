"""Generate base + fine-tune outputs on eval prompts using vLLM.

Production-grade: uses vLLM's batched offline inference with LoRA hot-swap.
This matches what `deploy/spark.py` and `deploy/runpod.py` run in production,
so eval results are representative of what clients will actually see.

Pipeline:
  1. Auto-unload gpt-oss-120b (via lifecycle)
  2. Load base model in vLLM with LoRA support enabled
  3. Issue ONE batched generate call with the LoRA adapter (fine-tune outputs)
  4. Issue ONE batched generate call without LoRA (base outputs)
  5. Pair, write JSONL with side-by-side comparisons + actual TTFT/throughput

Why vLLM not transformers.generate():
  - HF generate on Blackwell + NF4 + DoRA: ~5-15 tok/sec (catastrophically slow)
  - vLLM with continuous batching + paged attention: ~50-150 tok/sec
  - vLLM also gives us PROD-equivalent numbers for client reports
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def generate_samples(
    adapter_path: Path,
    eval_path: Path,
    out_path: Path,
    cfg: dict,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> Path:
    """Generate base + fine-tune outputs via vLLM batched offline inference."""
    from ops.gptoss_lifecycle import cycle_for_training
    from ops.memory_guard import ensure_headroom, estimate_model_gib
    cycle_for_training()

    base_model = cfg["base_model"]
    quant = cfg["quantization"]["bnb_4bit_quant_type"]
    # vLLM in BF16 needs ~2x the NF4 footprint for the base model, plus
    # LoRA + KV cache headroom. Be generous on the estimate.
    needed = max(estimate_model_gib(base_model, quant="bf16"), 16.0)
    ensure_headroom(needed_gib=needed, reserve_gib=8.0)

    import ops.cuda_compat       # noqa: F401  — must precede vllm import
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    log.info(f"Loading vLLM base model: {base_model}")
    t0 = time.time()
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=cfg["lora"]["r"],
        max_loras=2,
        gpu_memory_utilization=0.55,    # leave room for KV cache + later judge
        enforce_eager=True,             # required for Blackwell sm_121
        max_model_len=cfg["training"].get("max_seq_length", 2048),
        trust_remote_code=True,
    )
    log.info(f"vLLM loaded in {time.time()-t0:.1f}s")

    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens,
        repetition_penalty=1.1,
    )

    # Build prompt list using the model's chat template via vLLM's tokenizer
    eval_rows = []
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                eval_rows.append(json.loads(line))

    max_prompts = cfg.get("eval", {}).get("sample_prompts", 10)
    eval_rows = eval_rows[:max_prompts]
    log.info(f"Generating on {len(eval_rows)} eval prompts")

    tokenizer = llm.get_tokenizer()
    prompt_texts: list[str] = []
    references: list[str | None] = []
    user_turns: list[str] = []
    row_ids: list[str] = []

    for i, row in enumerate(eval_rows):
        messages = row.get("messages", [])
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        if not user_msg:
            continue
        prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
        rendered = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(rendered)
        user_turns.append(user_msg["content"])
        references.append(next((m["content"] for m in messages if m.get("role") == "assistant"), None))
        row_ids.append(row.get("id", str(i)))

    if not prompt_texts:
        raise RuntimeError("No valid eval prompts (need user turn). Check eval data format.")

    # === FINE-TUNE INFERENCE (batched) ===
    log.info(f"Generating fine-tune outputs for {len(prompt_texts)} prompts (batched)")
    t0 = time.time()
    lora_req = LoRARequest("client_ft", 1, str(adapter_path.resolve()))
    ft_outputs = llm.generate(prompt_texts, sampling, lora_request=lora_req)
    ft_wall = time.time() - t0
    log.info(f"  Fine-tune batch done in {ft_wall:.1f}s ({sum(len(o.outputs[0].token_ids) for o in ft_outputs)/ft_wall:.1f} tok/s aggregate)")

    # === BASE INFERENCE (batched, no adapter) ===
    log.info(f"Generating base outputs for {len(prompt_texts)} prompts (batched)")
    t0 = time.time()
    base_outputs = llm.generate(prompt_texts, sampling)
    base_wall = time.time() - t0
    log.info(f"  Base batch done in {base_wall:.1f}s ({sum(len(o.outputs[0].token_ids) for o in base_outputs)/base_wall:.1f} tok/s aggregate)")

    # === Pair + write ===
    results = []
    for i, (prompt_text, user_turn, ref, base_o, ft_o, rid) in enumerate(
        zip(prompt_texts, user_turns, references, base_outputs, ft_outputs, row_ids), 1
    ):
        results.append({
            "id": rid,
            "prompt": user_turn,
            "reference": ref,
            "base_output": base_o.outputs[0].text.strip(),
            "base_tokens": len(base_o.outputs[0].token_ids),
            "finetune_output": ft_o.outputs[0].text.strip(),
            "finetune_tokens": len(ft_o.outputs[0].token_ids),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Also save aggregate perf for the client report
    perf = {
        "base_wall_s": round(base_wall, 2),
        "finetune_wall_s": round(ft_wall, 2),
        "base_tokens_total": sum(r["base_tokens"] for r in results),
        "finetune_tokens_total": sum(r["finetune_tokens"] for r in results),
        "base_tok_per_sec": round(sum(r["base_tokens"] for r in results) / base_wall, 1) if base_wall else 0,
        "finetune_tok_per_sec": round(sum(r["finetune_tokens"] for r in results) / ft_wall, 1) if ft_wall else 0,
        "engine": "vllm",
        "num_prompts": len(results),
        "max_new_tokens": max_new_tokens,
    }
    (out_path.parent / "samples_perf.json").write_text(json.dumps(perf, indent=2))

    # Free vLLM memory cleanly so judge step can load gpt-oss
    log.info("Tearing down vLLM engine to free memory")
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        # vLLM caches engine state in process — full cleanup may need waiting
        torch.cuda.synchronize()
    except Exception:
        pass

    log.info(f"Samples written to {out_path}  perf -> {out_path.parent / 'samples_perf.json'}")
    return out_path


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("eval_jsonl", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    generate_samples(args.adapter, args.eval_jsonl, args.out, cfg)
