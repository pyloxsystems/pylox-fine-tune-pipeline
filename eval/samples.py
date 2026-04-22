"""Generate inference samples on held-out eval prompts.

Loads base + LoRA adapter, generates fine-tuned outputs, disables adapter
to get base outputs on the same prompts. Writes side-by-side JSONL.
"""
from __future__ import annotations

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
    """Run both base and fine-tune inference on eval set. Write side-by-side."""
    from ops.gptoss_lifecycle import cycle_for_training
    from ops.memory_guard import ensure_headroom, estimate_model_gib
    cycle_for_training()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_model = cfg["base_model"]
    quant = cfg["quantization"]["bnb_4bit_quant_type"]
    ensure_headroom(needed_gib=estimate_model_gib(base_model, quant=quant), reserve_gib=5.0)

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    log.info(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    log.info(f"Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # Load eval prompts
    eval_rows = []
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                eval_rows.append(json.loads(line))

    max_prompts = cfg.get("eval", {}).get("sample_prompts", 10)
    eval_rows = eval_rows[:max_prompts]
    log.info(f"Generating on {len(eval_rows)} eval prompts")

    results = []
    for i, row in enumerate(eval_rows, 1):
        messages = row.get("messages", [])
        user_turn = next((m for m in messages if m.get("role") == "user"), None)
        if not user_turn:
            continue
        prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
        if not any(m.get("role") == "user" for m in prompt_msgs):
            continue

        inputs = tokenizer.apply_chat_template(
            prompt_msgs, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        # Fine-tuned output
        with torch.no_grad():
            t0 = time.time()
            out_ft = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
            ft_latency = time.time() - t0
        ft_text = tokenizer.decode(out_ft[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        # Base output (adapter disabled)
        with model.disable_adapter():
            with torch.no_grad():
                t0 = time.time()
                out_base = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
                base_latency = time.time() - t0
        base_text = tokenizer.decode(out_base[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        reference = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
        results.append({
            "id": row.get("id", str(i)),
            "prompt": user_turn["content"],
            "reference": reference,
            "base_output": base_text,
            "base_latency_s": round(base_latency, 2),
            "finetune_output": ft_text,
            "finetune_latency_s": round(ft_latency, 2),
        })
        log.info(f"  [{i}/{len(eval_rows)}] base={base_latency:.1f}s finetune={ft_latency:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Free memory for optional judge step
    del model
    del base
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"Samples written to {out_path}")
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
