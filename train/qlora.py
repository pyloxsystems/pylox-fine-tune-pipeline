"""QLoRA (+ DoRA + NEFTune + packed sequences) training.

Config-driven. Reads the tier YAML, loads the base model, trains a LoRA
adapter on the supplied JSONL dataset. Designed for DGX Spark (Blackwell).

Key features baked in:
    - NF4 quantization via bitsandbytes
    - DoRA (weight-decomposed LoRA) when config.lora.use_dora=true
    - NEFTune noise injection via config.training.neftune_noise_alpha
    - Packed sequences via SFTTrainer packing=true
    - Gradient checkpointing (non-reentrant)
    - Paged AdamW 8-bit optimizer
    - Memory pre-flight check (respects ops.memory_guard hard limit)

Outputs `{client_dir}/adapter/` with adapter_config.json + adapter_model.safetensors.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _quantization_config(cfg: dict):
    import torch
    from transformers import BitsAndBytesConfig
    q = cfg["quantization"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map[q["bnb_4bit_compute_dtype"]],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def _lora_config(cfg: dict):
    from peft import LoraConfig
    l = cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        bias=l["bias"],
        task_type=l["task_type"],
        target_modules=l["target_modules"],
        use_dora=l.get("use_dora", False),
    )


def train(
    data_path: Path,
    client_dir: Path,
    cfg: dict,
    client_slug: str,
) -> Path:
    """Run full QLoRA training. Returns path to adapter directory."""
    from ops.gptoss_lifecycle import cycle_for_training
    from ops.memory_guard import ensure_headroom, estimate_model_gib

    # ALWAYS unload gpt-oss-120b before training, regardless of tier.
    # Prevents OOM when training model + KV cache + gpt-oss all want the same pool.
    cycle_for_training()

    base_model = cfg["base_model"]
    quant = cfg["quantization"]["bnb_4bit_quant_type"]
    needed = estimate_model_gib(base_model, quant=quant)
    log.info(f"Pre-flight memory check: need ~{needed:.1f} GiB for {base_model} ({quant})")
    ensure_headroom(needed_gib=needed, reserve_gib=5.0)

    # Heavy imports deferred until after memory check
    import torch
    from datasets import load_dataset
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_dir = client_dir / "adapter"
    run_dir = client_dir / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    log.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    log.info(f"Loading base model (quantized={quant})")
    attn_impl = cfg.get("attn_implementation", "sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_quantization_config(cfg),
        device_map={"": 0},                # force GPU, no CPU offload
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Load dataset (training module expects 'text' field already rendered)
    log.info(f"Loading dataset: {data_path}")
    ds = load_dataset("json", data_files=str(data_path), split="train")

    # Training args
    t = cfg["training"]
    sft_config = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_length=t["max_seq_length"],
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs=t.get("gradient_checkpointing_kwargs", {}),
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to="none",
        neftune_noise_alpha=t.get("neftune_noise_alpha"),
        packing=t.get("packing", False),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=_lora_config(cfg),
        processing_class=tokenizer,  # TRL 1.x renamed tokenizer -> processing_class
    )

    log.info(f"Starting training — {len(ds)} examples, {t['num_train_epochs']} epochs")
    trainer.train()

    # Save adapter to canonical location
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info(f"Adapter saved to {out_dir}")

    # Save run metadata
    meta = {
        "client": client_slug,
        "tier": cfg["tier"],
        "base_model": base_model,
        "data_path": str(data_path),
        "example_count": len(ds),
        "run_dir": str(run_dir),
        "adapter_path": str(out_dir),
        "config_snapshot": cfg,
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Free model memory before caller potentially loads another
    del trainer
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return out_dir


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Training JSONL (with 'text' field)")
    parser.add_argument("client_dir", type=Path)
    parser.add_argument("config", type=Path, help="Tier config YAML")
    parser.add_argument("client_slug", type=str)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    adapter = train(args.data, args.client_dir, cfg, args.client_slug)
    print(f"Adapter at: {adapter}")
