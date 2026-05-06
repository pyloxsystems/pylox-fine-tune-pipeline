# Pylox Systems — Fine-Tune Pipeline

Production pipeline for on-premises LLM fine-tuning and deployment, purpose-built for the NVIDIA DGX Spark (Blackwell GB10, 128 GB unified memory).

One repo, one CLI, tier-standardized configs for 8B / 32B / 70B. Each client engagement is a config change plus an SFTP upload. Training, enrichment, eval, and deployment are all idempotent and resumable.

**Compliance-safe by architecture:** client data never leaves the Spark. Claude Code designs preprocessors using only sanitized samples. Local gpt-oss-120b handles full-corpus enrichment. Anthropic/OpenAI/Together never see a client record.

---

## Quick start

```bash
pip install -r requirements.txt

# One-shot onboarding: intake → enrich → train → eval → deploy
python cli.py onboard \
    --client acme-health \
    --tier 8b \
    --data ./acme-corpus.jsonl \
    --target runpod

# Individual stages (all idempotent)
python cli.py intake    --client acme-health --tier 8b --data ./corpus.jsonl
python cli.py enrich    --client acme-health
python cli.py train     --client acme-health
python cli.py evaluate  --client acme-health
python cli.py deploy    --client acme-health --target runpod

# Status + lifecycle
python cli.py status                      # all clients
python cli.py status   --client acme-health
python cli.py refresh  --client acme-health --data ./new-data.jsonl
python cli.py teardown --client acme-health
```

---

## Architecture

```
Client data (sensitive)
    │
    ▼
┌──────────────────────────────────────────────┐
│ INTAKE                                       │
│   intake/validate.py   — schema detect       │
│   intake/format_chat.py — chat template      │
│   intake/sanitize_sample.py — 20-row sample  │
│                         for Claude Code      │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│ ENRICHMENT (local gpt-oss-120b :8002)        │
│   enrich/dedup.py    — Nomic semantic dedup  │
│   enrich/quality.py  — 5-dim LLM scoring     │
│   enrich/redact.py   — LLM-based NER PII     │
│   enrich/augment.py  — optional paraphrase   │
└─────────────────┬────────────────────────────┘
                  │ (auto-stops gpt-oss before training)
                  ▼
┌──────────────────────────────────────────────┐
│ TRAINING                                     │
│   train/qlora.py                             │
│     - NF4 quantization                       │
│     - DoRA (weight-decomposed LoRA)          │
│     - NEFTune noise injection                │
│     - Packed sequences                       │
│     - Gradient checkpointing (non-reentrant) │
│     - Paged AdamW 8-bit                      │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│ EVALUATION                                   │
│   eval/samples.py   — base vs fine-tune     │
│   eval/llm_judge.py — gpt-oss picks winner   │
│   eval/runner.py    — renders markdown PDF   │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│ DEPLOYMENT                                   │
│   deploy/runpod.py  — rent GPU near user     │
│   deploy/spark.py   — host on your Spark     │
│   deploy/launcher.py — target dispatch       │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│ PROVISIONING                                 │
│   ops/api_keys.py   — SHA-256-hashed keys    │
│   ops/cloudflare_dns.py — subdomain per client│
│   ops/stripe_billing.py — setup + monthly    │
│   ops/provision_endpoint.py — orchestrator   │
└──────────────────────────────────────────────┘
```

---

## Tier configs

Every client picks one of three tiers. Each tier is a tuned YAML under `configs/`:

| Tier | Base model | LoRA rank | EAGLE-3 | Setup fee | Monthly | Best for |
|---|---|---|---|---|---|---|
| 8B | Llama 3.1 8B Instruct | 16 | off (vanilla spec) | $3,000 | $800 | small gigs, consumer GPU clients |
| 32B | Gemma 3 27B Instruct | 32 | on | $7,000 | $1,800 | **lead offering**, single L40S |
| 70B | Llama 3.1 70B Instruct | 32 | on | $15,000 | $3,500 | latency-critical, quality-maxed |

Each tier bakes in best-practice training techniques (DoRA, NEFTune, packing, prefix caching, EAGLE-3 speculative decoding).

---

## The moat

Three things most contractors can't offer together:

1. **Owned hardware** — DGX Spark (Blackwell GB10, 128 GB unified memory). No cloud markup.
2. **On-prem gpt-oss-120b** — data enrichment runs on local hardware. Client PHI/privilege/NDA data never leaves the Spark. Zero third-party API exposure.
3. **EAGLE-3 + DoRA + NEFTune** — stacked optimizations most Upwork freelancers don't touch. Measurable throughput + quality gains.

Combined, these let us offer a compliance-safe story (BAA/privilege/TOS-compatible) that cloud-based competitors can't.

---

## Memory management

DGX Spark has 128 GB unified memory (CPU + GPU share the same pool). We enforce a **hard 110 GiB soft-ceiling** in `ops/memory_guard.py` to avoid allocator thrashing.

Model swap pattern:
- Enrichment step → `gpt-oss-120b` loaded (~70 GiB)
- Training step → auto-stops `gpt-oss-120b` first, then loads training base model
- Eval samples → unloads again, loads base + adapter
- Eval judge → reloads `gpt-oss-120b` for comparison
- Deploy → unloads all, starts vLLM

This is handled automatically by `ops/gptoss_lifecycle.py`. You don't have to manage it manually.

---

## Environment variables

For full functionality, set these (all optional for dev):

```bash
# Required for production deploys
export HF_TOKEN=hf_...                   # HuggingFace model downloads
export RUNPOD_API_KEY=...                # RunPod GPU deployment
export PYLOX_S3_BUCKET=pylox-adapters    # S3 for adapter storage + RunPod pulls

# Required for full client provisioning
export STRIPE_SECRET_KEY=sk_live_...     # billing
export CLOUDFLARE_API_TOKEN=...          # subdomain DNS
export CLOUDFLARE_ZONE_ID=...            # for pyloxsystems.com
```

Without these, the pipeline runs in "dev mode" — skips billing/DNS, uses local endpoints only.

---

## Hardware requirements

### Minimum (dev / 8B tier)
- Any 16 GB+ NVIDIA GPU (RTX 4080 / L4 / L40S / Blackwell)
- 32 GB RAM
- 500 GB disk (models + adapters + client data)

### Recommended (production / 32B tier)
- NVIDIA DGX Spark (Blackwell GB10) or H100 80GB
- 128 GB RAM (unified on Spark, dedicated on H100 host)
- 2 TB NVMe
- Cloudflare account + registered domain

### Premium (70B tier + multi-tenant hosting)
- DGX Spark + colocation (Miami Equinix or Montreal Cologix)
- Additional Sparks for redundancy
- Or H100 + datacenter GPUs

---

## Pricing model

**Setup fee** (one-time, 50/50 upfront + on delivery):
- 8B: $3,000
- 32B: $7,000
- 70B: $15,000

**Monthly hosting** (recurring):
- 8B: $800/mo
- 32B: $1,800/mo
- 70B: $3,500/mo

**Fine-tune refresh** (new data, same base):
- 8B: $1,500
- 32B: $2,500
- 70B: $5,000

---

## License

PolyForm-Noncommercial-1.0.0 — see [LICENSE](LICENSE).

Copyright © 2026 Emilio Girard / Pylox Systems.
