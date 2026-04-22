"""Memory guard for DGX Spark (unified memory architecture).

Hard limit: **110 GiB total GPU/CPU memory across all workloads.** Above this
the unified-memory allocator starts thrashing during KV cache pressure.
Loading a new model into an already-hot system is the primary risk.

DGX Spark uses UMA — CPU RAM and GPU memory are the same 128 GiB pool.
We read `/proc/meminfo` as the source of truth. On a traditional discrete
GPU host, we fall back to `nvidia-smi memory.*` fields.

Usage:
    from ops.memory_guard import ensure_headroom, gpu_state

    ensure_headroom(needed_gib=30, reserve_gib=5)   # raises MemoryError if insufficient
    state = gpu_state()
    print(f"Used: {state['used_gib']:.1f} / {state['total_gib']:.1f}")
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HARD_LIMIT_GIB = 110.0    # Per user rule. Do not raise this.

MEMINFO = Path("/proc/meminfo")


@dataclass
class GpuState:
    total_gib: float
    used_gib: float
    free_gib: float
    used_pct: float
    source: str            # "uma" or "nvidia-smi"

    def as_dict(self) -> dict:
        return {
            "total_gib": round(self.total_gib, 2),
            "used_gib": round(self.used_gib, 2),
            "free_gib": round(self.free_gib, 2),
            "used_pct": round(self.used_pct, 1),
            "source": self.source,
        }


def _uma_state() -> GpuState:
    """Read /proc/meminfo. On UMA, this IS GPU memory."""
    fields: dict[str, int] = {}
    with MEMINFO.open() as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val_parts = parts[1].strip().split()
            if val_parts and val_parts[0].isdigit():
                fields[key] = int(val_parts[0])  # kB

    total_kb = fields["MemTotal"]
    available_kb = fields.get("MemAvailable", fields.get("MemFree", 0))
    used_kb = total_kb - available_kb

    total_gib = total_kb / (1024 * 1024)
    used_gib = used_kb / (1024 * 1024)
    free_gib = available_kb / (1024 * 1024)

    return GpuState(
        total_gib=total_gib,
        used_gib=used_gib,
        free_gib=free_gib,
        used_pct=(used_gib / total_gib * 100) if total_gib else 0,
        source="uma",
    )


def _nvidia_smi_state() -> Optional[GpuState]:
    """Parse `nvidia-smi`. Returns None if unavailable or values are [N/A] (Spark UMA case)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    total_mib = used_mib = free_mib = 0
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                total_mib += int(parts[0])
                used_mib += int(parts[1])
                free_mib += int(parts[2])
            except ValueError:
                # "[N/A]" — Spark UMA reports this
                return None

    if total_mib == 0:
        return None

    total_gib = total_mib / 1024
    used_gib = used_mib / 1024
    free_gib = free_mib / 1024
    return GpuState(
        total_gib=total_gib,
        used_gib=used_gib,
        free_gib=free_gib,
        used_pct=(used_gib / total_gib * 100) if total_gib else 0,
        source="nvidia-smi",
    )


def gpu_state() -> GpuState:
    """Return current memory state. Tries nvidia-smi first, falls back to /proc/meminfo UMA."""
    s = _nvidia_smi_state()
    if s:
        return s
    if MEMINFO.exists():
        return _uma_state()
    raise RuntimeError("No memory source available (nvidia-smi returned N/A and /proc/meminfo missing)")


def ensure_headroom(
    needed_gib: float,
    reserve_gib: float = 5.0,
    poll_seconds: Optional[int] = None,
    hard_limit_gib: float = HARD_LIMIT_GIB,
) -> GpuState:
    """Verify `needed_gib + reserve_gib` is free AND total used would stay under hard limit.

    If poll_seconds is set and space is insufficient, poll until free or timeout.
    Otherwise raise MemoryError immediately.

    Returns current GpuState on success.
    """
    deadline = time.monotonic() + poll_seconds if poll_seconds else None

    while True:
        state = gpu_state()
        required = needed_gib + reserve_gib
        used_after_load = state.used_gib + needed_gib

        ok_free = state.free_gib >= required
        ok_limit = used_after_load <= hard_limit_gib

        if ok_free and ok_limit:
            log.info(
                f"Memory OK ({state.source}): need {needed_gib:.1f} + reserve {reserve_gib:.1f} "
                f"<= free {state.free_gib:.1f} GiB. Hard-limit post-load: "
                f"{used_after_load:.1f} / {hard_limit_gib:.1f} GiB."
            )
            return state

        msg = (
            f"Insufficient memory ({state.source}). Need {needed_gib:.1f} + reserve "
            f"{reserve_gib:.1f} GiB. Free: {state.free_gib:.1f}, Used: {state.used_gib:.1f}, "
            f"Post-load projection: {used_after_load:.1f} (hard limit {hard_limit_gib})."
        )
        log.warning(msg)

        if deadline is None or time.monotonic() >= deadline:
            raise MemoryError(msg)

        log.info(f"Polling every 10s for memory (remaining {int(deadline - time.monotonic())}s)…")
        time.sleep(10)


MODEL_FOOTPRINT_GIB = {
    "meta-llama/Llama-3.1-8B-Instruct": 6.0,
    "meta-llama/Llama-3.1-70B-Instruct": 45.0,
    "meta-llama/Llama-3.2-1B-Instruct": 1.5,
    "google/gemma-3-27b-it": 20.0,
    "google/gemma-3-1b-it": 1.5,
    "Qwen/Qwen3-32B": 22.0,
    "openai/gpt-oss-120b": 70.0,
}


def estimate_model_gib(base_model: str, quant: str = "nf4") -> float:
    """Return expected VRAM GiB for `base_model` at given quantization. Falls back
    to 8 GiB for unknown models (conservative)."""
    base = MODEL_FOOTPRINT_GIB.get(base_model, 8.0)
    if quant == "nf4":
        return base
    if quant == "fp8":
        return base * 1.5
    if quant == "bf16":
        return base * 3.0
    return base


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    state = gpu_state()
    print(f"Memory state: {state.as_dict()}")
    print(f"Hard limit: {HARD_LIMIT_GIB:.0f} GiB")
    print(f"Headroom under limit: {HARD_LIMIT_GIB - state.used_gib:.1f} GiB")
