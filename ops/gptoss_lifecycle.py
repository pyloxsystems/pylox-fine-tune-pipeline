"""Lifecycle control for the local gpt-oss-120b TRT-LLM container.

Automatically started before enrichment (pipeline.py calls ensure_running).
Automatically stopped before training (train/qlora.py calls ensure_stopped).

This is conservative by default — any model load unloads gpt-oss-120b first,
regardless of size. Avoids surprise OOMs when gpt-oss + training model + KV
cache all want the same 110 GiB pool.

Container name: `trtllm-eagle3` (matches /home/fenexpertai/trtllm-eagle3/start.sh)
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

CONTAINER_NAME = "trtllm-eagle3"
START_SCRIPT = Path("/home/fenexpertai/trtllm-eagle3/start.sh")
HEALTH_URL = "http://localhost:8002/v1/models"

STARTUP_TIMEOUT_S = 300        # 5 min — cold load of 120B
SHUTDOWN_TIMEOUT_S = 60
HEALTH_POLL_S = 5


def _is_container_running(name: str = CONTAINER_NAME) -> bool:
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
        return name in out
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_healthy() -> bool:
    try:
        r = httpx.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_running() -> bool:
    """Start gpt-oss-120b container if not running. Wait for healthy endpoint."""
    if is_healthy():
        log.info("gpt-oss-120b already healthy — skipping start")
        return True

    if _is_container_running():
        log.info(f"Container '{CONTAINER_NAME}' running but endpoint not healthy — waiting...")
    else:
        if not START_SCRIPT.exists():
            raise FileNotFoundError(f"Start script not found: {START_SCRIPT}")
        log.info(f"Starting gpt-oss-120b via {START_SCRIPT}")
        subprocess.run(["bash", str(START_SCRIPT)], check=True)

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if is_healthy():
            log.info("gpt-oss-120b healthy ✓")
            return True
        time.sleep(HEALTH_POLL_S)

    raise TimeoutError(
        f"gpt-oss-120b did not become healthy within {STARTUP_TIMEOUT_S}s. "
        f"Check `docker logs {CONTAINER_NAME}`."
    )


def ensure_stopped() -> bool:
    """Stop gpt-oss-120b container. Frees ~70 GiB of GPU memory for training."""
    if not _is_container_running():
        log.info("gpt-oss-120b already stopped")
        return True

    log.info(f"Stopping {CONTAINER_NAME} to free memory for training...")
    try:
        subprocess.run(
            ["docker", "stop", "-t", str(SHUTDOWN_TIMEOUT_S), CONTAINER_NAME],
            check=True, timeout=SHUTDOWN_TIMEOUT_S + 10, capture_output=True,
        )
        log.info(f"Stopped {CONTAINER_NAME}")
    except subprocess.CalledProcessError as e:
        log.warning(f"docker stop returned non-zero: {e}")

    # Verify
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _is_container_running():
            log.info(f"{CONTAINER_NAME} confirmed stopped")
            # Wait a beat for UMA allocator to release pages
            time.sleep(5)
            return True
        time.sleep(2)

    raise TimeoutError(f"{CONTAINER_NAME} did not stop in time")


def cycle_for_training() -> None:
    """Canonical pre-training hook: stop gpt-oss, wait for memory to drain."""
    from ops.memory_guard import gpu_state
    before = gpu_state()
    log.info(f"Pre-training memory: {before.used_gib:.1f}/{before.total_gib:.1f} GiB used")
    ensure_stopped()
    after = gpu_state()
    log.info(f"Post-stop memory: {after.used_gib:.1f}/{after.total_gib:.1f} GiB used "
             f"(freed {before.used_gib - after.used_gib:.1f} GiB)")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status", "cycle"])
    args = parser.parse_args()

    if args.action == "start":
        ensure_running()
    elif args.action == "stop":
        ensure_stopped()
    elif args.action == "status":
        running = _is_container_running()
        healthy = is_healthy()
        print(f"container_running={running}  endpoint_healthy={healthy}")
    elif args.action == "cycle":
        cycle_for_training()
