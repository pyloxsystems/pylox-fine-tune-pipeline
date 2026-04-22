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

# We track PIDs of trtllm-serve process(es) at startup. On Spark, the container
# is launched with --ipc=host --network=host --runtime=nvidia, which gives it
# enough host privilege that docker stop/kill no longer work — even with root.
# Only kernel-level SIGKILL on the actual PIDs reliably tears it down.
PID_TRACKER = Path(__file__).parent / ".trtllm_pids"

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


def _record_pids() -> None:
    """Record PIDs of trtllm-serve + container's main bash so we can kill them later."""
    pids: list[int] = []
    try:
        # Container main PID
        out = subprocess.run(
            ["docker", "inspect", CONTAINER_NAME, "--format", "{{.State.Pid}}"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        if out and out.isdigit():
            pids.append(int(out))
    except subprocess.SubprocessError:
        pass

    # All trtllm-serve and child python processes
    try:
        out = subprocess.run(
            ["pgrep", "-f", "trtllm-serve"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        for line in out.splitlines():
            if line.isdigit():
                pids.append(int(line))
    except subprocess.SubprocessError:
        pass

    if pids:
        PID_TRACKER.write_text("\n".join(str(p) for p in sorted(set(pids))))
        log.info(f"Tracking PIDs for kill: {sorted(set(pids))}")


def ensure_running() -> bool:
    """Start gpt-oss-120b container if not running. Verifies memory headroom first."""
    if is_healthy():
        log.info("gpt-oss-120b already healthy — skipping start")
        _record_pids()
        return True

    if _is_container_running():
        log.info(f"Container '{CONTAINER_NAME}' running but endpoint not healthy — waiting...")
    else:
        # ALWAYS check memory before launching the 120B model.
        # gpt-oss-120b @ MXFP4 needs ~70 GiB. Adding 8 GiB reserve for KV cache.
        from ops.memory_guard import ensure_headroom
        ensure_headroom(needed_gib=70.0, reserve_gib=8.0)

        if not START_SCRIPT.exists():
            raise FileNotFoundError(f"Start script not found: {START_SCRIPT}")
        log.info(f"Starting gpt-oss-120b via {START_SCRIPT}")
        subprocess.run(["bash", str(START_SCRIPT)], check=True)

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if is_healthy():
            log.info("gpt-oss-120b healthy ✓")
            _record_pids()
            return True
        time.sleep(HEALTH_POLL_S)

    raise TimeoutError(
        f"gpt-oss-120b did not become healthy within {STARTUP_TIMEOUT_S}s. "
        f"Check `docker logs {CONTAINER_NAME}`."
    )


def _kill_tracked_pids() -> bool:
    """Kill the PIDs we recorded at container startup. Bypasses Docker entirely."""
    if not PID_TRACKER.exists():
        return False
    pids = [int(p) for p in PID_TRACKER.read_text().split() if p.strip().isdigit()]
    if not pids:
        return False

    log.info(f"Killing tracked PIDs: {pids}")
    killed_any = False
    for pid in pids:
        # Try without sudo first (works if container ran as user)
        try:
            subprocess.run(["kill", "-9", str(pid)], check=True, capture_output=True, timeout=5)
            killed_any = True
            continue
        except subprocess.SubprocessError:
            pass
        # Sudo (works if /etc/sudoers.d/pylox-trtllm allows kill)
        try:
            subprocess.run(["sudo", "-n", "kill", "-9", str(pid)],
                           check=True, capture_output=True, timeout=5)
            killed_any = True
        except subprocess.SubprocessError as e:
            log.debug(f"Could not kill PID {pid}: {e}")

    if killed_any:
        time.sleep(3)
        PID_TRACKER.unlink(missing_ok=True)
    return killed_any


def ensure_stopped() -> bool:
    """Stop gpt-oss-120b. Frees ~70 GiB of GPU memory for training.

    Cascade (each step only runs if prior failed):
      1. Direct PID kill of tracked trtllm-serve processes (most reliable on Spark)
      2. docker stop -t 60 (graceful)
      3. docker kill (force)
      4. sudo -n docker rm -f (if passwordless sudo configured)
    """
    if not _is_container_running() and not is_healthy():
        log.info("gpt-oss-120b already stopped")
        PID_TRACKER.unlink(missing_ok=True)
        return True

    log.info(f"Stopping {CONTAINER_NAME} to free memory...")

    # Step 1: kill tracked PIDs (works even when docker can't reach the container)
    if _kill_tracked_pids():
        # Wait for docker to notice
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if not _is_container_running():
                log.info(f"{CONTAINER_NAME} stopped via tracked PID kill")
                time.sleep(5)         # let UMA allocator drain
                return True
            time.sleep(2)

    # Step 2: docker stop (graceful)
    try:
        result = subprocess.run(
            ["docker", "stop", "-t", str(SHUTDOWN_TIMEOUT_S), CONTAINER_NAME],
            timeout=SHUTDOWN_TIMEOUT_S + 30, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning(f"docker stop returned {result.returncode}: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        log.warning(f"docker stop exceeded timeout — falling back to docker kill")

    # Verify with short polling window
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _is_container_running():
            log.info(f"{CONTAINER_NAME} confirmed stopped")
            time.sleep(5)          # let UMA allocator release pages
            return True
        time.sleep(2)

    # Fallback 1: docker kill (may fail on nvidia-runtime + AppArmor hosts)
    log.warning(f"{CONTAINER_NAME} still running after stop — trying docker kill")
    try:
        subprocess.run(
            ["docker", "kill", CONTAINER_NAME],
            timeout=30, capture_output=True, check=True,
        )
    except subprocess.SubprocessError as e:
        log.warning(f"docker kill failed: {e}")

    # Fallback 2: passwordless sudo docker rm -f (recommended config — see README)
    if _is_container_running():
        log.warning("Trying `sudo -n docker rm -f` (requires sudoers config — see README)")
        try:
            subprocess.run(
                ["sudo", "-n", "docker", "rm", "-f", CONTAINER_NAME],
                timeout=30, capture_output=True, check=True,
            )
            log.info(f"{CONTAINER_NAME} removed via sudo docker rm -f")
        except subprocess.SubprocessError as e:
            log.warning(f"sudo docker rm -f failed: {e}")

    # Fallback 3: find container PID and SIGKILL directly
    if _is_container_running():
        log.warning(f"Falling back to direct PID kill")
        try:
            pid_out = subprocess.run(
                ["docker", "inspect", CONTAINER_NAME, "--format", "{{.State.Pid}}"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
            if pid_out and pid_out.isdigit():
                pid = int(pid_out)
                try:
                    subprocess.run(["kill", "-9", str(pid)], check=True, capture_output=True, timeout=5)
                except subprocess.SubprocessError:
                    subprocess.run(["sudo", "-n", "kill", "-9", str(pid)],
                                   check=False, capture_output=True, timeout=5)
        except subprocess.SubprocessError as e:
            log.error(f"Direct PID kill also failed: {e}")

    # Final verify
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _is_container_running():
            log.info(f"{CONTAINER_NAME} killed")
            time.sleep(5)
            return True
        time.sleep(2)

    raise TimeoutError(
        f"{CONTAINER_NAME} could not be stopped. "
        f"Manual intervention required:  sudo docker rm -f {CONTAINER_NAME}  "
        f"or  sudo kill -9 $(docker inspect {CONTAINER_NAME} --format '{{{{.State.Pid}}}}')"
    )


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
