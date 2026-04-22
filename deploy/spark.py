"""Deploy a client adapter on the local DGX Spark via vLLM.

Production behavior:
  1. Stop gpt-oss-120b (free memory)
  2. Stop any prior pylox-vllm process (clean slate)
  3. Stop any prior vllm-clients tracker
  4. Launch `vllm serve` as a systemd-style background process with logs
  5. Wait for /health endpoint to return 200
  6. Verify the LoRA adapter mount works by issuing a test completion
  7. Return the live endpoint URL

Multi-tenant: shared base + per-client LoRA adapters mounted via --lora-modules.
Adding/removing a client triggers a single restart.

State: clients/deployed_adapters.json tracks active mounts.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

import ops.cuda_compat       # noqa: F401  — vllm subprocess inherits LD_LIBRARY_PATH

log = logging.getLogger(__name__)

SPARK_VLLM_PORT = 8010
PIPELINE_ROOT = Path(__file__).parent.parent
DEPLOY_DIR = PIPELINE_ROOT / "deploy" / "runs"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

PIDFILE = DEPLOY_DIR / "vllm.pid"
LOGFILE = DEPLOY_DIR / "vllm.log"
ADAPTERS_STATE = DEPLOY_DIR / "deployed_adapters.json"

VLLM_BOOT_TIMEOUT_S = 300


def _read_adapters() -> dict[str, str]:
    if ADAPTERS_STATE.exists():
        return json.loads(ADAPTERS_STATE.read_text())
    return {}


def _write_adapters(adapters: dict[str, str]) -> None:
    ADAPTERS_STATE.write_text(json.dumps(adapters, indent=2))


def _vllm_pid() -> int | None:
    if not PIDFILE.exists():
        return None
    try:
        pid = int(PIDFILE.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def _stop_vllm() -> None:
    """Stop the pipeline's vllm serve if running."""
    pid = _vllm_pid()
    if pid:
        log.info(f"Stopping existing pylox-vllm (pid {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(1)
        else:
            log.warning(f"vllm pid {pid} didn't exit on SIGTERM, sending SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(3)
        PIDFILE.unlink(missing_ok=True)


def _build_command(base_model: str, adapters: dict[str, str], cfg: dict) -> list[str]:
    cmd = [
        "vllm", "serve", base_model,
        "--host", "0.0.0.0",
        "--port", str(SPARK_VLLM_PORT),
    ]
    cmd.extend(cfg["deploy"]["spark"]["vllm_args"])
    if adapters:
        cmd.append("--lora-modules")
        cmd.extend(f"{name}={Path(path).resolve()}" for name, path in adapters.items())
    return cmd


def _wait_for_health(url: str, timeout_s: int = VLLM_BOOT_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=3)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become healthy at {url} within {timeout_s}s")


def _verify_adapter(client_slug: str, base_model: str) -> None:
    url = f"http://localhost:{SPARK_VLLM_PORT}/v1/chat/completions"
    payload = {
        "model": client_slug,                 # vLLM treats LoRA mount name as a model id
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    r = httpx.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Adapter test request failed: {r.status_code} {r.text[:200]}")
    log.info(f"Adapter '{client_slug}' verified — test completion succeeded")


def deploy_client_endpoint(
    client_slug: str,
    adapter_path: Path,
    cfg: dict,
) -> str:
    """Stop old vLLM, mount this client + all prior clients, restart, verify."""
    from ops.gptoss_lifecycle import ensure_stopped
    from ops.memory_guard import ensure_headroom, estimate_model_gib

    ensure_stopped()
    _stop_vllm()

    # Verify we have headroom for vLLM + adapters BEFORE launching subprocess.
    # Raises MemoryError if not enough — fail-fast instead of OOM-killing later.
    base_model = cfg["base_model"]
    needed_for_vllm = max(estimate_model_gib(base_model, quant="bf16"), 16.0)
    # Plus rough KV cache budget per concurrent client adapter
    adapters_count = len(_read_adapters()) + 1
    needed_for_kv = 2.0 * adapters_count
    ensure_headroom(needed_gib=needed_for_vllm + needed_for_kv, reserve_gib=8.0)

    adapters = _read_adapters()
    adapters[client_slug] = str(adapter_path.resolve())
    _write_adapters(adapters)

    cmd = _build_command(cfg["base_model"], adapters, cfg)
    log.info(f"Launching vLLM with {len(adapters)} adapter(s): {list(adapters.keys())}")
    log.debug("CMD: " + " ".join(shlex.quote(x) for x in cmd))

    log_fh = LOGFILE.open("a")
    log_fh.write(f"\n\n=== vllm boot {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,             # detach from current process group
    )
    PIDFILE.write_text(str(proc.pid))
    log.info(f"vLLM started (pid {proc.pid}). Logs: {LOGFILE}")

    health_url = f"http://localhost:{SPARK_VLLM_PORT}/health"
    _wait_for_health(health_url)
    log.info(f"vLLM healthy on {health_url}")

    _verify_adapter(client_slug, cfg["base_model"])

    return f"http://localhost:{SPARK_VLLM_PORT}/v1"


def stop_client_endpoint(client_slug: str) -> None:
    """Remove a client from the mounted adapter set + restart vLLM (if other clients remain)."""
    adapters = _read_adapters()
    if client_slug not in adapters:
        log.info(f"Client {client_slug} not deployed")
        return
    del adapters[client_slug]
    _write_adapters(adapters)

    _stop_vllm()
    if not adapters:
        log.info("No remaining clients — vLLM stopped, not restarting")
        return

    # Restart with remaining adapters
    cmd = _build_command(_first_cfg_base(), adapters, _first_cfg())
    log_fh = LOGFILE.open("a")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    PIDFILE.write_text(str(proc.pid))
    _wait_for_health(f"http://localhost:{SPARK_VLLM_PORT}/health")
    log.info(f"vLLM restarted with {len(adapters)} remaining client(s)")


def _first_cfg() -> dict:
    """Hack: read the tier-8b config to get reasonable defaults for restart.
    For multi-tier deployments, we'd track per-client tier in deployed_adapters.json
    and pick the highest-tier base model."""
    import yaml
    return yaml.safe_load((PIPELINE_ROOT / "configs" / "tier-8b-llama.yml").read_text())


def _first_cfg_base() -> str:
    return _first_cfg()["base_model"]


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", type=str)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    print(deploy_client_endpoint(args.client, args.adapter, cfg))
