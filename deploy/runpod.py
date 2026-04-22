"""Deploy a client adapter to RunPod Secure Cloud.

Uses RunPod's REST API + a boot script that:
  1. Installs vLLM
  2. Pulls base model from HF
  3. Pulls client adapter (from S3 or direct upload)
  4. Launches `vllm serve` with EAGLE-3 + LoRA

Requires env var RUNPOD_API_KEY.

For first-gig simplicity we deploy one pod per client (isolated billing,
easy teardown). Multi-tenant shared-base is the optimization to add at
client #5+ once we understand their usage patterns.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

API_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("Set RUNPOD_API_KEY env var (see runpod.io/user/settings)")
    return key


def _gpu_type_id_from_name(name: str) -> str:
    """Map config-friendly GPU name -> RunPod type ID. Update as RunPod catalog evolves."""
    # These IDs shift over time. Check https://rest.runpod.io/v1/gpu-types when in doubt.
    mapping = {
        "NVIDIA L4":            "NVIDIA L4",
        "NVIDIA L40S":          "NVIDIA L40S",
        "NVIDIA A100 80GB PCIe": "NVIDIA A100 80GB PCIe",
        "NVIDIA H100 PCIe 80GB": "NVIDIA H100 80GB HBM3",
        "NVIDIA RTX A6000":     "NVIDIA RTX A6000",
    }
    return mapping.get(name, name)


def _boot_script(base_model: str, adapter_s3_url: str, port: int, vllm_args: list[str]) -> str:
    quoted_args = " ".join(vllm_args)
    return f"""#!/bin/bash
set -e
pip install -q -U vllm huggingface_hub
mkdir -p /workspace/adapter

if [ -n "${{HF_TOKEN}}" ]; then
  huggingface-cli login --token "${{HF_TOKEN}}"
fi

if [[ "{adapter_s3_url}" == s3://* ]]; then
  pip install -q awscli
  aws s3 sync "{adapter_s3_url}" /workspace/adapter
elif [[ "{adapter_s3_url}" == https://* ]]; then
  curl -L "{adapter_s3_url}" -o /workspace/adapter.tgz
  tar xzf /workspace/adapter.tgz -C /workspace/adapter
fi

exec vllm serve "{base_model}" \\
    --host 0.0.0.0 --port {port} \\
    --lora-modules client=/workspace/adapter \\
    {quoted_args}
"""


def deploy_to_runpod(
    client_slug: str,
    adapter_path: Path,
    cfg: dict,
    adapter_s3_url: Optional[str] = None,
) -> dict:
    """Create a pod serving this client's adapter. Returns pod metadata + endpoint URL.

    If adapter_s3_url is None, the caller is expected to upload the adapter to S3
    first and pass the URL. This module does NOT upload — keep storage concerns
    in ops/provision_endpoint.py.
    """
    if adapter_s3_url is None:
        raise ValueError("Must upload adapter to S3 and provide adapter_s3_url")

    base_model = cfg["base_model"]
    deploy_cfg = cfg["deploy"]["runpod"]
    gpu_name = _gpu_type_id_from_name(deploy_cfg["gpu"])
    vllm_args = deploy_cfg["vllm_args"]

    boot = _boot_script(base_model, adapter_s3_url, 8000, vllm_args)

    payload = {
        "name": f"pylox-{client_slug}",
        "imageName": "runpod/pytorch:2.3.0-py3.10-cuda12.1.0",
        "gpuTypeIds": [gpu_name],
        "containerDiskInGb": 50,
        "ports": "8000/http",
        "env": [
            {"key": "HF_TOKEN", "value": os.environ.get("HF_TOKEN", "")},
        ],
        "dockerStartCmd": ["bash", "-c", boot],
    }

    headers = {"Authorization": f"Bearer {_api_key()}"}
    log.info(f"Creating RunPod pod for {client_slug} with GPU={gpu_name}")

    with httpx.Client(timeout=60) as http:
        r = http.post(f"{REST_URL}/pods", json=payload, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"RunPod create failed ({r.status_code}): {r.text}")
        pod = r.json()

    pod_id = pod.get("id") or pod.get("podId")
    log.info(f"Pod created: {pod_id}. Waiting for RUNNING state...")

    endpoint_url = None
    with httpx.Client(timeout=30) as http:
        for _ in range(60):
            time.sleep(10)
            r = http.get(f"{REST_URL}/pods/{pod_id}", headers=headers)
            if r.status_code != 200:
                continue
            p = r.json()
            status = p.get("desiredStatus") or p.get("status")
            if status and status.upper() == "RUNNING":
                endpoint_url = _extract_endpoint(p)
                if endpoint_url:
                    break

    if not endpoint_url:
        log.warning("Pod is running but proxy URL not yet exposed. Check RunPod dashboard.")

    return {
        "provider": "runpod",
        "pod_id": pod_id,
        "endpoint": endpoint_url,
        "base_model": base_model,
        "gpu": gpu_name,
    }


def _extract_endpoint(pod: dict) -> Optional[str]:
    ports = pod.get("ports") or []
    for p in ports:
        if "publicPort" in p and p.get("type") == "http":
            return f"https://{pod['id']}-{p['publicPort']}.proxy.runpod.net/v1"
    # Alt shape
    proxy = pod.get("proxyUrl") or pod.get("runpodProxyUrl")
    if proxy:
        return f"{proxy.rstrip('/')}/v1"
    return None


def stop_pod(pod_id: str) -> None:
    headers = {"Authorization": f"Bearer {_api_key()}"}
    with httpx.Client(timeout=30) as http:
        r = http.delete(f"{REST_URL}/pods/{pod_id}", headers=headers)
        if r.status_code >= 400:
            log.warning(f"Stop pod {pod_id} returned {r.status_code}: {r.text}")


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", type=str)
    parser.add_argument("adapter_url", type=str, help="s3:// or https:// URL to adapter tarball")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    result = deploy_to_runpod(args.client, Path("/n/a"), cfg, adapter_s3_url=args.adapter_url)
    print(json.dumps(result, indent=2))
