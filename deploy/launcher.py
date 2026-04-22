"""Unified deploy entry point. Dispatches to spark or runpod."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

log = logging.getLogger(__name__)


def _upload_adapter_to_s3(adapter_path: Path, client_slug: str) -> str:
    """Optionally upload adapter to S3 for RunPod pull. Requires AWS creds + BUCKET env var."""
    bucket = os.environ.get("PYLOX_S3_BUCKET")
    if not bucket:
        raise RuntimeError("Set PYLOX_S3_BUCKET env var to upload adapter for RunPod.")

    import boto3
    tar_path = adapter_path.parent / f"{client_slug}_adapter.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(adapter_path, arcname=".")
    log.info(f"Packed adapter -> {tar_path} ({tar_path.stat().st_size // 1024} KB)")

    s3 = boto3.client("s3")
    key = f"adapters/{client_slug}/adapter.tar.gz"
    s3.upload_file(str(tar_path), bucket, key)
    url = f"s3://{bucket}/{key}"
    log.info(f"Uploaded to {url}")
    return url


def deploy_adapter(adapter_path: Path, client_slug: str, cfg: dict, target: str) -> str:
    """Deploy adapter to `target` (spark | runpod). Returns endpoint URL."""
    if target == "spark":
        from deploy.spark import deploy_client_endpoint
        endpoint = deploy_client_endpoint(client_slug, adapter_path, cfg)
        return endpoint

    if target == "runpod":
        from deploy.runpod import deploy_to_runpod
        adapter_url = _upload_adapter_to_s3(adapter_path, client_slug)
        result = deploy_to_runpod(client_slug, adapter_path, cfg, adapter_s3_url=adapter_url)
        return result.get("endpoint", "(pending — check RunPod dashboard)")

    raise ValueError(f"Unknown deploy target: {target}. Use 'spark' or 'runpod'.")


# Alias to match cli.py imports
def _runpod_client_stop(pod_id: str):
    from deploy.runpod import stop_pod
    return stop_pod(pod_id)


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("client", type=str)
    parser.add_argument("target", choices=["spark", "runpod"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    print(deploy_adapter(args.adapter, args.client, cfg, args.target))
