"""Cloudflare DNS + Tunnel automation for pyloxsystems.com client subdomains.

Each client gets their own subdomain: client-{slug}.api.pyloxsystems.com
pointing to their deployment endpoint (RunPod pod or Spark tunnel).

Requires env vars:
    CLOUDFLARE_API_TOKEN  — scoped to zone:edit + tunnel:edit
    CLOUDFLARE_ZONE_ID    — the zone ID for pyloxsystems.com
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

log = logging.getLogger(__name__)

API_BASE = "https://api.cloudflare.com/client/v4"
DEFAULT_DOMAIN_ROOT = "api.pyloxsystems.com"


def _headers() -> dict:
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("Set CLOUDFLARE_API_TOKEN (dash.cloudflare.com → profile → API Tokens)")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _zone_id() -> str:
    zid = os.environ.get("CLOUDFLARE_ZONE_ID")
    if not zid:
        raise RuntimeError("Set CLOUDFLARE_ZONE_ID (zone overview page shows it)")
    return zid


def create_client_subdomain(
    client_slug: str,
    target: str,
    record_type: str = "CNAME",
    domain_root: str = DEFAULT_DOMAIN_ROOT,
    proxied: bool = True,
) -> dict:
    """Create a DNS record like `client-acme.api.pyloxsystems.com` → target.

    target can be:
      - hostname (CNAME) e.g. "xyz.proxy.runpod.net"
      - IP (A record) e.g. your colo static IP
    """
    name = f"client-{client_slug}.{domain_root}"
    existing = _find_record_by_name(name)

    payload = {
        "type": record_type,
        "name": name,
        "content": target,
        "ttl": 1,          # 1 = automatic
        "proxied": proxied,
        "comment": f"pylox client {client_slug}",
    }

    with httpx.Client(timeout=30) as http:
        if existing:
            r = http.put(
                f"{API_BASE}/zones/{_zone_id()}/dns_records/{existing['id']}",
                json=payload, headers=_headers(),
            )
        else:
            r = http.post(
                f"{API_BASE}/zones/{_zone_id()}/dns_records",
                json=payload, headers=_headers(),
            )
        r.raise_for_status()
        record = r.json()["result"]

    log.info(f"DNS {record_type} record for {name} → {target} ({'proxied' if proxied else 'direct'})")
    return {
        "subdomain": name,
        "url": f"https://{name}",
        "record_id": record["id"],
        "target": target,
    }


def _find_record_by_name(name: str) -> Optional[dict]:
    with httpx.Client(timeout=15) as http:
        r = http.get(
            f"{API_BASE}/zones/{_zone_id()}/dns_records",
            params={"name": name},
            headers=_headers(),
        )
        r.raise_for_status()
        records = r.json().get("result", [])
        return records[0] if records else None


def delete_client_subdomain(client_slug: str, domain_root: str = DEFAULT_DOMAIN_ROOT) -> bool:
    name = f"client-{client_slug}.{domain_root}"
    record = _find_record_by_name(name)
    if not record:
        log.info(f"No DNS record for {name}")
        return False

    with httpx.Client(timeout=30) as http:
        r = http.delete(
            f"{API_BASE}/zones/{_zone_id()}/dns_records/{record['id']}",
            headers=_headers(),
        )
        r.raise_for_status()
    log.info(f"Deleted DNS record for {name}")
    return True


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("client")
    c.add_argument("target")
    c.add_argument("--type", default="CNAME", choices=["A", "CNAME"])

    d = sub.add_parser("delete")
    d.add_argument("client")

    args = parser.parse_args()
    if args.cmd == "create":
        print(json.dumps(create_client_subdomain(args.client, args.target, record_type=args.type), indent=2))
    elif args.cmd == "delete":
        print("deleted" if delete_client_subdomain(args.client) else "no record found")
