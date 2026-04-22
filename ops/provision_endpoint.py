"""Provision a complete client endpoint: API key + DNS subdomain + Stripe billing.

Called from cli.py after `deploy` completes. Ties together the three ops modules
into a single onboarding flow:

    1. Generate per-client API key
    2. Create Cloudflare DNS subdomain pointing to deployment endpoint
    3. Set up Stripe customer + setup invoice + monthly subscription
    4. Return a consolidated dict for storage in state.json + the handoff email

Every step is best-effort — if Stripe credentials aren't set, skip billing and
warn. That way dev/smoke runs work without requiring real accounts.
"""
from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import urlparse

log = logging.getLogger(__name__)


def provision(
    client_slug: str,
    tier_config: dict,
    deployment_endpoint: str,
    contact_email: Optional[str] = None,
    contact_name: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Return a dict with api_key, public_url, stripe_*, subscription_id."""
    result: dict = {"client_slug": client_slug}

    # 1. API key — always works (local SQLite)
    from ops.api_keys import generate_key
    if dry_run:
        result["api_key"] = "psk_DRYRUN_NOT_REAL"
    else:
        result["api_key"] = generate_key(client_slug, metadata=f"tier={tier_config.get('tier')}")

    # 2. Cloudflare DNS — optional (requires CLOUDFLARE_API_TOKEN)
    public_url = None
    if os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ZONE_ID"):
        from ops.cloudflare_dns import create_client_subdomain
        try:
            parsed = urlparse(deployment_endpoint)
            target = parsed.netloc or parsed.path.lstrip("/")
            if ":" in target:
                target = target.split(":")[0]      # strip port
            is_ip = target.replace(".", "").isdigit()
            record_type = "A" if is_ip else "CNAME"
            dns_result = create_client_subdomain(client_slug, target, record_type=record_type)
            public_url = dns_result["url"]
            result["dns"] = dns_result
        except Exception as e:
            log.warning(f"Cloudflare DNS provisioning failed: {e}")
            public_url = deployment_endpoint
    else:
        log.warning("Skipping Cloudflare DNS (CLOUDFLARE_API_TOKEN not set)")
        public_url = deployment_endpoint

    result["public_url"] = public_url

    # 3. Stripe billing — optional (requires STRIPE_SECRET_KEY)
    if os.environ.get("STRIPE_SECRET_KEY") and contact_email and contact_name:
        from ops.stripe_billing import setup_client
        try:
            pricing = tier_config.get("pricing", {})
            stripe_result = setup_client(
                email=contact_email,
                name=contact_name,
                client_slug=client_slug,
                tier=tier_config["tier"],
                setup_usd=pricing.get("setup_fee_usd", 5000),
                monthly_usd=pricing.get("monthly_hosting_usd", 1500),
            )
            result["stripe"] = stripe_result
        except Exception as e:
            log.warning(f"Stripe provisioning failed: {e}")
    else:
        log.warning("Skipping Stripe billing (set STRIPE_SECRET_KEY + pass --email/--name to enable)")

    return result


if __name__ == "__main__":
    import argparse
    import json
    import yaml
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--email")
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    result = provision(
        client_slug=args.client,
        tier_config=cfg,
        deployment_endpoint=args.endpoint,
        contact_email=args.email,
        contact_name=args.name,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
