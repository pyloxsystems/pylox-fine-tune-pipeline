"""Stripe subscription setup for client engagements.

Setup fees: one-time invoice. Monthly hosting: recurring subscription.
Designed so you can run `setup_client(...)` once at contract sign and never
touch Stripe again — webhooks handle the rest.

Requires env var: STRIPE_SECRET_KEY (starts with sk_live_ or sk_test_)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


def _client():
    import stripe
    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        raise RuntimeError("Set STRIPE_SECRET_KEY env var (dashboard.stripe.com/apikeys)")
    stripe.api_key = key
    return stripe


def get_or_create_customer(email: str, name: str, client_slug: str) -> str:
    """Return Stripe customer ID. Idempotent — searches by metadata first."""
    stripe = _client()
    existing = stripe.Customer.search(query=f'metadata["client_slug"]:"{client_slug}"')
    if existing.data:
        log.info(f"Found existing customer for {client_slug}: {existing.data[0].id}")
        return existing.data[0].id

    customer = stripe.Customer.create(
        email=email,
        name=name,
        metadata={"client_slug": client_slug},
    )
    log.info(f"Created Stripe customer: {customer.id}")
    return customer.id


def get_or_create_monthly_product(tier: str, monthly_usd: int) -> tuple[str, str]:
    """Return (product_id, price_id) for the tier's monthly subscription. Idempotent."""
    stripe = _client()
    product_name = f"Pylox Systems — {tier.upper()} tier hosting"

    existing = stripe.Product.search(query=f'metadata["pylox_tier"]:"{tier}"')
    if existing.data:
        product = existing.data[0]
    else:
        product = stripe.Product.create(
            name=product_name,
            metadata={"pylox_tier": tier},
        )
        log.info(f"Created product: {product.id}")

    # Find existing price
    for price in stripe.Price.list(product=product.id, active=True).data:
        if price.unit_amount == monthly_usd * 100 and price.recurring:
            return product.id, price.id

    price = stripe.Price.create(
        product=product.id,
        unit_amount=monthly_usd * 100,
        currency="usd",
        recurring={"interval": "month"},
        metadata={"pylox_tier": tier},
    )
    log.info(f"Created price: {price.id} (${monthly_usd}/mo)")
    return product.id, price.id


def create_setup_invoice(customer_id: str, setup_usd: int, client_slug: str, tier: str) -> str:
    """One-time setup fee invoice. Returns invoice ID."""
    stripe = _client()
    item = stripe.InvoiceItem.create(
        customer=customer_id,
        amount=setup_usd * 100,
        currency="usd",
        description=f"Pylox Systems — {tier.upper()} tier setup fee",
    )
    invoice = stripe.Invoice.create(
        customer=customer_id,
        auto_advance=True,    # auto-finalize + attempt charge
        metadata={"client_slug": client_slug, "invoice_kind": "setup"},
        collection_method="send_invoice",
        days_until_due=14,
    )
    finalized = stripe.Invoice.finalize_invoice(invoice.id)
    log.info(f"Setup invoice {finalized.id} — ${setup_usd} due in 14 days")
    return finalized.id


def start_monthly_subscription(customer_id: str, price_id: str, client_slug: str) -> str:
    """Start monthly hosting subscription. Returns subscription ID."""
    stripe = _client()
    sub = stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": price_id}],
        metadata={"client_slug": client_slug},
        collection_method="charge_automatically",
        payment_behavior="default_incomplete",
        expand=["latest_invoice.payment_intent"],
    )
    log.info(f"Monthly subscription {sub.id} — status={sub.status}")
    return sub.id


def setup_client(
    email: str,
    name: str,
    client_slug: str,
    tier: str,
    setup_usd: int,
    monthly_usd: int,
) -> dict:
    """Do the whole Stripe setup in one shot at contract sign."""
    customer_id = get_or_create_customer(email, name, client_slug)
    _, price_id = get_or_create_monthly_product(tier, monthly_usd)
    invoice_id = create_setup_invoice(customer_id, setup_usd, client_slug, tier)
    subscription_id = start_monthly_subscription(customer_id, price_id, client_slug)
    return {
        "customer_id": customer_id,
        "setup_invoice_id": invoice_id,
        "subscription_id": subscription_id,
        "tier": tier,
        "setup_usd": setup_usd,
        "monthly_usd": monthly_usd,
    }


def cancel_client(client_slug: str) -> int:
    """Cancel all active subscriptions for a client. Returns count cancelled."""
    stripe = _client()
    subs = stripe.Subscription.search(
        query=f'metadata["client_slug"]:"{client_slug}" AND status:"active"'
    )
    count = 0
    for sub in subs.data:
        stripe.Subscription.delete(sub.id)
        count += 1
    log.info(f"Cancelled {count} subscriptions for {client_slug}")
    return count


if __name__ == "__main__":
    import argparse
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup")
    s.add_argument("--email", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--client", required=True)
    s.add_argument("--tier", required=True)
    s.add_argument("--setup-usd", type=int, required=True)
    s.add_argument("--monthly-usd", type=int, required=True)

    c = sub.add_parser("cancel")
    c.add_argument("client")

    args = parser.parse_args()
    if args.cmd == "setup":
        print(json.dumps(setup_client(
            email=args.email, name=args.name, client_slug=args.client,
            tier=args.tier, setup_usd=args.setup_usd, monthly_usd=args.monthly_usd,
        ), indent=2))
    elif args.cmd == "cancel":
        print(f"cancelled {cancel_client(args.client)}")
