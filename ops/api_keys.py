"""Per-client API key management.

Generates cryptographically strong keys, stores only a SHA-256 hash on disk.
Middleware (ops/api_middleware.py, separate) validates inbound requests.

Storage: SQLite at `ops/keys.db`. Schema created on first use.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "keys.db"
KEY_PREFIX = "psk_"          # "pylox systems key"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            client_slug TEXT NOT NULL,
            key_hash TEXT PRIMARY KEY,
            key_preview TEXT NOT NULL,   -- first 12 chars for debugging (non-sensitive)
            created_at REAL NOT NULL,
            revoked_at REAL,
            metadata TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_api_keys_client ON api_keys(client_slug)
    """)
    return conn


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_key(client_slug: str, metadata: str = "") -> str:
    """Generate + store a new API key. Return the RAW key once (only time it's visible)."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    h = _hash(raw)
    preview = raw[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO api_keys (client_slug, key_hash, key_preview, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (client_slug, h, preview, time.time(), metadata),
        )
    log.info(f"Generated key for {client_slug}: {preview}…")
    return raw


def verify_key(raw: str) -> dict | None:
    """Look up by hash. Returns {client_slug, key_preview, created_at} or None if not found/revoked."""
    with _conn() as c:
        row = c.execute(
            "SELECT client_slug, key_preview, created_at, revoked_at FROM api_keys WHERE key_hash = ?",
            (_hash(raw),),
        ).fetchone()
    if not row:
        return None
    client_slug, preview, created, revoked = row
    if revoked:
        return None
    return {"client_slug": client_slug, "key_preview": preview, "created_at": created}


def revoke_key(raw: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE key_hash = ? AND revoked_at IS NULL",
            (time.time(), _hash(raw)),
        )
        return cur.rowcount > 0


def revoke_all_for_client(client_slug: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE client_slug = ? AND revoked_at IS NULL",
            (time.time(), client_slug),
        )
        return cur.rowcount


def list_keys(client_slug: str | None = None) -> list[dict]:
    q = "SELECT client_slug, key_preview, created_at, revoked_at, metadata FROM api_keys"
    args = ()
    if client_slug:
        q += " WHERE client_slug = ?"
        args = (client_slug,)
    q += " ORDER BY created_at DESC"
    with _conn() as c:
        return [
            {
                "client_slug": r[0],
                "key_preview": r[1],
                "created_at": r[2],
                "revoked_at": r[3],
                "metadata": r[4],
            }
            for r in c.execute(q, args).fetchall()
        ]


if __name__ == "__main__":
    import argparse
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("client")

    rev = sub.add_parser("revoke")
    rev.add_argument("key")

    rev_all = sub.add_parser("revoke-all")
    rev_all.add_argument("client")

    lst = sub.add_parser("list")
    lst.add_argument("--client", default=None)

    args = parser.parse_args()
    if args.cmd == "generate":
        print(generate_key(args.client))
    elif args.cmd == "revoke":
        print("revoked" if revoke_key(args.key) else "not found / already revoked")
    elif args.cmd == "revoke-all":
        print(f"revoked {revoke_all_for_client(args.client)} keys")
    elif args.cmd == "list":
        print(json.dumps(list_keys(args.client), indent=2, default=str))
