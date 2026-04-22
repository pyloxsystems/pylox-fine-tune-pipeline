"""Minimal OpenAI-compatible client for the local gpt-oss-120b TRT-LLM server.

Uses httpx directly (not the openai SDK) so we can:
  - Bound timeouts tightly per call
  - Retry on timeout/5xx with exponential backoff
  - Batch concurrent requests under a bounded semaphore
  - Inspect `reasoning` field when gpt-oss returns empty `content` (known quirk)

Endpoint: http://localhost:8002/v1 by default.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://localhost:8002/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT = 120.0
DEFAULT_CONCURRENCY = 4


@dataclass
class ChatMessage:
    role: str
    content: str


class GPTOSSClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        concurrency: int = DEFAULT_CONCURRENCY,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._sem = asyncio.Semaphore(concurrency)

    async def health(self) -> bool:
        async with httpx.AsyncClient(timeout=5) as client:
            try:
                r = await client.get(f"{self.endpoint}/models")
                return r.status_code == 200
            except httpx.HTTPError:
                return False

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
        retries: int = 2,
    ) -> str:
        """Single chat call. Returns string. Unwraps `reasoning` field if content is empty."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.__dict__ if isinstance(m, ChatMessage) else m for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        async with self._sem:
            last_err: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        r = await client.post(f"{self.endpoint}/chat/completions", json=payload)
                        r.raise_for_status()
                        data = r.json()
                        msg = data["choices"][0]["message"]
                        content = msg.get("content") or ""
                        if not content.strip():
                            # gpt-oss-120b quirk: sometimes output lands in `reasoning` field
                            content = msg.get("reasoning", "") or ""
                        return content.strip()
                except (httpx.HTTPError, KeyError, IndexError) as e:
                    last_err = e
                    if attempt < retries:
                        backoff = 2 ** attempt
                        log.warning(f"Chat call failed (attempt {attempt+1}): {e}. Retry in {backoff}s.")
                        await asyncio.sleep(backoff)
                    else:
                        raise RuntimeError(f"Chat call failed after {retries+1} attempts: {last_err}") from e
        return ""  # unreachable

    async def chat_batch(
        self,
        batch: list[list[ChatMessage]],
        **kwargs,
    ) -> list[str]:
        """Dispatch `batch` chat calls concurrently (bounded by self._sem)."""
        tasks = [self.chat(msgs, **kwargs) for msgs in batch]
        return await asyncio.gather(*tasks)


def sync_chat(client: GPTOSSClient, messages, **kwargs) -> str:
    """Convenience sync wrapper for non-async callers."""
    return asyncio.run(client.chat(messages, **kwargs))


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Ping the local gpt-oss-120b server.")
    parser.add_argument("--prompt", default="Say hello in 5 words.")
    args = parser.parse_args()

    client = GPTOSSClient()
    async def _run():
        if not await client.health():
            print(f"ERROR: server at {client.endpoint} is not healthy", file=sys.stderr)
            sys.exit(1)
        t = time.time()
        resp = await client.chat([ChatMessage("user", args.prompt)], max_tokens=64)
        print(f"({(time.time()-t)*1000:.0f}ms) {resp}")

    asyncio.run(_run())
