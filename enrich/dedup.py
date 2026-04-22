"""Semantic deduplication using Nomic embeddings.

Unlike string/hash dedup, this catches paraphrases and near-duplicates.
We embed every row, cluster by cosine similarity, keep one representative
per cluster.

This does NOT call gpt-oss (would be overkill + slow). Uses
sentence-transformers with `nomic-ai/nomic-embed-text-v1.5` — same model family
as the docs-ingest-demo.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_THRESHOLD = 0.92   # cosine similarity above this = duplicate


def _row_text(row: dict) -> str:
    """Concatenate all message content for embedding."""
    msgs = row.get("messages", [])
    return " \n ".join(m.get("content", "") for m in msgs)


def _load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _embed(model, texts: Iterable[str]) -> np.ndarray:
    prefixed = [f"search_document: {t}" for t in texts]
    vecs = model.encode(prefixed, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _greedy_cluster(vecs: np.ndarray, threshold: float) -> list[int]:
    """Return cluster ID for each row. Simple single-pass greedy; good enough for typical corpora."""
    n = vecs.shape[0]
    assigned = np.full(n, -1, dtype=np.int64)
    next_cluster = 0
    cluster_reps: list[np.ndarray] = []

    for i in range(n):
        if not cluster_reps:
            assigned[i] = next_cluster
            cluster_reps.append(vecs[i])
            next_cluster += 1
            continue

        # Cosine similarity = dot product (vectors are normalized)
        sims = np.stack(cluster_reps) @ vecs[i]
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            assigned[i] = best
        else:
            assigned[i] = next_cluster
            cluster_reps.append(vecs[i])
            next_cluster += 1

    return assigned.tolist()


def dedup_file(
    in_path: Path,
    out_path: Path,
    dup_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Semantic dedup. Writes `out_path` (kept representatives) + `dup_path` (removed duplicates)."""
    from sentence_transformers import SentenceTransformer

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(in_path)
    if not rows:
        log.warning(f"No rows in {in_path}")
        out_path.write_text("")
        dup_path.write_text("")
        return {"total": 0, "kept": 0, "duplicates": 0}

    # Run on CPU by default — avoids CUDA OOM when gpt-oss-120b is loaded in
    # parallel. Nomic v1.5 (~550MB) encodes ~100 docs/sec on CPU, which is
    # fine for typical client corpora. If you need GPU dedup, unload gpt-oss
    # first and set device="cuda" manually.
    device = "cpu"
    log.info(f"Loading embedding model on {device}: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    log.info(f"Embedding {len(rows)} rows...")
    texts = [_row_text(r) for r in rows]
    vecs = _embed(model, texts)

    log.info(f"Clustering with cosine threshold {threshold}")
    cluster_ids = _greedy_cluster(vecs, threshold)

    # For each cluster, keep the first row (could be fanciest — keep the longest etc)
    seen: set[int] = set()
    kept_rows = []
    dup_rows = []
    for row, cid in zip(rows, cluster_ids):
        if cid not in seen:
            seen.add(cid)
            kept_rows.append(row)
        else:
            dup_rows.append({**row, "_dup_of_cluster": cid})

    with out_path.open("w") as f:
        for row in kept_rows:
            f.write(json.dumps(row) + "\n")
    with dup_path.open("w") as f:
        for row in dup_rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"Kept {len(kept_rows)}, removed {len(dup_rows)} duplicates (into {len(set(cluster_ids))} clusters)")
    return {
        "total": len(rows),
        "kept": len(kept_rows),
        "duplicates": len(dup_rows),
        "clusters": len(set(cluster_ids)),
        "threshold": threshold,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    result = dedup_file(args.input, args.output, args.duplicates, threshold=args.threshold)
    print(json.dumps(result, indent=2))
