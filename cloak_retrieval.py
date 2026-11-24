#!/usr/bin/env python3
"""
Phase 3b: Plain vs Cloaked AI top-k on the private CS+BIO index (same data as game3).

  python cloak_retrieval.py
  python cloak_retrieval.py --k 5 --csv results/cloak_retrieval.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

import numpy as np
from ironcore_alloy import AlloyMetadata

from cloak.retrieval import (
    encrypt_index,
    encrypt_query_vectors,
    overlap_at_k,
    top_k_cloaked,
    top_k_plain,
)
from cloak.sdk import DEFAULT_TENANT_ID, build_standalone_sdk
from game3 import build_embedder, load_file, setup_db


CS_QUERIES = [
    "How does indexing speed up database search?",
    "What is a B-tree used for?",
    "Explain hash tables in computer science.",
    "How do graph traversal algorithms work?",
]

BIO_QUERIES = [
    "How does genome sequencing work?",
    "What is sequence alignment in bioinformatics?",
    "Explain phylogenetic trees and evolution.",
    "How are protein structures predicted?",
]


def _load_private_db():
    cs = load_file("data/CS.txt")
    bio = load_file("data/BIO.txt")
    raw = {
        "texts": cs + bio,
        "subjects": ["CS"] * len(cs) + ["Bio"] * len(bio),
        "chunk_sizes": [len(t.encode("utf-8")) for t in cs + bio],
    }
    all_texts = raw["texts"] + CS_QUERIES + BIO_QUERIES
    embed_fn = build_embedder(all_texts)
    db = setup_db(raw, embed_fn)
    return db, embed_fn


async def run_comparison(k: int, metric: str = "cosine") -> list[dict]:
    db, embed_fn = _load_private_db()
    vectors = np.asarray(db["vectors"], dtype=np.float64)
    sdk = build_standalone_sdk()
    meta = AlloyMetadata.new_simple(DEFAULT_TENANT_ID)

    print(f"Encrypting {len(vectors)} index vectors (dim={vectors.shape[1]}, metric={metric})...")
    index_enc = await encrypt_index(vectors, sdk=sdk, metadata=meta)

    rows = []
    for topic, queries in (("CS", CS_QUERIES), ("Bio", BIO_QUERIES)):
        for q_text in queries:
            q_vec = embed_fn([q_text])[0]
            plain_ids = top_k_plain(vectors, q_vec, k)
            q_enc = await encrypt_query_vectors(q_vec, sdk=sdk, metadata=meta, query_id="q")
            cloak_ids = top_k_cloaked(index_enc, q_enc, k, metric=metric)
            ov = overlap_at_k(plain_ids, cloak_ids, k)
            rows.append(
                {
                    "topic": topic,
                    "query": q_text,
                    "k": k,
                    "metric": metric,
                    "plain_topk": plain_ids,
                    "cloaked_topk": cloak_ids,
                    "overlap_at_k": ov,
                }
            )
    return rows


def _print_report(rows: list[dict], k: int) -> None:
    print(f"\n{'topic':5s} | {'overlap@'+str(k):10s} | query (truncated)")
    print("-" * 72)
    for r in rows:
        q_short = r["query"][:48] + ("…" if len(r["query"]) > 48 else "")
        print(f"{r['topic']:5s} | {r['overlap_at_k']:10.2f} | {q_short}")
        print(f"       plain:   {r['plain_topk']}")
        print(f"       cloaked: {r['cloaked_topk']}")

    mean_ov = float(np.mean([r["overlap_at_k"] for r in rows]))
    print(f"\nMean overlap@{k} over {len(rows)} queries: {mean_ov:.3f}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic", "query", "k", "metric", "plain_topk", "cloaked_topk", "overlap_at_k"])
        for r in rows:
            w.writerow(
                [
                    r["topic"],
                    r["query"],
                    r["k"],
                    r["metric"],
                    " ".join(map(str, r["plain_topk"])),
                    " ".join(map(str, r["cloaked_topk"])),
                    f"{r['overlap_at_k']:.4f}",
                ]
            )
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", type=int, default=3, help="top-k results")
    ap.add_argument("--csv", type=Path, default=None, help="optional output CSV path")
    ap.add_argument("--metric", choices=["cosine", "dot_product", "l2"], default="cosine")
    args = ap.parse_args()

    rows = asyncio.run(run_comparison(args.k, args.metric))
    _print_report(rows, args.k)
    if args.csv:
        _write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
