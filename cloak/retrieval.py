"""Encrypt embeddings and run top-k on Cloaked vectors (Phase 3b)."""

from __future__ import annotations

import numpy as np
from ironcore_alloy import AlloyMetadata, EncryptedVector, PlaintextVector, Standalone
from ciphertext_math import score_vectors

from cloak.sdk import (
    DEFAULT_DERIVATION_PATH,
    DEFAULT_SECRET_PATH,
    DEFAULT_TENANT_ID,
    build_standalone_sdk,
)


def _row_to_plain(row: np.ndarray) -> PlaintextVector:
    v = np.asarray(row, dtype=np.float64)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return PlaintextVector(
        plaintext_vector=v.astype(np.float32).tolist(),
        secret_path=DEFAULT_SECRET_PATH,
        derivation_path=DEFAULT_DERIVATION_PATH,
    )


def _encrypted_to_matrix(encrypted: list[EncryptedVector]) -> np.ndarray:
    return np.asarray([ev.encrypted_vector for ev in encrypted], dtype=np.float64)


async def encrypt_index(
    vectors: np.ndarray,
    sdk: Standalone | None = None,
    metadata: AlloyMetadata | None = None,
) -> list[EncryptedVector]:
    sdk = sdk or build_standalone_sdk()
    meta = metadata or AlloyMetadata.new_simple(DEFAULT_TENANT_ID)
    vec = sdk.vector()
    batch = {str(i): _row_to_plain(vectors[i]) for i in range(len(vectors))}
    result = await vec.encrypt_batch(batch, meta)
    if result.failures:
        raise RuntimeError(f"encrypt_batch failures: {result.failures}")
    return [result.successes[str(i)] for i in range(len(vectors))]


async def encrypt_query_vectors(
    query_vec: np.ndarray,
    sdk: Standalone | None = None,
    metadata: AlloyMetadata | None = None,
    query_id: str = "q",
) -> list[EncryptedVector]:
    sdk = sdk or build_standalone_sdk()
    meta = metadata or AlloyMetadata.new_simple(DEFAULT_TENANT_ID)
    vec = sdk.vector()
    pt = _row_to_plain(query_vec)
    pack = await vec.generate_query_vectors({query_id: pt}, meta)
    return list(pack[query_id])


def top_k_plain(vectors: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    q = np.asarray(query, dtype=float)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    scores = vectors @ q
    kk = min(k, len(scores))
    idx = np.argpartition(-scores, kk - 1)[:kk]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx]


def top_k_cloaked(
    index_encrypted: list[EncryptedVector],
    query_encrypted: list[EncryptedVector],
    k: int,
    metric: str = "cosine",
) -> list[int]:
    """Rank one active key's vectors. Cosine matches IronCore's examples.

    Ciphertext norms vary, so cosine and raw dot product are not equivalent.
    Search rotated keys in their matching index segments, not by summing scores.
    """
    if len(query_encrypted) != 1:
        raise ValueError("Search one active vector key at a time")
    if k <= 0 or not index_encrypted:
        raise ValueError("Expected a nonempty index and positive k")
    index_mat = _encrypted_to_matrix(index_encrypted)
    q = np.asarray(query_encrypted[0].encrypted_vector, dtype=np.float64)
    scores = score_vectors(index_mat, q, metric)
    kk = min(k, len(scores))
    idx = np.argpartition(-scores, kk - 1)[:kk]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx]


def overlap_at_k(plain: list[int], cloaked: list[int], k: int) -> float:
    a, b = plain[:k], cloaked[:k]
    if not a:
        return 0.0
    return len(set(a) & set(b)) / float(k)
