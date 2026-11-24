#!/usr/bin/env python3
"""
Phase 3a smoke test: IronCore Alloy Standalone + Cloaked AI vector encrypt/decrypt.

Run from repo root with venv active:
  python cloak_spike.py
"""

from __future__ import annotations

import argparse
import asyncio
import math

from ironcore_alloy import AlloyMetadata, PlaintextVector

from cloak.sdk import (
    DEFAULT_DERIVATION_PATH,
    DEFAULT_SECRET_PATH,
    DEFAULT_TENANT_ID,
    build_standalone_sdk,
)


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


async def run_smoke() -> int:
    sdk = build_standalone_sdk()
    vec = sdk.vector()
    meta = AlloyMetadata.new_simple(DEFAULT_TENANT_ID)

    plain = PlaintextVector(
        plaintext_vector=_normalize([0.6, 0.0, 0.8, 0.1, 0.2]),
        secret_path=DEFAULT_SECRET_PATH,
        derivation_path=DEFAULT_DERIVATION_PATH,
    )
    enc = await vec.encrypt(plain, meta)
    dec = await vec.decrypt(enc, meta)
    err = max(abs(a - b) for a, b in zip(plain.plaintext_vector, dec.plaintext_vector))

    query = PlaintextVector(
        plaintext_vector=_normalize([0.55, 0.1, 0.82, 0.0, 0.25]),
        secret_path=DEFAULT_SECRET_PATH,
        derivation_path=DEFAULT_DERIVATION_PATH,
    )
    q_pack = await vec.generate_query_vectors({"demo-query": query}, meta)

    print("Cloaked AI (Standalone) smoke test")
    print(f"  tenant:        {DEFAULT_TENANT_ID}")
    print(f"  secret_path:   {DEFAULT_SECRET_PATH}")
    print(f"  embedding dim: {len(enc.encrypted_vector)}")
    print(f"  decrypt L_inf: {err:.2e} (should be tiny)")
    print(f"  query vectors: {len(q_pack['demo-query'])} variant(s) for search")
    print("OK — ironcore-alloy is wired up.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(asyncio.run(run_smoke()))


if __name__ == "__main__":
    main()