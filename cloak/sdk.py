"""
Standalone Cloaked AI SDK wiring (Phase 3a).

Secrets: 32-byte vector key via env `CLOAK_VECTOR_SECRET` (hex) or a local `.env` file
(see `.env.example`). Do not commit real secrets.
"""

from __future__ import annotations

import os
import math
import secrets
from pathlib import Path

from ironcore_alloy import (
    RotatableSecret,
    Secret,
    Standalone,
    StandaloneConfiguration,
    StandaloneSecret,
    StandardSecrets,
    VectorSecret,
)

DEFAULT_SECRET_PATH = "rag-chunks"
DEFAULT_DERIVATION_PATH = "cs666-demo"
DEFAULT_TENANT_ID = "tenant-cs666"
APPROXIMATION_FACTOR = float(os.environ.get("CLOAK_APPROXIMATION_FACTOR", "7.2"))


def _parse_secret_bytes(raw: str) -> bytes:
    raw = raw.strip()
    if raw.startswith("0x"):
        raw = raw[2:]
    if len(raw) == 64 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return bytes.fromhex(raw)
    raise ValueError(
        "CLOAK_VECTOR_SECRET must be 32 bytes as 64 hex characters (optional 0x prefix)."
    )


def load_vector_secret_bytes() -> bytes:
    """
    Load the 32-byte standalone vector secret.
    If CLOAK_VECTOR_SECRET is unset, generates one and prints a warning (ephemeral demo only).
    """
    raw = os.environ.get("CLOAK_VECTOR_SECRET", "").strip()
    if raw:
        return _parse_secret_bytes(raw)

    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == "CLOAK_VECTOR_SECRET":
                return _parse_secret_bytes(val.strip().strip('"').strip("'"))

    key = secrets.token_bytes(32)
    print(
        "WARNING: CLOAK_VECTOR_SECRET not set — using a one-off random key for this SDK instance."
    )
    return key


def build_standalone_sdk(
    key_bytes: bytes | None = None, approximation_factor: float | None = None,
) -> Standalone:
    if key_bytes is None:
        key_bytes = load_vector_secret_bytes()
    if len(key_bytes) != 32:
        raise ValueError("Vector secret must be exactly 32 bytes.")
    factor = APPROXIMATION_FACTOR if approximation_factor is None else approximation_factor
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Approximation factor must be finite and positive")

    std = StandaloneSecret(1, Secret(key_bytes))
    vector_secrets = {
        DEFAULT_SECRET_PATH: VectorSecret(
            factor,
            RotatableSecret(std, None),
        ),
    }
    config = StandaloneConfiguration(
        StandardSecrets(None, []),
        {},
        vector_secrets,
    )
    return Standalone(config)
