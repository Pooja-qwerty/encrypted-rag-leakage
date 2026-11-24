"""Sweep Cloaked approximation factors and derived keys on a fixed workload."""

import argparse
import asyncio
import csv
import hashlib
import hmac
import json
import math
from pathlib import Path

import numpy as np

from cloak.sdk import build_standalone_sdk, load_vector_secret_bytes
from leakage_experiment import prepare, score_saved

DEFAULT_FACTORS = [1.0, 1.5, 2.0, 4.0, 7.2]


def derive_key(root_key, index):
    """Repeat a key family from the configured root without saving raw keys."""
    if len(root_key) != 32 or index < 0:
        raise ValueError("Expected a 32-byte root key and nonnegative key index")
    return hmac.new(root_key, f"cs666-search-sweep-v1:{index}".encode(), hashlib.sha256).digest()


async def validate_metrics(root_key, key_count):
    """A tiny-noise diagnostic checks direction and normalization, not security."""
    from ironcore_alloy import AlloyMetadata
    from cloak.sdk import DEFAULT_TENANT_ID
    from cloak.retrieval import encrypt_index, encrypt_query_vectors, top_k_cloaked

    vectors = np.array([[1., 0.], [.6, .8], [-1., 0.], [0., -1.]])
    metadata = AlloyMetadata.new_simple(DEFAULT_TENANT_ID)
    results = []
    for index in range(key_count):
        sdk = build_standalone_sdk(derive_key(root_key, index), approximation_factor=1e-6)
        encrypted = await encrypt_index(vectors, sdk, metadata)
        query = await encrypt_query_vectors(vectors[0], sdk, metadata)
        for metric in ("cosine", "dot_product", "l2"):
            observed = top_k_cloaked(encrypted, query, 3, metric)
            passed = observed == [0, 1, 3]
            results.append({"key_index": index, "metric": metric,
                            "expected": [0, 1, 3], "observed": observed, "passed": passed})
    if not all(row["passed"] for row in results):
        raise RuntimeError("Known-neighbor metric validation failed")
    return {"diagnostic_factor": 1e-6,
            "note": "Tiny-noise correctness control only; not a secure configuration recommendation.",
            "checks": results}


def summarize(rows):
    summaries = []
    for factor in sorted({r["approximation_factor"] for r in rows}):
        group = [r for r in rows if r["approximation_factor"] == factor]
        entry = {"approximation_factor": factor, "keys": len(group)}
        for metric in ("query_accuracy", "advantage", "mean_overlap_at_3", "query_signal_fraction"):
            values = np.array([r[metric] for r in group])
            entry.update({metric + "_mean": float(values.mean()),
                          metric + "_std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
                          metric + "_min": float(values.min()),
                          metric + "_max": float(values.max())})
        summaries.append(entry)
    return summaries


def sweep(directory, factors=DEFAULT_FACTORS, key_count=10, seed=42, metric="cosine",
          known_per_topic=1, context_count=80, root_key=None):
    if key_count < 1 or not factors or any(not math.isfinite(f) or f <= 0 for f in factors):
        raise ValueError("Expected positive key count and finite positive factors")
    if len(set(factors)) != len(factors):
        raise ValueError("Approximation factors must be unique")
    if metric not in {"cosine", "dot_product", "l2"}:
        raise ValueError("Unsupported metric")
    root_key = load_vector_secret_bytes() if root_key is None else root_key
    validation = asyncio.run(validate_metrics(root_key, key_count))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metric_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    settings = {"factors": list(factors), "keys_per_factor": key_count,
                "seed": seed, "metric": metric, "level": "AP-set",
                "known_per_topic": known_per_topic, "context_count": context_count,
                "key_derivation": "HMAC-SHA256(root, cs666-search-sweep-v1:<key_index>)",
                "variation": "Derived keys and SDK encryption randomness; workload is fixed.",
                "uncertainty": "Standard deviations and ranges across runs, not query confidence intervals."}
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    prepare(directory / "plaintext", seed=seed, known_per_topic=known_per_topic,
            context_count=context_count)
    baseline = score_saved(directory / "plaintext")
    rows = []
    for factor in factors:
        for index in range(key_count):
            destination = directory / f"factor_{factor:g}" / f"key_{index:02d}"
            _, private = prepare(destination, seed=seed, known_per_topic=known_per_topic,
                                 context_count=context_count, backend="cloaked",
                                 approximation_factor=factor, metric=metric,
                                 key_bytes=derive_key(root_key, index))
            private["run"]["key_index"] = index
            (destination / "evaluator_only.json").write_text(json.dumps(private, indent=2) + "\n")
            rows.append({"approximation_factor": factor, "key_index": index,
                         **score_saved(destination)})
        print(f"Completed factor {factor:g}: {key_count} keys", flush=True)
    with (directory / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"settings": settings, "plaintext": baseline, "factors": summarize(rows)}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    parser.add_argument("--keys", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", choices=["cosine", "dot_product", "l2"], default="cosine")
    parser.add_argument("--known-per-topic", type=int, default=1)
    parser.add_argument("--context-count", type=int, default=80)
    args = parser.parse_args()
    result = sweep(args.directory, args.factors, args.keys, args.seed, args.metric,
                   args.known_per_topic, args.context_count)
    print("factor | overlap@3 | topic accuracy | advantage (means across keys)")
    for row in result["factors"]:
        print(f"{row['approximation_factor']:6g} | {row['mean_overlap_at_3_mean']:.3f}     | "
              f"{row['query_accuracy_mean']:.3f}          | {row['advantage_mean']:+.3f}")


if __name__ == "__main__":
    main()
