"""Compare label budgets, attacks, relevance, and session topics on held-out queries."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cloak.sdk import load_vector_secret_bytes
from game3 import load_file
from leakage_experiment import evaluate, prepare
from log_attack import run_attack, run_session_attack
from sweep_experiment import derive_key

ROOT = Path(__file__).resolve().parent


def load_workload(path=ROOT / "data/evaluation_queries.json"):
    workload = json.loads(Path(path).read_text())
    for name, expected in workload["corpus_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Corpus changed: review relevance judgments before running")
    refs = {f"{topic}:{i}" for topic, filename in [("CS", "CS"), ("Bio", "BIO")]
            for i in range(len(load_file(f"data/{filename}.txt")))}
    questions = workload["challenges"]
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    texts = [q["text"] for q in questions]
    if len(set(texts)) != len(texts) or set(texts) & set(workload["context"]):
        raise ValueError("Questions must be distinct and separate from context")
    for q in questions:
        if q["topic"] not in {"CS", "Bio"} or not q["relevant"] or not set(q["relevant"]) <= refs:
            raise ValueError("Invalid topic or relevance references")
    return workload


def select_known(truth, budget, anchor_seed):
    """Select before attack fitting; nested budgets and independent of logs."""
    rng = np.random.default_rng(anchor_seed)
    result = {}
    for topic in ("CS", "Bio"):
        candidates = sorted(cid for cid, label in truth.items() if label == topic)
        if budget < 0 or budget > len(candidates):
            raise ValueError("Invalid known-label budget")
        result.update({str(cid): topic for cid in rng.permutation(candidates)[:budget]})
    return result


def project_view(base, known, level):
    def project(entry):
        result = {"num_results": entry["num_results"]}
        if "query_id" in entry:
            result["query_id"] = entry["query_id"]
        if level != "Volume":
            ids = list(entry["chunk_ids"])
            result["chunk_ids"] = sorted(ids) if level == "AP-set" else ids
        return result
    return {"level": level, "known_labels": dict(known),
            "context": [project(e) for e in base["context"]],
            "challenges": [project(e) for e in base["challenges"]]}


def relevance_scores(private):
    precision, ndcg = [], []
    for row in private["retrieval_utility"].values():
        relevant = set(row["relevant_ids"])
        hits = [cid in relevant for cid in row["backend_topk"]]
        precision.append(sum(hits) / 3)
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(3, len(relevant))))
        ndcg.append(dcg / ideal)
    return {"precision_at_3": float(np.mean(precision)), "ndcg_at_3": float(np.mean(ndcg))}


def make_sessions(query_truth, seed, count_per_topic=20):
    """Synthetic 4:1 majority sessions resampled from held-out retrieval traces."""
    rng = np.random.default_rng(seed)
    sessions, labels = [], {}
    for majority in ("CS", "Bio"):
        own = sorted(q for q, t in query_truth.items() if t == majority)
        other = sorted(q for q, t in query_truth.items() if t != majority)
        for _ in range(count_per_topic):
            queries = list(rng.choice(own, 4, replace=False)) + [str(rng.choice(other))]
            rng.shuffle(queries)
            sid = "s_" + rng.bytes(12).hex()
            sessions.append({"session_id": sid, "query_ids": [str(q) for q in queries]})
            labels[sid] = majority
    rng.shuffle(sessions)
    return {"sessions": sessions}, labels


def session_scores(predictions, truth):
    if set(predictions) != set(truth) or not truth:
        raise ValueError("Expected one prediction per session")
    rates = {topic: np.mean([predictions[s]["topic"] == "Bio" for s in truth if truth[s] == topic])
             for topic in ("CS", "Bio")}
    return {"session_accuracy": float(np.mean([predictions[s]["topic"] == truth[s] for s in truth])),
            "session_advantage": float(rates["Bio"] - rates["CS"]),
            "session_signal_fraction": float(np.mean([p["has_signal"] for p in predictions.values()]))}


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["backend"], row["approximation_factor"], row["budget"], row["level"], row["method"])
        groups.setdefault(key, []).append(row)
    result = []
    metrics = ["query_accuracy", "advantage", "query_signal_fraction", "session_accuracy",
               "session_advantage", "precision_at_3", "ndcg_at_3", "mean_overlap_at_3"]
    for key, group in groups.items():
        entry = dict(zip(["backend", "approximation_factor", "budget", "level", "method"], key))
        entry["runs"] = len(group)
        for metric in metrics:
            values = np.array([row[metric] for row in group])
            entry[metric + "_mean"] = float(values.mean())
            entry[metric + "_std"] = float(values.std(ddof=1)) if len(group) > 1 else 0.
            entry[metric + "_min"] = float(values.min())
            entry[metric + "_max"] = float(values.max())
        result.append(entry)
    return result


def paired_gains(rows):
    """Clustering minus direct voting for identical logs and disclosed labels."""
    pairs = {}
    for row in rows:
        key = (row["backend"], row["approximation_factor"], row["budget"], row["level"],
               row["key_index"], row["anchor_seed"])
        pairs.setdefault(key, {})[row["method"]] = row
    groups = {}
    for key, methods in pairs.items():
        if set(methods) != {"known_only", "clustering"}:
            raise ValueError("Both methods are required for paired comparison")
        gain = {metric: methods["clustering"][metric] - methods["known_only"][metric]
                for metric in ("query_accuracy", "session_accuracy")}
        groups.setdefault(key[:4], []).append(gain)
    result = []
    for key, gains in groups.items():
        entry = dict(zip(["backend", "approximation_factor", "budget", "level"], key))
        entry["pairs"] = len(gains)
        for metric in ("query_accuracy", "session_accuracy"):
            values = np.array([g[metric] for g in gains])
            entry[metric + "_gain_mean"] = float(values.mean())
            entry[metric + "_gain_std"] = float(values.std(ddof=1)) if len(gains) > 1 else 0.
        result.append(entry)
    return result


def study(directory, factors=(1.5, 7.2), keys=5, label_seeds=10, budgets=(0, 1, 2, 4),
          seed=42, sessions_per_topic=20, root_key=None):
    if (keys < 1 or label_seeds < 1 or sessions_per_topic < 1 or not factors or not budgets
            or any(not math.isfinite(f) or f <= 0 for f in factors)):
        raise ValueError("Expected positive run counts and factors")
    workload = load_workload()
    if any(b < 0 or b > 12 for b in budgets):
        raise ValueError("Budgets must be between 0 and 12")
    root_key = load_vector_secret_bytes() if root_key is None else root_key
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {"workload": workload["name"], "seed": seed, "context_searches": 96,
                "context_types": len(workload["context"]), "test_queries": len(workload["challenges"]),
                "workload_sha256": hashlib.sha256((ROOT / "data/evaluation_queries.json").read_bytes()).hexdigest(),
                "factors": list(factors), "keys": keys, "label_seeds": label_seeds,
                "budgets_per_topic": list(budgets), "metric": "cosine",
                "sessions_per_topic": sessions_per_topic, "session_majority": "4 of 5 queries",
                "limits": "Synthetic author-defined qrels pending team review. Same 24 chunks. "
                          "Sessions reuse held-out traces. Spread across key/label draws is not a "
                          "confidence interval over independent real-user queries."}
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    configs = [("plaintext", None, None)] + [("cloaked", f, k) for f in factors for k in range(keys)]
    rows = []
    for backend, factor, key_index in configs:
        name = "plaintext" if backend == "plaintext" else f"cloaked_{factor:g}_key_{key_index:02d}"
        destination = directory / name
        base, private = prepare(destination, level="AP-ordered", known_per_topic=0, seed=seed,
                                context_count=96, backend=backend, approximation_factor=factor,
                                key_bytes=None if key_index is None else derive_key(root_key, key_index),
                                workload=workload)
        private["run"]["key_index"] = key_index
        sessions, session_truth = make_sessions(private["query_topics"], seed + 1000, sessions_per_topic)
        private["session_topics"] = session_truth
        (destination / "evaluator_only.json").write_text(json.dumps(private, indent=2) + "\n")
        (destination / "session_view.json").write_text(json.dumps(sessions, indent=2) + "\n")
        quality = relevance_scores(private)
        for anchor_seed in range(label_seeds):
            for budget in budgets:
                known = select_known(private["chunk_topics"], budget, anchor_seed)
                for level in ("Volume", "AP-set", "AP-ordered"):
                    view = project_view(base, known, level)
                    for method in ("known_only", "clustering"):
                        attack = run_attack(view, method)
                        session_predictions = run_session_attack(view, sessions, attack["chunk_topics"])
                        report = {**evaluate(view, private, attack), **quality,
                                  **session_scores(session_predictions, session_truth)}
                        rows.append({"backend": backend, "approximation_factor": factor,
                                     "key_index": key_index, "anchor_seed": anchor_seed,
                                     "budget": budget, "level": level, "method": method, **report})
                        if anchor_seed == 0 and budget == 1 and level == "AP-set":
                            example = destination / f"example_{method}"
                            example.mkdir(exist_ok=True)
                            for filename, payload in [("attacker_view", view), ("predictions", attack),
                                                      ("session_predictions", session_predictions)]:
                                (example / f"{filename}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Completed {name}: retrieval shared across budgets, methods, and log views", flush=True)
    with (directory / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"settings": settings, "summaries": aggregate(rows), "clustering_gains": paired_gains(rows)}
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--factors", nargs="+", type=float, default=[1.5, 7.2])
    parser.add_argument("--keys", type=int, default=5)
    parser.add_argument("--label-seeds", type=int, default=10)
    parser.add_argument("--budgets", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = study(args.directory, args.factors, args.keys, args.label_seeds, args.budgets, args.seed)
    print("backend factor budget method: query accuracy / session accuracy / precision@3 / nDCG@3")
    for row in result["summaries"]:
        if row["level"] == "AP-set":
            print(f"{row['backend']} {row['approximation_factor']} {row['budget']} {row['method']}: "
                  f"{row['query_accuracy_mean']:.3f} / {row['session_accuracy_mean']:.3f} / "
                  f"{row['precision_at_3_mean']:.3f} / {row['ndcg_at_3_mean']:.3f}")


if __name__ == "__main__":
    main()
