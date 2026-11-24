"""Prepare, score, or compare plaintext and Cloaked log-only experiments."""

import argparse
import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from game3 import load_file, top_n
from log_attack import LEVELS

# Context queries build the co-retrieval graph; challenge wording is separate.
CONTEXT = [
    "B-tree indexing and range lookup in databases",
    "Hash tables and probabilistic membership filters",
    "Breadth-first and depth-first graph traversal",
    "Query optimization with hash joins and index scans",
    "DNA sequencing reads and genome assembly",
    "Protein sequence alignment and structure comparison",
    "Phylogenetic trees and evolutionary relationships",
    "RNA sequencing gene expression and normalization",
]
CHALLENGES = [
    ("CS", "How does indexing speed up database search?"),
    ("CS", "What is a B-tree used for?"),
    ("CS", "Explain hash tables in computer science."),
    ("CS", "How do graph traversal algorithms work?"),
    ("Bio", "How does genome sequencing work?"),
    ("Bio", "What is sequence alignment in bioinformatics?"),
    ("Bio", "Explain phylogenetic trees and evolution."),
    ("Bio", "How are protein structures predicted?"),
]


async def retrieve(vectors, queries, backend, approximation_factor, metric="cosine", key_bytes=None):
    """Return actual backend IDs and plaintext reference IDs for utility scoring."""
    plain = [top_n({"vectors": vectors}, q, n_results=3) for q in queries]
    if backend == "plaintext":
        return plain, plain, {"metric": "dot_product"}
    if backend != "cloaked":
        raise ValueError("Unknown retrieval backend")
    from ironcore_alloy import AlloyMetadata
    from cloak.sdk import APPROXIMATION_FACTOR, DEFAULT_TENANT_ID, build_standalone_sdk
    from cloak.retrieval import encrypt_index, encrypt_query_vectors, top_k_cloaked

    factor = APPROXIMATION_FACTOR if approximation_factor is None else approximation_factor
    sdk_options = {"approximation_factor": factor}
    if key_bytes is not None:
        sdk_options["key_bytes"] = key_bytes
    sdk = build_standalone_sdk(**sdk_options)
    metadata = AlloyMetadata.new_simple(DEFAULT_TENANT_ID)
    encrypted_index = await encrypt_index(vectors, sdk, metadata)
    results = []
    for i, query in enumerate(queries):
        encrypted_query = await encrypt_query_vectors(query, sdk, metadata, query_id=str(i))
        if len(encrypted_query) != 1:
            raise ValueError("This standalone experiment expects one active vector key")
        results.append(top_k_cloaked(encrypted_index, encrypted_query, 3, metric=metric))
    return results, plain, {"metric": metric, "approximation_factor": factor,
                            "ironcore_alloy_version": version("ironcore-alloy")}


def prepare(output_dir, level="AP-set", known_per_topic=1, seed=42, context_count=80,
            backend="plaintext", approximation_factor=None, metric="cosine", key_bytes=None,
            workload=None):
    return asyncio.run(prepare_async(output_dir, level, known_per_topic, seed, context_count,
                                     backend, approximation_factor, metric, key_bytes, workload))


async def prepare_async(output_dir, level, known_per_topic, seed, context_count,
                        backend, approximation_factor, metric="cosine", key_bytes=None,
                        workload=None):
    if level not in LEVELS or known_per_topic < 0 or context_count < 0:
        raise ValueError("Invalid leakage level, known-label budget, or context count")
    # Independent streams keep records and queries identical across budget sweeps.
    shuffle_rng, id_rng, anchor_rng, query_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    cs, bio = load_file("data/CS.txt"), load_file("data/BIO.txt")
    texts = cs + bio
    subjects = ["CS"] * len(cs) + ["Bio"] * len(bio)
    source_refs = [f"CS:{i}" for i in range(len(cs))] + [f"Bio:{i}" for i in range(len(bio))]
    if known_per_topic > min(len(cs), len(bio)):
        raise ValueError("Known-label budget exceeds the available chunks per topic")
    order = shuffle_rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    subjects = [subjects[i] for i in order]
    source_refs = [source_refs[i] for i in order]
    chunk_ids = ["c_" + id_rng.bytes(12).hex() for _ in texts]
    # Fit only on indexed documents; no query fitting or model download.
    embedder = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vectors = embedder.fit_transform(texts).toarray()
    truth = dict(zip(chunk_ids, subjects))
    ref_to_id = dict(zip(source_refs, chunk_ids))
    known = {}
    for topic in ("CS", "Bio"):
        candidates = [cid for cid in chunk_ids if truth[cid] == topic]
        # Selection happens before clustering and is independent of logs.
        picks = anchor_rng.permutation(candidates)[:known_per_topic]
        known.update({str(cid): topic for cid in picks})

    context_pool = CONTEXT if workload is None else workload["context"]
    challenge_pool = ([{"topic": t, "text": q} for t, q in CHALLENGES]
                      if workload is None else workload["challenges"])
    context_texts = [context_pool[i] for i in query_rng.integers(len(context_pool), size=context_count)]
    challenge_order = query_rng.permutation(len(challenge_pool))
    challenge_texts = [challenge_pool[i]["text"] for i in challenge_order]
    query_vectors = embedder.transform(context_texts + challenge_texts).toarray()
    retrieved, plain, backend_settings = await retrieve(
        vectors, query_vectors, backend, approximation_factor, metric, key_bytes)

    def log_query(indices):
        entry = {"num_results": len(indices)}
        if level != "Volume":
            ids = [chunk_ids[i] for i in indices]
            entry["chunk_ids"] = sorted(ids) if level == "AP-set" else ids
        return entry

    context = [log_query(ids) for ids in retrieved[:context_count]]
    challenges, query_truth, utility = [], {}, {}
    for offset, i in enumerate(challenge_order, start=context_count):
        topic = challenge_pool[i]["topic"]
        qid = "q_" + id_rng.bytes(12).hex()
        challenges.append({"query_id": qid, **log_query(retrieved[offset])})
        query_truth[qid] = topic
        utility[qid] = {
            "plain_topk": [chunk_ids[j] for j in plain[offset]],
            "backend_topk": [chunk_ids[j] for j in retrieved[offset]],
            "overlap_at_3": len(set(plain[offset]) & set(retrieved[offset])) / 3,
        }
        if workload is not None:
            utility[qid]["relevant_ids"] = [ref_to_id[ref] for ref in challenge_pool[i]["relevant"]]
            utility[qid]["source_query_id"] = challenge_pool[i]["id"]
    view = {"level": level, "context": context,
            "challenges": challenges, "known_labels": known}
    private = {"chunk_topics": truth, "query_topics": query_truth, "retrieval_utility": utility,
               "run": {"backend": backend, "embedder": "TF-IDF", "seed": seed,
                       "numpy_version": version("numpy"),
                       "sklearn_version": version("scikit-learn"), **backend_settings,
                       "known_per_topic": known_per_topic, "context_count": context_count,
                       "context_pool_size": len(context_pool), "k": 3}}
    if workload is not None:
        private["run"]["workload"] = workload["name"]
        private["run"]["relevance_judgments"] = "Synthetic author-defined binary labels; team review pending"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attacker_view.json").write_text(json.dumps(view, indent=2) + "\n")
    (output_dir / "evaluator_only.json").write_text(json.dumps(private, indent=2) + "\n")
    return view, private


def evaluate(view, private, result):
    truth = private["query_topics"]
    predictions = result["predictions"]
    if not truth or set(predictions) != set(truth):
        raise ValueError("Predictions must match all evaluation queries")
    if any(p["topic"] not in {"CS", "Bio"} for p in predictions.values()):
        raise ValueError("Unknown predicted topic")
    rates = {}
    for topic in ("CS", "Bio"):
        qids = [qid for qid, label in truth.items() if label == topic]
        if not qids:
            raise ValueError("Both query topics are required")
        rates[topic] = sum(predictions[q]["topic"] == "Bio" for q in qids) / len(qids)
    # Exclude disclosed anchors from chunk inference scores.
    unknown = set(private["chunk_topics"]) - set(view["known_labels"])
    inferred = unknown & set(result["chunk_topics"])
    return {
        "backend": private["run"]["backend"], "level": view["level"],
        "known_labels": len(view["known_labels"]), "queries": len(truth),
        "query_accuracy": sum(predictions[q]["topic"] == truth[q] for q in truth) / len(truth),
        "majority_baseline": max(list(truth.values()).count(t) for t in ("CS", "Bio")) / len(truth),
        "advantage": rates["Bio"] - rates["CS"],
        "query_signal_fraction": sum(p["has_signal"] for p in predictions.values()) / len(truth),
        "unknown_chunks": len(unknown), "inferred_unknown_chunks": len(inferred),
        "unknown_chunk_accuracy": (sum(result["chunk_topics"][c] == private["chunk_topics"][c]
                                       for c in inferred) / len(inferred)) if inferred else None,
        "mean_overlap_at_3": float(np.mean([
            row["overlap_at_3"] for row in private["retrieval_utility"].values()
        ])) if "retrieval_utility" in private else None,
    }


def score_saved(directory):
    """Replay saved attacker inputs, then separately score with evaluator labels."""
    from log_attack import run_attack
    directory = Path(directory)
    view = json.loads((directory / "attacker_view.json").read_text())
    private = json.loads((directory / "evaluator_only.json").read_text())
    predictions = run_attack(view)
    (directory / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    report = evaluate(view, private, predictions)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(directory, level="AP-set", known_per_topic=1, seed=42, context_count=80,
            approximation_factor=None, metric="cosine"):
    """Use matching inputs and the unchanged attacker on both saved backend views."""
    directory = Path(directory)
    reports = []
    for backend in ("plaintext", "cloaked"):
        destination = directory / backend
        prepare(destination, level, known_per_topic, seed, context_count, backend,
                approximation_factor, metric)
        reports.append(score_saved(destination))
    (directory / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    prep.add_argument("--backend", choices=["plaintext", "cloaked"], default="plaintext")
    comparison = commands.add_parser("compare")
    comparison.add_argument("directory", type=Path)
    for command in (prep, comparison):
        command.add_argument("--level", choices=sorted(LEVELS), default="AP-set")
        command.add_argument("--known-per-topic", type=int, default=1)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--context-count", type=int, default=80)
        command.add_argument("--approximation-factor", type=float, default=None)
        command.add_argument("--metric", choices=["cosine", "dot_product", "l2"], default="cosine")
    score = commands.add_parser("evaluate")
    score.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        reports = compare(args.directory, args.level, args.known_per_topic,
                          args.seed, args.context_count, args.approximation_factor, args.metric)
        print(json.dumps(reports, indent=2))
    elif args.command == "prepare":
        view, _ = prepare(args.directory, args.level, args.known_per_topic,
                          args.seed, args.context_count, args.backend, args.approximation_factor,
                          args.metric)
        print(f"Saved {len(view['context'])} context logs, 8 challenges, "
              f"{len(view['known_labels'])} known labels. Backend: {args.backend}.")
        print("Give the attacker only attacker_view.json; keep evaluator_only.json separate.")
    else:
        def read(name):
            return json.loads((args.directory / name).read_text())
        report = evaluate(read("attacker_view.json"), read("evaluator_only.json"),
                          read("predictions.json"))
        (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
