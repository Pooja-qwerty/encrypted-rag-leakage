"""Export actual Cloaked server observations for Phase 2 development replay."""

import argparse
import asyncio
import hashlib
import json
import secrets
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ironcore_alloy import AlloyMetadata
from sklearn.feature_extraction.text import TfidfVectorizer

from ciphertext_math import METRICS, rank_scores, score_vectors
from cloak.retrieval import encrypt_index, encrypt_query_vectors
from cloak.sdk import DEFAULT_TENANT_ID, build_standalone_sdk
from evaluate_project import load_workload
from game3 import load_file
from phase2_views import SCHEMA_VERSION, VIEWS, project_view, replay


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


async def export_async(directory, context_count=48, history_count=24, history_length=1,
                       known_per_topic=1, approximation_factor=1.5, metric="cosine", seed=42,
                       key_bytes=None):
    if (type(context_count) is not int or context_count < 0 or type(history_count) is not int
            or history_count < 1 or type(history_length) is not int or history_length not in (1, 5)
            or not isinstance(metric, str) or metric not in METRICS
            or type(known_per_topic) is not int or not 0 <= known_per_topic <= 12):
        raise ValueError("Invalid context/history counts, metric, history length, or known-label budget")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory must be empty; preserve previous randomized traces")
    workload = load_workload()
    # This seed controls workload draws only. Challenge bits and IDs use secrets.
    rng = np.random.default_rng(seed)
    cs, bio = load_file("data/CS.txt"), load_file("data/BIO.txt")
    texts, labels = cs + bio, ["CS"] * len(cs) + ["Bio"] * len(bio)
    refs = [f"CS:{i}" for i in range(len(cs))] + [f"Bio:{i}" for i in range(len(bio))]
    shuffle = np.random.default_rng(secrets.randbits(128)).permutation(len(texts))
    records = [{"id": "c_" + secrets.token_hex(12), "text": texts[i], "topic": labels[i], "ref": refs[i]}
               for i in shuffle]
    records.sort(key=lambda record: record["id"])
    record_ids = [record["id"] for record in records]
    truth = {r["id"]: r["topic"] for r in records}
    ref_to_id = {r["ref"]: r["id"] for r in records}
    known = {}
    for topic in ("CS", "Bio"):
        members = [cid for cid in record_ids if truth[cid] == topic]
        known.update({str(cid): topic for cid in rng.permutation(members)[:known_per_topic]})
    embedder = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vectors = embedder.fit_transform([r["text"] for r in records]).toarray()
    options = {"approximation_factor": approximation_factor}
    if key_bytes is not None:
        options["key_bytes"] = key_bytes
    sdk = build_standalone_sdk(**options)
    metadata = AlloyMetadata.new_simple(DEFAULT_TENANT_ID)
    encrypted = await encrypt_index(vectors, sdk, metadata)
    index = {cid: [float(x) for x in ev.encrypted_vector] for cid, ev in zip(record_ids, encrypted)}
    matrix = np.asarray([index[cid] for cid in record_ids], dtype=float)
    private_events = {}

    async def request(text, source_id, topic=None, relevant=None):
        event_id = "e_" + secrets.token_hex(12)
        plaintext = embedder.transform([text]).toarray()[0]
        # Do not cache tokens: even repeated texts make a fresh SDK request.
        query_pack = await encrypt_query_vectors(plaintext, sdk, metadata, query_id=event_id)
        if len(query_pack) != 1:
            raise ValueError("Phase 2 currently supports one active vector key")
        query = [float(x) for x in query_pack[0].encrypted_vector]
        scores = score_vectors(matrix, query, metric)
        ranking = rank_scores(scores, record_ids, 3)
        result_ids = [record_ids[i] for i in ranking]
        plain_scores = score_vectors(vectors, plaintext, "cosine")
        plain_ranking = rank_scores(plain_scores, record_ids, 3)
        private_events[event_id] = {"source_query_id": source_id, "query_topic": topic,
                                    "plain_topk": [record_ids[i] for i in plain_ranking],
                                    "overlap_at_3": len(set(ranking) & set(plain_ranking)) / 3}
        if relevant is not None:
            private_events[event_id]["relevant_ids"] = [ref_to_id[ref] for ref in relevant]
        return {"event_id": event_id, "result_count": 3, "result_ids": result_ids,
                "result_scores": [float(scores[i]) for i in ranking],
                "all_scores": scores.tolist(), "encrypted_query": query}

    # Generate all side-independent context before drawing any challenge bit.
    context = []
    for i in rng.integers(len(workload["context"]), size=context_count):
        context.append(await request(workload["context"][i], f"context_{i}"))
    pools = {topic: [q for q in workload["challenges"] if q["topic"] == topic]
             for topic in ("CS", "Bio")}
    histories, private_histories = [], {}
    for _ in range(history_count):
        bit = secrets.randbelow(2)
        majority = ("CS", "Bio")[bit]
        minority = ("CS", "Bio")[1 - bit]
        count = 1 if history_length == 1 else 4
        chosen = [pools[majority][i] for i in rng.choice(len(pools[majority]), count, replace=False)]
        if history_length == 5:
            chosen.append(pools[minority][int(rng.integers(len(pools[minority])))])
            rng.shuffle(chosen)
        hid = "h_" + secrets.token_hex(12)
        events = [await request(q["text"], q["id"], q["topic"], q["relevant"]) for q in chosen]
        histories.append({"history_id": hid, "events": events})
        private_histories[hid] = {"challenge_bit": bit, "majority_topic": majority,
                                  "event_ids": [e["event_id"] for e in events]}
    combined = {"schema_version": SCHEMA_VERSION, "view": "combined",
                "setup": {"record_ids": record_ids, "vector_dimension": matrix.shape[1],
                          "embedding_algorithm": "tfidf-word-unigram-bigram-l2", "metric": metric,
                          "approximation_factor": approximation_factor, "k": 3,
                          "topic_pair": ["CS", "Bio"], "context_count": context_count,
                          "history_count": history_count, "history_length": history_length,
                          "majority_count": 1 if history_length == 1 else 4,
                          "session_linkage": history_length > 1, "tie_breaker": "record_id_ascending",
                          "score_rule_version": 1},
                "known_labels": known, "context": context, "histories": histories,
                "encrypted_index": index}
    # Validate and reproduce the complete trace before creating any exports.
    replay(combined)
    manifests = []
    for view in VIEWS:
        packet = project_view(combined, view)
        report = replay(packet)["summary"]
        destination = directory / "views" / view / "attacker_view.json"
        write_json(destination, packet)
        manifests.append({"view": view, "file": str(destination.relative_to(directory)),
                          "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), **report})
    evaluator = {"evaluation_role": "development_only", "chunk_topics": truth,
                 "events": private_events, "histories": private_histories,
                 "run": {"workload": workload["name"], "workload_seed": seed,
                         "known_per_topic": known_per_topic, "ironcore_alloy_version": version("ironcore-alloy"),
                         "numpy_version": version("numpy"), "sklearn_version": version("scikit-learn"),
                         "challenge_bits": "secrets.randbelow(2); not derived from workload_seed",
                         "note": "Existing Phase 1 questions; these are not a fresh final test set."}}
    write_json(directory / "evaluator_only.json", evaluator)
    manifest = {"schema_version": SCHEMA_VERSION, "evaluation_role": "development_only", "views": manifests}
    write_json(directory / "export_manifest.json", manifest)
    return manifest


def export_development(directory, **options):
    return asyncio.run(export_async(directory, **options))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--context-count", type=int, default=48)
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--history-length", type=int, choices=[1, 5], default=1)
    parser.add_argument("--known-per-topic", type=int, default=1)
    parser.add_argument("--approximation-factor", type=float, default=1.5)
    parser.add_argument("--metric", choices=sorted(METRICS), default="cosine")
    parser.add_argument("--seed", type=int, default=42, help="workload draws only; does not set challenge bits")
    args = parser.parse_args()
    manifest = export_development(args.directory, context_count=args.context_count, history_count=args.histories,
                                  history_length=args.history_length, known_per_topic=args.known_per_topic,
                                  approximation_factor=args.approximation_factor, metric=args.metric, seed=args.seed)
    print(f"Saved {len(manifest['views'])} development views; no Phase 2 inference result is claimed.")
    print("Give an attacker one view file only. Keep evaluator and manifest files separate.")
    for view in manifest["views"]:
        print(f"{view['view']}: {view['events']} events validated, {view['recomputed_events']} recomputed")


if __name__ == "__main__":
    main()
