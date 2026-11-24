"""Topic inference from saved retrieval logs and a limited known-label map only."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

TOPICS = {"CS", "Bio"}
LEVELS = {"Volume", "AP-set", "AP-ordered"}


def validate_view(view):
    """Reject accidental extra fields, including evaluation labels and vectors."""
    if set(view) != {"level", "context", "challenges", "known_labels"}:
        raise ValueError("Expected only level, context, challenges, known_labels")
    level = view["level"]
    if level not in LEVELS:
        raise ValueError("Unsupported leakage level")
    if any(not isinstance(cid, str) or topic not in TOPICS
           for cid, topic in view["known_labels"].items()):
        raise ValueError("Known labels must map opaque string IDs to CS or Bio")
    qids = set()
    for entries, challenge in ((view["context"], False), (view["challenges"], True)):
        for entry in entries:
            expected = {"num_results"}
            if level != "Volume":
                expected.add("chunk_ids")
            if challenge:
                expected.add("query_id")
            if set(entry) != expected:
                raise ValueError("Unexpected log fields")
            count = entry["num_results"]
            if type(count) is not int or count < 0:
                raise ValueError("Invalid result count")
            if level != "Volume":
                ids = entry["chunk_ids"]
                if (not isinstance(ids, list) or any(not isinstance(cid, str) for cid in ids)
                        or len(set(ids)) != len(ids) or len(ids) != count):
                    raise ValueError("Invalid chunk IDs")
            if challenge:
                qid = entry["query_id"]
                if not isinstance(qid, str) or qid in qids:
                    raise ValueError("Challenge IDs must be unique strings")
                qids.add(qid)


def infer_chunk_topics(context, known_labels):
    """Adapt the simulation's spectral co-retrieval clustering to opaque IDs.

    Fit on context logs only. Known labels name clusters by majority vote;
    unnamed or tied clusters remain unknown. Directly known labels are retained.
    """
    ids = sorted({cid for entry in context for cid in entry.get("chunk_ids", [])})
    chunk_topics = dict(known_labels)
    if len(ids) < 2:
        return chunk_topics
    positions = {cid: i for i, cid in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)))
    for entry in context:
        indices = [positions[cid] for cid in entry.get("chunk_ids", [])]
        matrix[np.ix_(indices, indices)] += 1
    np.fill_diagonal(matrix, 0)
    seen = np.flatnonzero(matrix.sum(axis=1) > 0)
    if len(seen) < 2:
        return chunk_topics
    adjacency = matrix[np.ix_(seen, seen)]
    inverse_degree = 1 / np.sqrt(adjacency.sum(axis=1))
    normalized = adjacency * inverse_degree[:, None] * inverse_degree[None, :]
    _, eigenvectors = np.linalg.eigh(normalized)
    features = eigenvectors[:, -2:]
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(features)
    for cluster in set(clusters):
        members = [ids[seen[i]] for i in np.flatnonzero(clusters == cluster)]
        votes = Counter(known_labels[cid] for cid in members if cid in known_labels)
        if votes["CS"] == votes["Bio"]:
            continue
        topic = max(TOPICS, key=lambda t: votes[t])
        for cid in members:
            chunk_topics.setdefault(cid, topic)
    return chunk_topics


def query_votes(entry, chunk_topics, level):
    votes = {"CS": 0.0, "Bio": 0.0}
    for rank, cid in enumerate(entry.get("chunk_ids", [])):
        topic = chunk_topics.get(cid)
        if topic is not None:
            weight = 1 / (rank + 1) if level == "AP-ordered" else 1
            votes[topic] += weight
    return votes


def predict_session(entries, chunk_topics, level):
    votes = {"CS": 0.0, "Bio": 0.0}
    for entry in entries:
        for topic, score in query_votes(entry, chunk_topics, level).items():
            votes[topic] += score
    # A fixed fallback keeps every challenge/session in the denominator.
    return {"topic": "Bio" if votes["Bio"] > votes["CS"] else "CS",
            "has_signal": votes["CS"] != votes["Bio"]}


def predict_query(entry, chunk_topics, level):
    return predict_session([entry], chunk_topics, level)


def run_attack(view, method="clustering"):
    validate_view(view)
    if method == "clustering":
        topics = infer_chunk_topics(view["context"], view["known_labels"])
    elif method == "known_only":
        topics = dict(view["known_labels"])
    else:
        raise ValueError("Method must be clustering or known_only")
    predictions = {entry["query_id"]: predict_query(entry, topics, view["level"])
                   for entry in view["challenges"]}
    return {"chunk_topics": topics, "predictions": predictions}


def run_session_attack(view, session_view, chunk_topics):
    """Only session membership is additional leakage; no session/topic labels."""
    validate_view(view)
    if set(session_view) != {"sessions"}:
        raise ValueError("Expected session membership only")
    entries = {e["query_id"]: e for e in view["challenges"]}
    predictions = {}
    for session in session_view["sessions"]:
        if set(session) != {"session_id", "query_ids"}:
            raise ValueError("Unexpected session fields")
        sid, qids = session["session_id"], session["query_ids"]
        if (not isinstance(sid, str) or sid in predictions or not qids
                or len(set(qids)) != len(qids) or any(q not in entries for q in qids)):
            raise ValueError("Invalid session membership")
        predictions[sid] = predict_session([entries[q] for q in qids], chunk_topics, view["level"])
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", type=Path, help="attacker_view.json only")
    parser.add_argument("output", type=Path, help="prediction JSON")
    parser.add_argument("--method", choices=["clustering", "known_only"], default="clustering")
    parser.add_argument("--sessions", type=Path, help="optional saved session membership")
    args = parser.parse_args()
    view = json.loads(args.view.read_text())
    result = run_attack(view, args.method)
    if args.sessions:
        result["session_predictions"] = run_session_attack(
            view, json.loads(args.sessions.read_text()), result["chunk_topics"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Predicted {len(result['predictions'])} queries using the saved view only.")


if __name__ == "__main__":
    main()
