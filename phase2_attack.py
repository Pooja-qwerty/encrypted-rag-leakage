"""Phase 2 inference from one allowlisted attacker view, without evaluator access."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ciphertext_math import score_vectors
from log_attack import infer_chunk_topics
from phase2_views import replay

ATTACK_VERSION = 1
EPSILON = 1e-10
PARAMETERS = {"graph_neighbors": 5, "query_neighbors": 5, "graph_steps": 50,
              "graph_alpha": .85, "combined_geometry_weight": .5,
              "fallback_guess": 0, "evidence_epsilon": EPSILON}
METHODS = {
    "volume": ("constant",),
    "ids": ("constant", "known_only", "co_retrieval"),
    "ranked_ids": ("constant", "known_only", "co_retrieval"),
    "scores": ("constant", "known_only", "co_retrieval", "score_known", "score_cluster"),
    "ciphertexts": ("constant", "known_only", "co_retrieval", "score_known", "score_cluster",
                    "anchor_geometry", "geometry", "combined"),
    "combined": ("constant", "known_only", "co_retrieval", "score_known", "score_cluster",
                 "anchor_geometry", "geometry", "combined"),
}


def softmax_weights(values, scale):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    weights = np.exp((values - values.max()) / max(float(scale), 1e-12))
    return weights / weights.sum()


def scale_from(values):
    scale = float(np.std(values)) if len(values) else 0.
    return scale if scale > 1e-12 else 1.


def votes(entry, topics, topic_pair, weights):
    totals = [0., 0.]
    for cid, weight in zip(entry.get("result_ids", []), weights):
        if cid in topics:
            totals[topic_pair.index(topics[cid])] += float(weight)
    return totals[1] - totals[0]


def co_retrieval_topics(packet, observed):
    topic_pair = packet["setup"]["topic_pair"]
    canonical = {topic_pair[0]: "CS", topic_pair[1]: "Bio"}
    known = {cid: canonical[t] for cid, t in packet["known_labels"].items()}
    if not known:
        return {}
    context = [{"num_results": e["result_count"], "chunk_ids": observed[e["event_id"]].get("result_ids", [])}
               for e in packet["context"]]
    fitted = infer_chunk_topics(context, known)
    return {cid: topic_pair[0] if t == "CS" else topic_pair[1] for cid, t in fitted.items()}


def fit_geometry(packet):
    """Clamped label propagation on a symmetric encrypted-record neighbor graph."""
    setup = packet["setup"]
    ids, topic_pair = setup["record_ids"], setup["topic_pair"]
    matrix = np.asarray([packet["encrypted_index"][cid] for cid in ids])
    similarities = np.asarray([score_vectors(matrix, row, setup["metric"]) for row in matrix])
    n = len(ids)
    scale = scale_from(similarities[~np.eye(n, dtype=bool)])
    posterior = np.full((n, 2), .5)
    anchors = {i: topic_pair.index(packet["known_labels"][cid])
               for i, cid in enumerate(ids) if cid in packet["known_labels"]}
    for i, label in anchors.items():
        posterior[i] = np.eye(2)[label]
    if n > 1:
        affinity = np.zeros((n, n))
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            neighbors = sorted(candidates, key=lambda j: (-similarities[i, j], ids[j]))[:PARAMETERS["graph_neighbors"]]
            affinity[i, neighbors] = softmax_weights(similarities[i, neighbors], scale)
        affinity = (affinity + affinity.T) / 2
        transition = affinity / affinity.sum(axis=1, keepdims=True)
        for _ in range(PARAMETERS["graph_steps"]):
            posterior = PARAMETERS["graph_alpha"] * (transition @ posterior) + (1 - PARAMETERS["graph_alpha"]) * .5
            for i, label in anchors.items():
                posterior[i] = np.eye(2)[label]
    return {"posterior": posterior, "scale": scale,
            "anchor_indices": {label: [i for i, cid in enumerate(ids)
                                       if packet["known_labels"].get(cid) == topic_pair[label]]
                               for label in (0, 1)}}


def decision(evidence):
    has_signal = abs(evidence) > EPSILON
    return {"guess": int(evidence > EPSILON), "has_signal": has_signal,
            "evidence": float(evidence) if has_signal else 0.}


def run_attack(packet, method):
    start = time.perf_counter()
    reconstructed = replay(packet)  # Validates fields; ciphertexts reproduce their own logs.
    view = packet["view"]
    if method not in METHODS[view]:
        raise ValueError("Method requires observations absent from this view")
    observed = reconstructed["events"]
    setup, known = packet["setup"], dict(packet["known_labels"])
    topics = co_retrieval_topics(packet, observed) if method in {"co_retrieval", "score_cluster", "combined"} else known
    top_scores = [s for entry in packet["context"] for s in observed[entry["event_id"]].get("result_scores", [])]
    score_scale = scale_from(top_scores)
    geometry = fit_geometry(packet) if method in {"anchor_geometry", "geometry", "combined"} else None
    fit_seconds = time.perf_counter() - start

    def evidence(event):
        entry = observed[event["event_id"]]
        ids = entry.get("result_ids", [])
        if method == "constant":
            return 0.
        if method in {"known_only", "co_retrieval"}:
            weights = [1.] * len(ids) if view == "ids" else [1 / (rank + 1) for rank in range(len(ids))]
            return votes(entry, topics, setup["topic_pair"], weights)
        if method in {"score_known", "score_cluster", "combined"}:
            weights = softmax_weights(entry["result_scores"], score_scale)
            score_evidence = votes(entry, topics, setup["topic_pair"], weights)
            if method != "combined":
                return score_evidence
        scores = np.asarray(entry["all_scores"])
        if method == "anchor_geometry":
            first, second = geometry["anchor_indices"][0], geometry["anchor_indices"][1]
            if not first or not second:
                return 0.
            return float(np.tanh((scores[second].mean() - scores[first].mean()) / geometry["scale"]))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], setup["record_ids"][i]))[:PARAMETERS["query_neighbors"]]
        weights = softmax_weights(scores[order], geometry["scale"])
        probabilities = weights @ geometry["posterior"][order]
        geometry_evidence = float(probabilities[1] - probabilities[0])
        if method == "combined":
            weight = PARAMETERS["combined_geometry_weight"]
            return weight * geometry_evidence + (1 - weight) * score_evidence
        return geometry_evidence

    prediction_start = time.perf_counter()
    events, histories = {}, {}
    for history in packet["histories"]:
        values = []
        for event in history["events"]:
            value = evidence(event)
            events[event["event_id"]] = decision(value)
            values.append(value)
        # Use all linked requests, never labels or another challenge history.
        histories[history["history_id"]] = decision(float(np.mean(values)))
    return {"attack_version": ATTACK_VERSION, "view": view, "method": method,
            "parameters": {**PARAMETERS, "score_scale": score_scale,
                           "geometry_scale": geometry["scale"] if geometry else None},
            "event_predictions": events, "history_predictions": histories,
            "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - prediction_start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", required=True, choices=sorted({m for methods in METHODS.values() for m in methods}))
    args = parser.parse_args()
    result = run_attack(json.loads(args.view_file.read_text()), args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['history_predictions'])} history predictions from {result['view']}; no labels read.")


if __name__ == "__main__":
    main()
