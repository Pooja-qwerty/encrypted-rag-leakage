"""Allowlisted Phase 2 views and standalone replay. No SDK or private-file imports."""

import argparse
import copy
import json
import math
import re
from pathlib import Path

import numpy as np

from ciphertext_math import METRICS, rank_scores, score_vectors

SCHEMA_VERSION = 1
VIEWS = ("volume", "ids", "ranked_ids", "scores", "ciphertexts", "combined")
SETUP_FIELDS = {"record_ids", "vector_dimension", "embedding_algorithm", "metric",
                "approximation_factor", "k", "topic_pair", "context_count", "history_count",
                "history_length", "majority_count", "session_linkage", "tie_breaker", "score_rule_version"}


def _keys(value, fields, location):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"Unexpected or missing fields in {location}")


def _positive(value, location):
    if type(value) is not int or value < 1:
        raise ValueError(f"Expected positive integer for {location}")


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, length, location):
    if not isinstance(value, list) or len(value) != length or not all(_number(x) for x in value):
        raise ValueError(f"Invalid finite vector in {location}")


def _opaque(value, prefix):
    return isinstance(value, str) and re.fullmatch(prefix + r"_[0-9a-f]{24}", value) is not None


def event_fields(view):
    fields = {"event_id", "result_count"}
    if view in {"ids", "ranked_ids", "scores", "combined"}:
        fields.add("result_ids")
    if view in {"scores", "combined"}:
        fields.add("result_scores")
    if view in {"ciphertexts", "combined"}:
        fields.add("encrypted_query")
    if view == "combined":
        fields.add("all_scores")
    return fields


def validate_view(packet):
    if not isinstance(packet, dict) or packet.get("view") not in VIEWS:
        raise ValueError("Unknown Phase 2 view")
    view = packet["view"]
    fields = {"schema_version", "view", "setup", "known_labels", "context", "histories"}
    if view in {"ciphertexts", "combined"}:
        fields.add("encrypted_index")
    _keys(packet, fields, "view")
    if type(packet["schema_version"]) is not int or packet["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported Phase 2 schema version")
    setup = packet["setup"]
    _keys(setup, SETUP_FIELDS, "public setup")
    inventory = setup["record_ids"]
    if (not isinstance(inventory, list) or not inventory
            or not all(_opaque(cid, "c") for cid in inventory)
            or inventory != sorted(set(inventory))):
        raise ValueError("Expected sorted unique opaque record inventory")
    for field in ("vector_dimension", "k", "history_count", "history_length", "majority_count"):
        _positive(setup[field], field)
    if (type(setup["context_count"]) is not int or setup["context_count"] < 0
            or setup["k"] > len(inventory) or setup["majority_count"] > setup["history_length"]
            or 2 * setup["majority_count"] <= setup["history_length"]):
        raise ValueError("Invalid count or history shape")
    if (not isinstance(setup["metric"], str) or setup["metric"] not in METRICS
            or not _number(setup["approximation_factor"])
            or setup["approximation_factor"] <= 0
            or setup["embedding_algorithm"] != "tfidf-word-unigram-bigram-l2"
            or setup["tie_breaker"] != "record_id_ascending"
            or type(setup["score_rule_version"]) is not int or setup["score_rule_version"] != 1
            or type(setup["session_linkage"]) is not bool
            or setup["session_linkage"] != (setup["history_length"] > 1)):
        raise ValueError("Unsupported public search configuration")
    topics = setup["topic_pair"]
    if (not isinstance(topics, list) or len(topics) != 2 or any(not isinstance(t, str) or not t for t in topics)
            or len(set(topics)) != 2):
        raise ValueError("Expected two distinct topic names")
    known = packet["known_labels"]
    if (not isinstance(known, dict) or not set(known) <= set(inventory)
            or any(not isinstance(t, str) or t not in topics for t in known.values())):
        raise ValueError("Invalid disclosed labels")
    if view in {"ciphertexts", "combined"}:
        index = packet["encrypted_index"]
        _keys(index, inventory, "encrypted index")
        for vector in index.values():
            _vector(vector, setup["vector_dimension"], "encrypted index")
    if not isinstance(packet["context"], list) or len(packet["context"]) != setup["context_count"]:
        raise ValueError("Context count mismatch")
    if not isinstance(packet["histories"], list) or len(packet["histories"]) != setup["history_count"]:
        raise ValueError("History count mismatch")
    seen_events, seen_histories = set(), set()

    def event(entry):
        _keys(entry, event_fields(view), "request")
        eid = entry["event_id"]
        if not _opaque(eid, "e") or eid in seen_events:
            raise ValueError("Event IDs must be fresh opaque IDs")
        seen_events.add(eid)
        if type(entry["result_count"]) is not int or entry["result_count"] != setup["k"]:
            raise ValueError("Result count must equal public k")
        if "result_ids" in entry:
            ids = entry["result_ids"]
            if (not isinstance(ids, list) or len(ids) != setup["k"] or not all(isinstance(i, str) for i in ids)
                    or len(set(ids)) != len(ids) or not set(ids) <= set(inventory)):
                raise ValueError("Invalid returned record IDs")
            if view == "ids" and ids != sorted(ids):
                raise ValueError("Unordered ID view must use canonical order")
        for field, length in (("result_scores", setup["k"]), ("all_scores", len(inventory)),
                              ("encrypted_query", setup["vector_dimension"])):
            if field in entry:
                _vector(entry[field], length, field)

    for entry in packet["context"]:
        event(entry)
    for history in packet["histories"]:
        _keys(history, {"history_id", "events"}, "history")
        hid = history["history_id"]
        if not _opaque(hid, "h") or hid in seen_histories:
            raise ValueError("History IDs must be unique opaque IDs")
        seen_histories.add(hid)
        if not isinstance(history["events"], list) or len(history["events"]) != setup["history_length"]:
            raise ValueError("History length mismatch")
        for entry in history["events"]:
            event(entry)


def project_view(combined, view):
    validate_view(combined)
    if combined["view"] != "combined" or view not in VIEWS:
        raise ValueError("Projection requires a combined trace and a supported view")
    fields = event_fields(view)

    def project(entry):
        result = {name: copy.deepcopy(entry[name]) for name in fields}
        if view == "ids":
            result["result_ids"] = sorted(result["result_ids"])
        return result

    result = {"schema_version": SCHEMA_VERSION, "view": view,
              "setup": copy.deepcopy(combined["setup"]), "known_labels": dict(combined["known_labels"]),
              "context": [project(e) for e in combined["context"]],
              "histories": [{"history_id": h["history_id"], "events": [project(e) for e in h["events"]]}
                            for h in combined["histories"]]}
    if view in {"ciphertexts", "combined"}:
        result["encrypted_index"] = copy.deepcopy(combined["encrypted_index"])
    validate_view(result)
    return result


def replay(packet):
    """Recompute from ciphertexts if available; otherwise validate declared logs."""
    validate_view(packet)
    setup, view = packet["setup"], packet["view"]
    has_vectors = view in {"ciphertexts", "combined"}
    if has_vectors:
        matrix = np.asarray([packet["encrypted_index"][cid] for cid in setup["record_ids"]])
    events = packet["context"] + [e for h in packet["histories"] for e in h["events"]]
    reconstructed = {}
    for entry in events:
        result = {name: copy.deepcopy(value) for name, value in entry.items()
                  if name not in {"event_id", "encrypted_query"}}
        if has_vectors:
            scores = score_vectors(matrix, entry["encrypted_query"], setup["metric"])
            indices = rank_scores(scores, setup["record_ids"], setup["k"])
            ids = [setup["record_ids"][i] for i in indices]
            top_scores = [float(scores[i]) for i in indices]
            if view == "combined":
                if (entry["result_ids"] != ids or not np.allclose(entry["result_scores"], top_scores, rtol=1e-9, atol=1e-12)
                        or not np.allclose(entry["all_scores"], scores, rtol=1e-9, atol=1e-12)):
                    raise ValueError("Saved scores or ranks do not match ciphertext replay")
            result.update({"result_ids": ids, "result_scores": top_scores, "all_scores": scores.tolist()})
        reconstructed[entry["event_id"]] = result
    return {"summary": {"schema_version": SCHEMA_VERSION, "view": view, "events": len(events),
                        "histories": len(packet["histories"]), "status": "passed",
                        "recomputed_events": len(events) if has_vectors else 0,
                        "scope": "ciphertext_recompute" if has_vectors else "schema_validation_only"},
            "events": reconstructed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view_file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(json.loads(args.view_file.read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
