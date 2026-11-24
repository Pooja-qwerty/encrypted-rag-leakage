"""Trace-based fetch-padding experiment with explicit observer controls."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_project import project_view, relevance_scores, select_known, session_scores
from leakage_experiment import evaluate
from log_attack import run_attack, run_session_attack


def pad_fetches(base, inventory, dummy_count, seed):
    """Client mixes real requests with uniformly sampled distinct cover records.

    Only an unordered storage-fetch observer receives this view. The search
    server's original result/ranking is not changed or hidden by this step.
    """
    if type(dummy_count) is not int or dummy_count < 0:
        raise ValueError("Dummy count must be a nonnegative integer")
    inventory = sorted(set(inventory))
    rng = np.random.default_rng(seed)

    def pad(entry):
        real = set(entry["chunk_ids"])
        if not real <= set(inventory) or dummy_count > len(inventory) - len(real):
            raise ValueError("Invalid inventory or too many dummy fetches")
        candidates = [cid for cid in inventory if cid not in real]
        # Permuting all candidates makes the cover sets nested across budgets.
        cover = list(rng.permutation(candidates)[:dummy_count])
        ids = sorted(real | set(cover))
        result = {"num_results": len(ids), "chunk_ids": ids}
        if "query_id" in entry:
            result["query_id"] = entry["query_id"]
        return result

    return {"level": "AP-set", "known_labels": dict(base["known_labels"]),
            "context": [pad(e) for e in base["context"]],
            "challenges": [pad(e) for e in base["challenges"]]}


def filter_fetch_batch(batch, real_ranking):
    """The client keeps its original top-k in order, discarding cover records."""
    if not set(real_ranking) <= set(batch):
        raise ValueError("Fetch batch is missing a real result")
    return [cid for cid in real_ranking if cid in set(batch)]


def full_download_view(base, inventory):
    """Protocol-model baseline: fetch every record, send no query to the server."""
    inventory = sorted(set(inventory))

    def all_records(entry):
        result = {"num_results": len(inventory), "chunk_ids": list(inventory)}
        if "query_id" in entry:
            result["query_id"] = entry["query_id"]
        return result

    return {"level": "AP-set", "known_labels": dict(base["known_labels"]),
            "context": [all_records(e) for e in base["context"]],
            "challenges": [all_records(e) for e in base["challenges"]]}


def client_quality(private, observed, full_download=False):
    """Model delivered results separately from the larger visible fetch batch.

    Full download uses saved plaintext rankings to model client-local search;
    this is not a deployed download/decryption protocol.
    """
    client = copy.deepcopy(private)
    by_query = {entry["query_id"]: entry for entry in observed["challenges"]}
    for qid, row in client["retrieval_utility"].items():
        ranking = row["plain_topk"] if full_download else row["backend_topk"]
        row["backend_topk"] = filter_fetch_batch(by_query[qid]["chunk_ids"], ranking)
        row["overlap_at_3"] = len(set(row["backend_topk"]) & set(row["plain_topk"])) / 3
    return client


def summarize(rows):
    groups = {}
    dimensions = ["backend", "approximation_factor", "protocol", "observer", "dummy_count", "method"]
    for row in rows:
        groups.setdefault(tuple(row[key] for key in dimensions), []).append(row)
    summaries = []
    metrics = ["query_accuracy", "advantage", "session_accuracy", "session_advantage",
               "precision_at_3", "ndcg_at_3", "fetch_records_per_query", "fetch_multiplier",
               "query_accuracy_change", "session_accuracy_change"]
    for key, group in groups.items():
        entry = dict(zip(dimensions, key))
        entry["evaluations"] = len(group)
        entry["query_sent_to_server"] = group[0]["query_sent_to_server"]
        for metric in metrics:
            values = np.array([r[metric] for r in group])
            entry[metric + "_mean"] = float(values.mean())
            entry[metric + "_std"] = float(values.std(ddof=1)) if len(group) > 1 else 0.
        summaries.append(entry)
    return summaries


def run_defense_study(source, directory, dummy_counts=(0, 3, 9, 21), cover_seeds=3,
                      label_seeds=10, budget=4):
    """Reuse existing encrypted retrieval traces for an exact before/after pairing."""
    source, directory = Path(source), Path(directory)
    if (cover_seeds < 1 or label_seeds < 1 or budget < 0 or not dummy_counts
            or any(type(d) is not int or d < 0 for d in dummy_counts)
            or len(set(dummy_counts)) != len(dummy_counts)):
        raise ValueError("Invalid padding or evaluation settings")
    configurations = sorted(p for p in source.iterdir() if p.is_dir()
                            and (p / "attacker_view.json").is_file())
    if not configurations:
        raise ValueError("No saved study traces found; run evaluate_project.py first")
    directory.mkdir(parents=True, exist_ok=True)
    settings = {"source": str(source.resolve()), "dummy_counts": list(dummy_counts),
                "cover_seeds": cover_seeds, "label_seeds": label_seeds,
                "known_per_topic": budget, "level": "AP-set",
                "scope": "Trace simulation; fixed-size record counts only, not bytes or latency.",
                "padding_observer": "Separate fetch observer sees mixed IDs, not search results, "
                                    "scores, query vectors, or client filtering decisions.",
                "search_control": "Original result IDs remain visible to the search server.",
                "full_download": "Protocol model: all records fetched; query and ranking stay local. "
                                 "Saved plaintext rankings model client-local selection.",
                "limits": "Only the two existing attacks are evaluated. Padding-aware recovery, "
                          "timing, repeated-query intersection attacks, and collusion are not hidden."}
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows = []
    for configuration in configurations:
        base = json.loads((configuration / "attacker_view.json").read_text())
        private = json.loads((configuration / "evaluator_only.json").read_text())
        sessions = json.loads((configuration / "session_view.json").read_text())
        destination_root = directory / configuration.name
        destination_root.mkdir(parents=True, exist_ok=True)
        (destination_root / "session_view.json").write_text(json.dumps(sessions, indent=2) + "\n")
        inventory = sorted(private["chunk_topics"])
        n, k = len(inventory), private["run"]["k"]
        if any(d > n - k for d in dummy_counts):
            raise ValueError("Dummy count exceeds unused records in this corpus")
        unpadded = project_view(base, {}, "AP-set")
        baselines = {}
        for anchor_seed in range(label_seeds):
            known = select_known(private["chunk_topics"], budget, anchor_seed)
            view = project_view(base, known, "AP-set")
            for method in ("known_only", "clustering"):
                attack = run_attack(view, method)
                predictions = run_session_attack(view, sessions, attack["chunk_topics"])
                baselines[anchor_seed, method] = {
                    **evaluate(view, private, attack),
                    **session_scores(predictions, private["session_topics"])}

        scenarios = []
        for count in dummy_counts:
            for cover_seed in range(1 if count == 0 else cover_seeds):
                observed = pad_fetches(base, inventory, count, cover_seed)
                scenarios.append(("server_search_plus_padding", "fetch_only", count, cover_seed,
                                  observed, client_quality(private, observed), True))
            if count > 0:
                # A negative control: the full search server still sees real IDs.
                scenarios.append(("server_search_plus_padding", "search_result_control", count, None,
                                  unpadded, private, True))
        all_records = full_download_view(base, inventory)
        scenarios.append(("client_search_full_download", "server_no_query", n - k, None,
                          all_records, client_quality(private, all_records, full_download=True), False))
        for protocol, observer, count, cover_seed, observed, client, sent_query in scenarios:
            name = f"{observer}_dummy_{count}_seed_{cover_seed}"
            destination = directory / configuration.name / name
            destination.mkdir(parents=True, exist_ok=True)
            # Labels are disclosed per draw below; this base view contains none.
            (destination / "observed_view.json").write_text(json.dumps(observed, indent=2) + "\n")
            artifact = copy.deepcopy(client)
            artifact["run"]["defense"] = {"protocol": protocol, "observer": observer,
                                         "dummy_count": count, "cover_seed": cover_seed,
                                         "query_sent_to_server": sent_query}
            (destination / "evaluator_only.json").write_text(json.dumps(artifact, indent=2) + "\n")
            quality = relevance_scores(client)
            for anchor_seed in range(label_seeds):
                known = select_known(private["chunk_topics"], budget, anchor_seed)
                view = project_view(observed, known, "AP-set")
                for method in ("known_only", "clustering"):
                    attack = run_attack(view, method)
                    predictions = run_session_attack(view, sessions, attack["chunk_topics"])
                    report = {**evaluate(view, client, attack), **quality,
                              **session_scores(predictions, private["session_topics"])}
                    baseline = baselines[anchor_seed, method]
                    if observer == "search_result_control":
                        assert report["query_accuracy"] == baseline["query_accuracy"]
                        assert report["session_accuracy"] == baseline["session_accuracy"]
                    rows.append({"backend": private["run"]["backend"],
                                 "approximation_factor": private["run"].get("approximation_factor"),
                                 "key_index": private["run"].get("key_index"),
                                 "protocol": protocol, "observer": observer, "dummy_count": count,
                                 "cover_seed": cover_seed, "anchor_seed": anchor_seed,
                                 "budget": budget, "method": method, "query_sent_to_server": sent_query,
                                 "client_topk_preserved": all(
                                     r["backend_topk"] == private["retrieval_utility"][q]["backend_topk"]
                                     for q, r in client["retrieval_utility"].items()),
                                 "fetch_records_per_query": k + count, "fetch_multiplier": (k + count) / k,
                                 "query_accuracy_change": report["query_accuracy"] - baseline["query_accuracy"],
                                 "session_accuracy_change": report["session_accuracy"] - baseline["session_accuracy"],
                                 **report})
                    if anchor_seed == 0:
                        for filename, payload in [(f"example_{method}_view", view),
                                                  (f"example_{method}_predictions", attack),
                                                  (f"example_{method}_sessions", predictions)]:
                            (destination / f"{filename}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Completed defense/control pairing for {configuration.name}", flush=True)
    with (directory / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"settings": settings, "summaries": summarize(rows)}
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dummies", nargs="+", type=int, default=[0, 3, 9, 21])
    parser.add_argument("--cover-seeds", type=int, default=3)
    parser.add_argument("--label-seeds", type=int, default=10)
    parser.add_argument("--known-per-topic", type=int, default=4)
    args = parser.parse_args()
    result = run_defense_study(args.source, args.directory, args.dummies, args.cover_seeds,
                               args.label_seeds, args.known_per_topic)
    print("observer / dummies: query accuracy / session accuracy / fetch multiplier (Cloaked 1.5, direct)")
    for row in result["summaries"]:
        if row["backend"] == "cloaked" and row["approximation_factor"] == 1.5 and row["method"] == "known_only":
            print(f"{row['observer']} / {row['dummy_count']}: {row['query_accuracy_mean']:.3f} / "
                  f"{row['session_accuracy_mean']:.3f} / {row['fetch_multiplier_mean']:.1f}x")


if __name__ == "__main__":
    main()
