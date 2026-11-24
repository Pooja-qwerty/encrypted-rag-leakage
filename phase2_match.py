"""Exact log-signature matching for a conditional Phase 2 development diagnostic."""

import argparse
import copy
import hashlib
import json
import secrets
from collections import defaultdict
from pathlib import Path

from phase2_evaluate import (CONTROLS, control_view, evaluate_export, load_views,
                             score_predictions, write_json)
from phase2_views import SCHEMA_VERSION, VIEWS, project_view, replay, validate_view

MODES = ("unordered_ids", "ordered_ids", "result_counts_control")


def signature(history, mode):
    if mode not in MODES:
        raise ValueError("Unsupported matching mode")
    if mode == "result_counts_control":
        return tuple(e["result_count"] for e in history["events"])
    return tuple((e["result_count"], tuple(sorted(e["result_ids"]) if mode == "unordered_ids"
                                          else e["result_ids"])) for e in history["events"])


def select_pairs(combined, evaluator, mode):
    """Use only declared signatures and side membership, never attack outputs."""
    validate_view(combined)
    if combined["view"] != "combined" or mode not in MODES:
        raise ValueError("Matching requires a combined trace and a supported mode")
    histories = {h["history_id"]: h for h in combined["histories"]}
    truth = evaluator["histories"]
    if set(histories) != set(truth):
        raise ValueError("Source histories do not match evaluator labels")
    buckets = defaultdict(lambda: [[], []])
    for hid, history in histories.items():
        bit = truth[hid]["challenge_bit"]
        if type(bit) is not int or bit not in (0, 1):
            raise ValueError("Invalid challenge bit")
        buckets[signature(history, mode)][bit].append(hid)
    pairs = []
    eligible = possible = 0
    for log_signature, sides in sorted(buckets.items()):
        first, second = sorted(sides[0]), sorted(sides[1])
        if first and second:
            eligible += len(first) + len(second)
            possible += len(first) * len(second)
        # Maximum disjoint matching inside each bucket; each history used once.
        pairs.extend({"source_history_ids": [a, b], "signature": log_signature}
                     for a, b in zip(first, second))
    selected = {hid for pair in pairs for hid in pair["source_history_ids"]}
    by_side = {side: sum(t["challenge_bit"] == side for t in truth.values()) for side in (0, 1)}
    records = {cid for hid in selected for e in histories[hid]["events"] for cid in e["result_ids"]}
    source_questions = {evaluator["events"][e["event_id"]]["source_query_id"]
                        for hid in selected for e in histories[hid]["events"]}
    n = len(histories)
    coverage = {"mode": mode, "status": "matched" if pairs else "no_exact_matches",
                "source_histories": n, "source_topic0": by_side[0], "source_topic1": by_side[1],
                "unique_signatures": len(buckets),
                "cross_topic_signatures": sum(bool(a and b) for a, b in buckets.values()),
                "eligible_histories": eligible, "eligible_fraction": eligible / n,
                "possible_cross_pairs_not_independent": possible,
                "disjoint_pairs": len(pairs), "matched_histories": len(selected),
                "matched_fraction": len(selected) / n, "unmatched_histories": n - len(selected),
                "topic0_coverage": len(pairs) / by_side[0] if by_side[0] else None,
                "topic1_coverage": len(pairs) / by_side[1] if by_side[1] else None,
                "records_touched": len(records), "record_inventory_fraction": len(records) / len(combined["setup"]["record_ids"]),
                "unique_source_questions": len(source_questions),
                "pairing_rule": "Sort opaque history IDs and zip opposite sides within each exact signature; no history reuse",
                "claim_scope": "Result counts only; returned IDs can differ" if mode == "result_counts_control"
                               else "Exact per-request ID signature, retaining request order"}
    return pairs, coverage


def write_matched_export(combined, evaluator, pairs, mode, directory):
    """Blind membership/order/IDs; keep pairing and source labels evaluator-only."""
    if not pairs:
        raise ValueError("No matched export is created for an empty selection")
    directory = Path(directory)
    source_histories = {h["history_id"]: h for h in combined["histories"]}
    packet = copy.deepcopy(combined)
    histories, truth, private_pairs = [], {}, []
    events = {e["event_id"]: copy.deepcopy(evaluator["events"][e["event_id"]]) for e in combined["context"]}
    used_sources = set()
    for pair in pairs:
        source_ids = pair["source_history_ids"]
        if (len(source_ids) != 2 or any(hid in used_sources for hid in source_ids)
                or any(hid not in source_histories for hid in source_ids)):
            raise ValueError("Matched histories must be disjoint source pairs")
        if {evaluator["histories"][hid]["challenge_bit"] for hid in source_ids} != {0, 1}:
            raise ValueError("Each pair must contain opposite challenge sides")
        if signature(source_histories[source_ids[0]], mode) != signature(source_histories[source_ids[1]], mode):
            raise ValueError("Pair does not share the declared signature")
        used_sources.update(source_ids)
        blinded = []
        for old_id in source_ids:
            history = copy.deepcopy(source_histories[old_id])
            hid = "h_" + secrets.token_hex(12)
            history["history_id"] = hid
            for event in history["events"]:
                original = event["event_id"]
                event["event_id"] = "e_" + secrets.token_hex(12)
                events[event["event_id"]] = {**copy.deepcopy(evaluator["events"][original]),
                                            "source_event_id": original}
            histories.append(history)
            truth[hid] = {**copy.deepcopy(evaluator["histories"][old_id]), "source_history_id": old_id,
                          "event_ids": [e["event_id"] for e in history["events"]]}
            blinded.append(hid)
        private_pairs.append({"source_history_ids": source_ids, "history_ids": blinded,
                              "signature": pair["signature"]})
    secrets.SystemRandom().shuffle(histories)
    packet["histories"] = histories
    packet["setup"]["history_count"] = len(histories)
    replay(packet)
    manifest = {"schema_version": SCHEMA_VERSION, "evaluation_role": "development_only",
                "evaluation_design": "conditional_matched_development", "views": []}
    for name in VIEWS:
        view = project_view(packet, name)
        path = directory / "views" / name / "attacker_view.json"
        write_json(path, view)
        manifest["views"].append({"view": name, "file": str(path.relative_to(directory)),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **replay(view)["summary"]})
    private = {**copy.deepcopy(evaluator), "evaluation_design": "conditional_matched_development",
               "histories": truth, "events": events, "matching": {"mode": mode, "pairs": private_pairs}}
    write_json(directory / "evaluator_only.json", private)
    write_json(directory / "export_manifest.json", manifest)
    return packet, private


def subset_reports(packets, predictions_directory, evaluator, selected_ids):
    """Score frozen unconditional predictions on a selected population only."""
    if not selected_ids:
        return []
    labels = {"histories": {hid: t for hid, t in evaluator["histories"].items() if hid in selected_ids}}
    rows = []
    for path in sorted(Path(predictions_directory).glob("*/*/*.json")):
        control, view = path.parts[-3:-1]
        packet = control_view(packets[view], control)
        packet["histories"] = [h for h in packet["histories"] if h["history_id"] in selected_ids]
        packet["setup"]["history_count"] = len(packet["histories"])
        prediction = json.loads(path.read_text())
        prediction["history_predictions"] = {hid: p for hid, p in prediction["history_predictions"].items() if hid in selected_ids}
        rows.append({"control": control, **score_predictions(packet, prediction, labels)})
    return rows


def verify_conditional_predictions(original_directory, conditional_directory, private, mode):
    """Subsetting/blinding must not change a model fitted only on fixed context."""
    for path in sorted(Path(conditional_directory).glob("*/*/*.json")):
        relative = path.relative_to(conditional_directory)
        original = json.loads((Path(original_directory) / relative).read_text())["history_predictions"]
        conditioned = json.loads(path.read_text())["history_predictions"]
        for hid, prediction in conditioned.items():
            old_id = private["histories"][hid]["source_history_id"]
            if prediction != original[old_id]:
                raise ValueError("Conditioning changed a frozen attack's prediction")
        # Exact matching forces deterministic attacks on that log view to agree within each pair.
        reference_views = {"ids"} if mode == "unordered_ids" else {"ids", "ranked_ids"}
        if mode != "result_counts_control" and relative.parts[1] in reference_views:
            for pair in private["matching"]["pairs"]:
                first, second = pair["history_ids"]
                if conditioned[first] != conditioned[second]:
                    raise ValueError("Matched log view produced different per-pair predictions")


def analyze_export(source, directory, include_count_control=False):
    source, directory = Path(source), Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty matching output directory")
    # Freeze all attacks before matching learns any challenge labels.
    unconditional = evaluate_export(source, directory / "unconditional")
    packets = load_views(source)
    evaluator = json.loads((source / "evaluator_only.json").read_text())
    combined = packets["combined"]
    modes = MODES if include_count_control else MODES[:2]
    results = []
    all_ids = set(evaluator["histories"])
    for mode in modes:
        pairs, coverage = select_pairs(combined, evaluator, mode)
        destination = directory / mode
        write_json(destination / "coverage.json", coverage)
        write_json(destination / "pairing_evaluator_only.json", {"mode": mode, "pairs": pairs})
        selected = {hid for pair in pairs for hid in pair["source_history_ids"]}
        unmatched = subset_reports(packets, directory / "unconditional/predictions", evaluator, all_ids - selected)
        selected_reports = subset_reports(packets, directory / "unconditional/predictions", evaluator, selected)
        conditional = None
        if pairs:
            _, private = write_matched_export(combined, evaluator, pairs, mode, destination / "export")
            conditional = evaluate_export(destination / "export", destination / "attacks")
            verify_conditional_predictions(directory / "unconditional/predictions",
                                           destination / "attacks/predictions", private, mode)
            if mode != "result_counts_control":
                reference_views = {"ids"} if mode == "unordered_ids" else {"ids", "ranked_ids"}
                for row in conditional["reports"]:
                    if row["view"] in reference_views and row["accuracy"] != .5:
                        raise ValueError("Exact matched-log baseline must score 50% on balanced pairs")
        result = {"coverage": coverage, "conditional_reports": conditional["reports"] if conditional else [],
                  "conditional_paired_comparisons": conditional["paired_comparisons"] if conditional else [],
                  "selected_unconditional_reports": selected_reports, "unmatched_reports": unmatched,
                  "prediction_restriction_check": "passed" if pairs else "not_applicable"}
        write_json(destination / "analysis.json", result)
        results.append(result)
    report = {"evaluation_role": "development_only", "source": str(source.resolve()),
              "source_manifest_sha256": hashlib.sha256((source / "export_manifest.json").read_bytes()).hexdigest(),
              "history_length": combined["setup"]["history_length"],
              "known_labels": len(combined["known_labels"]), "unconditional_reports": unconditional["reports"],
              "selection": "Exact signatures and opposite-side membership only; no score/attack-success filtering",
              "limits": "Conditional development diagnostic, not fresh held-out evidence or a primitive-security break. "
                        "Disjoint histories can still reuse source questions. Count-only matching does not hold IDs fixed.",
              "modes": results}
    write_json(directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--include-count-control", action="store_true")
    args = parser.parse_args()
    result = analyze_export(args.source, args.directory, args.include_count_control)
    for entry in result["modes"]:
        c = entry["coverage"]
        print(f"{c['mode']}: {c['disjoint_pairs']} disjoint pairs, "
              f"{c['matched_histories']}/{c['source_histories']} histories ({c['matched_fraction']:.1%})")


if __name__ == "__main__":
    main()
