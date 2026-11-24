"""Run fixed Phase 2 attacks first, then score with separate evaluator labels."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from phase2_attack import ATTACK_VERSION, METHODS, PARAMETERS, run_attack
from phase2_views import VIEWS, project_view, replay

CONTROLS = ("observed_labels", "zero_labels", "swapped_labels")
COMPARISONS = (
    ("rank_over_ids", ("ranked_ids", "known_only"), ("ids", "known_only")),
    ("scores_over_rank_known", ("scores", "score_known"), ("ranked_ids", "known_only")),
    ("scores_over_rank_cluster", ("scores", "score_cluster"), ("ranked_ids", "co_retrieval")),
    ("anchors_over_ids", ("ciphertexts", "anchor_geometry"), ("ids", "known_only")),
    ("graph_over_id_cluster", ("ciphertexts", "geometry"), ("ids", "co_retrieval")),
    ("combined_over_score_cluster", ("combined", "combined"), ("scores", "score_cluster")),
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def control_view(packet, control):
    if control not in CONTROLS:
        raise ValueError("Unsupported control")
    result = copy.deepcopy(packet)
    if control == "zero_labels":
        result["known_labels"] = {}
    elif control == "swapped_labels":
        first, second = result["setup"]["topic_pair"]
        result["known_labels"] = {cid: second if topic == first else first
                                  for cid, topic in result["known_labels"].items()}
    return result


def load_views(source):
    source = Path(source)
    manifest = json.loads((source / "export_manifest.json").read_text())
    if manifest.get("evaluation_role") != "development_only":
        raise ValueError("This runner currently evaluates development exports only")
    entries = {entry["view"]: entry for entry in manifest["views"]}
    if set(entries) != set(VIEWS):
        raise ValueError("Expected the six matched Phase 2 views")
    packets = {}
    for name in VIEWS:
        expected_path = f"views/{name}/attacker_view.json"
        if entries[name]["file"] != expected_path:
            raise ValueError("Unexpected view path in manifest")
        path = source / expected_path
        if hashlib.sha256(path.read_bytes()).hexdigest() != entries[name]["sha256"]:
            raise ValueError("View differs from its export manifest")
        packets[name] = json.loads(path.read_text())
    combined = packets["combined"]
    replay(combined)
    for name, packet in packets.items():
        if packet != project_view(combined, name):
            raise ValueError("Views do not project from the same trace and auxiliary knowledge")
    return packets


def score_predictions(packet, predictions, evaluator):
    truth = evaluator["histories"]
    output = predictions["history_predictions"]
    expected_ids = {h["history_id"] for h in packet["histories"]}
    if set(truth) != expected_ids or set(output) != expected_ids:
        raise ValueError("Predictions and labels must match all challenge histories")
    if any(type(t["challenge_bit"]) is not int or t["challenge_bit"] not in (0, 1) for t in truth.values()):
        raise ValueError("Invalid evaluator challenge bits")
    if any(type(p["guess"]) is not int or p["guess"] not in (0, 1) for p in output.values()):
        raise ValueError("Invalid predicted bits")
    n = len(truth)
    correct = {hid: int(output[hid]["guess"] == truth[hid]["challenge_bit"]) for hid in truth}
    by_side = {side: [hid for hid in truth if truth[hid]["challenge_bit"] == side] for side in (0, 1)}
    if by_side[0] and by_side[1]:
        p0, p1 = [float(np.mean([output[hid]["guess"] for hid in by_side[side]])) for side in (0, 1)]
        delta = p1 - p0
        balanced = (1 + delta) / 2
    else:
        p0 = p1 = delta = balanced = None
    return {"view": packet["view"], "method": predictions["method"],
            "histories": n, "history_length": packet["setup"]["history_length"],
            "known_labels": len(packet["known_labels"]), "topic0_count": len(by_side[0]),
            "topic1_count": len(by_side[1]), "accuracy": float(np.mean(list(correct.values()))),
            "balanced_accuracy": balanced, "delta": delta, "advantage": abs(delta) if delta is not None else None,
            "right_guess_rate": p1, "left_guess_rate": p0,
            "fallback_rate": float(np.mean([not output[h]["has_signal"] for h in truth])),
            "chance_baseline": .5, "realized_majority_baseline": max(len(by_side[0]), len(by_side[1])) / n,
            "fit_seconds": predictions["fit_seconds"], "predict_seconds": predictions["predict_seconds"]}


def paired_comparisons(predictions, reports, evaluator):
    results = []
    for control in CONTROLS:
        for name, candidate, reference in COMPARISONS:
            ckey, rkey = (control, *candidate), (control, *reference)
            c = predictions[ckey]["history_predictions"]
            r = predictions[rkey]["history_predictions"]
            wins = losses = 0
            for hid, truth in evaluator["histories"].items():
                cc, rc = c[hid]["guess"] == truth["challenge_bit"], r[hid]["guess"] == truth["challenge_bit"]
                wins += int(cc and not rc)
                losses += int(rc and not cc)
            cb, rb = reports[ckey]["balanced_accuracy"], reports[rkey]["balanced_accuracy"]
            results.append({"control": control, "comparison": name,
                            "candidate": list(candidate), "reference": list(reference),
                            "histories": len(c), "wins": wins, "losses": losses,
                            "paired_accuracy_gain": (wins - losses) / len(c),
                            "balanced_accuracy_gain": cb - rb if cb is not None and rb is not None else None})
    return results


def evaluate_export(source, directory):
    source, directory = Path(source), Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty result directory to preserve previous evaluations")
    packets = load_views(source)
    predictions, controlled_packets = {}, {}
    for control in CONTROLS:
        for view in VIEWS:
            packet = control_view(packets[view], control)
            controlled_packets[control, view] = packet
            for method in METHODS[view]:
                key = (control, view, method)
                predictions[key] = run_attack(packet, method)
                write_json(directory / "predictions" / control / view / f"{method}.json", predictions[key])
    # All fitting and predictions are frozen before opening the truth file.
    evaluator = json.loads((source / "evaluator_only.json").read_text())
    if evaluator.get("evaluation_role") != "development_only":
        raise ValueError("Expected separate development evaluator labels")
    reports = {key: score_predictions(controlled_packets[key[:2]], prediction, evaluator)
               for key, prediction in predictions.items()}
    rows = [{"control": key[0], **report} for key, report in reports.items()]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"attack_version": ATTACK_VERSION, "evaluation_role": "development_only",
              "evaluation_design": evaluator.get("evaluation_design", "unconditional_development"),
              "source": str(source.resolve()), "parameters": PARAMETERS,
              "source_manifest_sha256": hashlib.sha256((source / "export_manifest.json").read_bytes()).hexdigest(),
              "known_labels": len(packets["ids"]["known_labels"]),
              "history_length": packets["ids"]["setup"]["history_length"],
              "histories": packets["ids"]["setup"]["history_count"],
              "controls": {"observed_labels": "Unchanged disclosed map",
                           "zero_labels": "Remove all known labels",
                           "swapped_labels": "Swap topic names on every known record; an orientation diagnostic, not a null p-value"},
              "limits": "Small development exports with repeated source questions, one key per export, "
                        "and few known labels. No final-test security claim. Any matched-ID "
                        "conditioning must be reported by the dedicated matching analysis. "
                        "Scores and rankings are derivable from ciphertexts.",
              "reports": rows, "paired_comparisons": paired_comparisons(predictions, reports, evaluator)}
    write_json(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = evaluate_export(args.source, args.directory)
    print("Development only: view / method: balanced accuracy / signed delta / fallback")
    for row in result["reports"]:
        if row["control"] == "observed_labels":
            print(f"{row['view']} / {row['method']}: {row['balanced_accuracy']} / {row['delta']} / {row['fallback_rate']:.3f}")


if __name__ == "__main__":
    main()
