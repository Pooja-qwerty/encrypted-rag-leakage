import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ciphertext_math import rank_scores, score_vectors
from log_attack import infer_chunk_topics, predict_session
from phase2_attack import METHODS, run_attack
from phase2_evaluate import control_view, evaluate_export, score_predictions
from phase2_views import VIEWS, project_view


def fixture():
    ids = [f"c_{i:024x}" for i in range(6)]
    matrix = np.array([[1., 0.], [.95, .1], [.85, .15], [0., 1.], [.1, .95], [.15, .85]])

    def event(i, query):
        scores = score_vectors(matrix, query, "cosine")
        order = rank_scores(scores, ids, 2)
        return {"event_id": f"e_{i:024x}", "result_count": 2,
                "result_ids": [ids[j] for j in order], "result_scores": scores[order].tolist(),
                "all_scores": scores.tolist(), "encrypted_query": query}

    return {"schema_version": 1, "view": "combined",
            "setup": {"record_ids": ids, "vector_dimension": 2,
                      "embedding_algorithm": "tfidf-word-unigram-bigram-l2", "metric": "cosine",
                      "approximation_factor": 1.5, "k": 2, "topic_pair": ["CS", "Bio"],
                      "context_count": 4, "history_count": 2, "history_length": 1,
                      "majority_count": 1, "session_linkage": False,
                      "tie_breaker": "record_id_ascending", "score_rule_version": 1},
            "known_labels": {ids[2]: "CS", ids[5]: "Bio"},
            "encrypted_index": dict(zip(ids, matrix.tolist())),
            "context": [event(i, q) for i, q in enumerate([[1., 0.], [.85, .15], [0., 1.], [.15, .85]])],
            "histories": [{"history_id": f"h_{i:024x}", "events": [event(i + 4, q)]}
                          for i, q in enumerate([[1., 0.], [0., 1.]])]}


class Phase2AttackTests(unittest.TestCase):
    def test_log_baselines_preserve_phase1_history_voting(self):
        combined = fixture()
        ids = combined["setup"]["record_ids"]
        combined["known_labels"] = {ids[0]: "CS", ids[3]: "Bio"}
        candidates = [combined["histories"][0]["events"][0], combined["histories"][1]["events"][0]]
        events = []
        for i, side in enumerate([0, 1, 1, 0, 1]):
            event = copy.deepcopy(candidates[side])
            event["event_id"] = f"e_{100+i:024x}"
            events.append(event)
        combined["histories"] = [{"history_id": "h_" + "0" * 24, "events": events}]
        combined["setup"].update(history_count=1, history_length=5, majority_count=3, session_linkage=True)
        for view, level in [("ids", "AP-set"), ("ranked_ids", "AP-ordered")]:
            packet = project_view(combined, view)
            context = [{"num_results": e["result_count"], "chunk_ids": e["result_ids"]} for e in packet["context"]]
            log = [{"num_results": e["result_count"], "chunk_ids": e["result_ids"]}
                   for e in packet["histories"][0]["events"]]
            for method in ("known_only", "co_retrieval"):
                topics = (packet["known_labels"] if method == "known_only"
                          else infer_chunk_topics(context, packet["known_labels"]))
                expected = predict_session(log, topics, level)["topic"]
                result = run_attack(packet, method)
                actual = next(iter(result["history_predictions"].values()))["guess"]
                self.assertEqual(actual, int(expected == "Bio"))

    def test_geometry_uses_anchors_outside_returned_ids(self):
        combined = fixture()
        ids = project_view(combined, "ids")
        direct = run_attack(ids, "known_only")
        self.assertTrue(all(not p["has_signal"] for p in direct["history_predictions"].values()))
        for method in ("anchor_geometry", "geometry"):
            attack = run_attack(project_view(combined, "ciphertexts"), method)
            self.assertEqual([p["guess"] for p in attack["history_predictions"].values()], [0, 1])

    def test_zero_labels_give_no_semantic_signal_for_every_method(self):
        combined = fixture()
        for name in VIEWS:
            packet = control_view(project_view(combined, name), "zero_labels")
            for method in METHODS[name]:
                result = run_attack(packet, method)
                self.assertTrue(all(p["guess"] == 0 and not p["has_signal"]
                                    for p in result["history_predictions"].values()), (name, method))

    def test_ciphertext_and_combined_views_have_identical_predictions(self):
        combined = fixture()
        for method in METHODS["combined"]:
            first = run_attack(project_view(combined, "ciphertexts"), method)
            second = run_attack(combined, method)
            self.assertEqual(first["history_predictions"], second["history_predictions"])

    def test_methods_cannot_use_unprovided_observations(self):
        for name, method in [("volume", "known_only"), ("ids", "score_known"), ("scores", "geometry")]:
            with self.assertRaises(ValueError):
                run_attack(project_view(fixture(), name), method)
        packet = project_view(fixture(), "ids")
        packet["histories"][0]["challenge_bit"] = 0
        with self.assertRaises(ValueError):
            run_attack(packet, "known_only")

    def test_score_margins_can_change_decisions_at_the_same_ranked_ids(self):
        packet = project_view(fixture(), "scores")
        packet["setup"]["k"] = 3
        inventory = packet["setup"]["record_ids"]
        chosen = [inventory[2], inventory[3], inventory[5]]
        packet["known_labels"] = {chosen[0]: "CS", chosen[1]: "Bio", chosen[2]: "Bio"}
        for entry in packet["context"]:
            entry.update(result_count=3, result_ids=chosen, result_scores=[0., 0., 0.])
        for history, scores in zip(packet["histories"], [[1., -1., -1.], [1., 1., 1.]]):
            history["events"][0].update(result_count=3, result_ids=chosen, result_scores=scores)
        result = run_attack(packet, "score_known")
        self.assertEqual([p["guess"] for p in result["history_predictions"].values()], [0, 1])

    def test_challenge_histories_do_not_change_fit_or_other_predictions(self):
        packet = project_view(fixture(), "scores")
        first = run_attack(packet, "score_cluster")
        changed = copy.deepcopy(packet)
        changed["histories"][1]["events"][0]["result_scores"] = [0., 0.]
        second = run_attack(changed, "score_cluster")
        hid = packet["histories"][0]["history_id"]
        self.assertEqual(first["parameters"], second["parameters"])
        self.assertEqual(first["history_predictions"][hid], second["history_predictions"][hid])

    def test_balanced_scoring_keeps_fallbacks_and_handles_one_sided_draws(self):
        packet = project_view(fixture(), "volume")
        prediction = run_attack(packet, "constant")
        truth = {"histories": {h["history_id"]: {"challenge_bit": i} for i, h in enumerate(packet["histories"])}}
        report = score_predictions(packet, prediction, truth)
        self.assertEqual(report["balanced_accuracy"], .5)
        self.assertEqual(report["fallback_rate"], 1.)
        for h in truth["histories"].values():
            h["challenge_bit"] = 1
        report = score_predictions(packet, prediction, truth)
        self.assertEqual(report["accuracy"], 0.)
        self.assertIsNone(report["balanced_accuracy"])
        self.assertIsNone(report["advantage"])

    def test_evaluator_opens_truth_after_all_attack_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            combined = fixture()
            manifest = {"evaluation_role": "development_only", "views": []}
            for name in VIEWS:
                path = source / "views" / name / "attacker_view.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(project_view(combined, name)))
                manifest["views"].append({"view": name, "file": str(path.relative_to(source)),
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (source / "export_manifest.json").write_text(json.dumps(manifest))
            original_read = Path.read_text

            def read_after_predictions(path, *args, **kwargs):
                if path.name == "evaluator_only.json":
                    expected = sum(len(methods) for methods in METHODS.values()) * 3
                    self.assertEqual(len(list((output / "predictions").rglob("*.json"))), expected)
                return original_read(path, *args, **kwargs)

            truth = {"evaluation_role": "development_only", "histories": {
                h["history_id"]: {"challenge_bit": i} for i, h in enumerate(combined["histories"])}}
            (source / "evaluator_only.json").write_text(json.dumps(truth))
            with patch.object(Path, "read_text", read_after_predictions):
                result = evaluate_export(source, output)
            self.assertEqual(len(result["reports"]), 84)
            self.assertEqual(len(result["paired_comparisons"]), 18)

    def test_standalone_attacker_has_no_sdk_or_evaluator_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root, repo = Path(directory), Path(__file__).resolve().parents[1]
            for name in ("phase2_attack.py", "phase2_views.py", "ciphertext_math.py", "log_attack.py"):
                shutil.copy(repo / name, root / name)
            packet = project_view(fixture(), "ciphertexts")
            (root / "view.json").write_text(json.dumps(packet))
            isolated = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'ironcore_alloy', 'cloak', 'game3', 'phase2_export', 'phase2_evaluate', 'evaluate_project'}:
        raise ImportError('Private dependency unavailable to attacker')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = ['phase2_attack.py', 'view.json', 'predictions.json', '--method', 'geometry']
runpy.run_path('phase2_attack.py', run_name='__main__')
"""
            subprocess.run([sys.executable, "-c", isolated], cwd=root, check=True, capture_output=True, text=True)
            result = json.loads((root / "predictions.json").read_text())
            self.assertEqual(result["history_predictions"], run_attack(packet, "geometry")["history_predictions"])


if __name__ == "__main__":
    unittest.main()
