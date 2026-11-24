import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from leakage_experiment import compare, evaluate, prepare
from log_attack import infer_chunk_topics, run_attack


class LogAttackTests(unittest.TestCase):
    def test_clusters_use_only_disclosed_labels(self):
        context = [{"chunk_ids": ["a", "b"], "num_results": 2},
                   {"chunk_ids": ["c", "d"], "num_results": 2}]
        inferred = infer_chunk_topics(context, {"a": "CS", "c": "Bio"})
        self.assertEqual(inferred, {"a": "CS", "b": "CS", "c": "Bio", "d": "Bio"})
        self.assertEqual(infer_chunk_topics(context, {}), {})
        self.assertEqual(infer_chunk_topics([], {"a": "CS"}), {"a": "CS"})

    def test_unknowns_and_volume_use_explicit_fallback(self):
        view = {"level": "Volume", "context": [{"num_results": 3}],
                "challenges": [{"query_id": "x", "num_results": 3}],
                "known_labels": {"a": "Bio"}}
        result = run_attack(view)
        self.assertEqual(result["predictions"]["x"], {"topic": "CS", "has_signal": False})

    def test_reject_private_fields(self):
        view = {"level": "AP-set", "context": [], "challenges": [], "known_labels": {}}
        for altered in [dict(view, vectors=[]),
                        dict(view, context=[{"num_results": 0, "chunk_ids": [], "topic": "CS"}])]:
            with self.assertRaises(ValueError):
                run_attack(altered)

    def test_challenges_do_not_train_the_clusters(self):
        view = {"level": "AP-set", "known_labels": {"a": "CS", "c": "Bio"},
                "context": [{"chunk_ids": ["a", "b"], "num_results": 2}],
                "challenges": [{"query_id": "x", "chunk_ids": ["c", "d"],
                                "num_results": 2}]}
        result = run_attack(view)
        self.assertNotIn("d", result["chunk_topics"])
        self.assertEqual(result["predictions"]["x"]["topic"], "Bio")

    def test_anchor_budget_does_not_change_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            zero, truth = prepare(directory, known_per_topic=0)
            two, _ = prepare(directory, known_per_topic=2)
            self.assertEqual(zero["context"], two["context"])
            self.assertEqual(zero["challenges"], two["challenges"])
            self.assertEqual(len(two["known_labels"]), 4)
            for cid, label in two["known_labels"].items():
                self.assertEqual(truth["chunk_topics"][cid], label)
            self.assertTrue(all(cid.startswith("c_") for cid in truth["chunk_topics"]))
            self.assertEqual(two, prepare(directory, known_per_topic=2)[0])

    def test_attacker_replays_without_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            view, private = prepare(root / "generator")
            isolated = root / "attacker"
            isolated.mkdir()
            script = Path(__file__).resolve().parents[1] / "log_attack.py"
            shutil.copy(script, isolated / "log_attack.py")
            (isolated / "view.json").write_text(json.dumps(view))
            subprocess.run([sys.executable, "log_attack.py", "view.json", "predictions.json"],
                           cwd=isolated, check=True, capture_output=True, text=True)
            predictions = json.loads((isolated / "predictions.json").read_text())
            self.assertEqual(predictions, run_attack(view))
            report = evaluate(view, private, predictions)
            self.assertEqual(report["queries"], 8)
            self.assertEqual(report["unknown_chunks"], 22)
            self.assertEqual(report["majority_baseline"], 0.5)
            # Truth is used only for scoring; changing it cannot alter saved predictions.
            changed = copy.deepcopy(private)
            changed["query_topics"] = {q: "CS" if t == "Bio" else "Bio"
                                       for q, t in private["query_topics"].items()}
            flipped = evaluate(view, changed, predictions)
            self.assertAlmostEqual(flipped["query_accuracy"], 1 - report["query_accuracy"])

    def test_volume_and_zero_anchor_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            for level, budget in [("Volume", 1), ("AP-set", 0)]:
                view, private = prepare(directory, level=level, known_per_topic=budget)
                report = evaluate(view, private, run_attack(view))
                self.assertEqual(report["advantage"], 0)
                self.assertEqual(report["query_accuracy"], 0.5)
                self.assertEqual(report["query_signal_fraction"], 0)
                self.assertEqual(report["inferred_unknown_chunks"], 0)

    def test_cloaked_comparison_keeps_the_same_inputs(self):
        from cloak.sdk import build_standalone_sdk

        def test_sdk(*, approximation_factor):
            # Nonsecret test key; never read the user's .env in this test.
            return build_standalone_sdk(bytes([3]) * 32, approximation_factor)

        with tempfile.TemporaryDirectory() as directory:
            with patch("cloak.sdk.build_standalone_sdk", side_effect=test_sdk) as sdk:
                reports = compare(directory, context_count=8, approximation_factor=1.5)
            sdk.assert_called_once_with(approximation_factor=1.5)
            root = Path(directory)
            def read(backend, filename):
                return json.loads((root / backend / filename).read_text())
            plain = read("plaintext", "attacker_view.json")
            encrypted = read("cloaked", "attacker_view.json")
            self.assertEqual(plain["known_labels"], encrypted["known_labels"])
            self.assertEqual([e["query_id"] for e in plain["challenges"]],
                             [e["query_id"] for e in encrypted["challenges"]])
            p_truth = read("plaintext", "evaluator_only.json")
            c_truth = read("cloaked", "evaluator_only.json")
            self.assertEqual(p_truth["chunk_topics"], c_truth["chunk_topics"])
            self.assertEqual(p_truth["query_topics"], c_truth["query_topics"])
            self.assertEqual(c_truth["run"]["approximation_factor"], 1.5)
            self.assertEqual(reports[0]["mean_overlap_at_3"], 1)
            for entry in encrypted["challenges"]:
                qid = entry["query_id"]
                utility = c_truth["retrieval_utility"][qid]
                self.assertEqual(set(entry["chunk_ids"]), set(utility["backend_topk"]))
                expected_overlap = len(set(utility["plain_topk"]) & set(utility["backend_topk"])) / 3
                self.assertEqual(utility["overlap_at_3"], expected_overlap)
            saved = read("cloaked", "predictions.json")
            self.assertEqual(saved, run_attack(encrypted))


if __name__ == "__main__":
    unittest.main()
