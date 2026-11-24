import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluate_project import (load_workload, make_sessions, project_view, relevance_scores,
                              select_known, session_scores, study)
from leakage_experiment import CHALLENGES, prepare
from log_attack import run_attack, run_session_attack


class ProjectEvaluationTests(unittest.TestCase):
    def test_test_questions_and_qrels_are_separate(self):
        workload = load_workload()
        self.assertEqual(len(workload["challenges"]), 48)
        self.assertEqual(len(workload["context"]), 24)
        self.assertFalse({q["text"] for q in workload["challenges"]} & {q for _, q in CHALLENGES})
        with tempfile.TemporaryDirectory() as directory:
            view, private = prepare(directory, workload=workload)
            self.assertEqual(len(view["challenges"]), 48)
            self.assertEqual(set(view), {"level", "known_labels", "context", "challenges"})
            self.assertEqual(len(private["retrieval_utility"]), 48)
            self.assertTrue(all(row["relevant_ids"] for row in private["retrieval_utility"].values()))

    def test_known_only_does_not_spread_labels(self):
        view = {"level": "AP-set", "known_labels": {"a": "Bio"},
                "context": [{"chunk_ids": ["a", "b"], "num_results": 2}],
                "challenges": [{"query_id": "q", "chunk_ids": ["b"], "num_results": 1}]}
        direct = run_attack(view, "known_only")
        self.assertEqual(direct["chunk_topics"], {"a": "Bio"})
        self.assertFalse(direct["predictions"]["q"]["has_signal"])

    def test_label_budgets_are_nested_and_repeatable(self):
        truth = {f"x{i}": "CS" if i < 6 else "Bio" for i in range(12)}
        one, four = select_known(truth, 1, 2), select_known(truth, 4, 2)
        self.assertEqual(len(one), 2)
        self.assertEqual(len(four), 8)
        self.assertLessEqual(one.items(), four.items())
        self.assertEqual(four, select_known(truth, 4, 2))

    def test_log_views_really_hide_ids_or_order(self):
        base = {"context": [{"num_results": 2, "chunk_ids": ["z", "a"]}],
                "challenges": [{"query_id": "x", "num_results": 2, "chunk_ids": ["z", "a"]}]}
        volume = project_view(base, {}, "Volume")
        self.assertNotIn("chunk_ids", volume["context"][0])
        self.assertEqual(project_view(base, {}, "AP-set")["context"][0]["chunk_ids"], ["a", "z"])
        self.assertEqual(base["context"][0]["chunk_ids"], ["z", "a"])

    def test_relevance_metrics_use_rank_and_judged_ids(self):
        private = {"retrieval_utility": {"q": {"relevant_ids": ["a", "b"],
                                                   "backend_topk": ["x", "a", "b"]}}}
        scores = relevance_scores(private)
        self.assertAlmostEqual(scores["precision_at_3"], 2 / 3)
        expected = (1 / math.log2(3) + .5) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(scores["ndcg_at_3"], expected)

    def test_sessions_have_balanced_majorities_without_leaking_labels(self):
        truth = {f"q{i}": "CS" if i < 5 else "Bio" for i in range(10)}
        sessions, labels = make_sessions(truth, 12, 3)
        self.assertEqual(list(labels.values()).count("CS"), 3)
        self.assertEqual(list(labels.values()).count("Bio"), 3)
        view = {"level": "AP-set", "context": [], "known_labels": {"a": "CS", "b": "Bio"},
                "challenges": [{"query_id": q, "num_results": 1,
                                "chunk_ids": ["a" if t == "CS" else "b"]} for q, t in truth.items()]}
        for session in sessions["sessions"]:
            self.assertEqual(len(set(session["query_ids"])), 5)
            self.assertEqual(sum(truth[q] == labels[session["session_id"]]
                                 for q in session["query_ids"]), 4)
        predictions = run_session_attack(view, sessions, view["known_labels"])
        self.assertEqual(session_scores(predictions, labels)["session_accuracy"], 1.)
        sessions["sessions"][0]["topic"] = "CS"
        with self.assertRaises(ValueError):
            run_session_attack(view, sessions, view["known_labels"])

    def test_small_end_to_end_study_and_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            report = study(directory, factors=[1.5], keys=1, label_seeds=2,
                           budgets=[0, 1, 4], sessions_per_topic=2, root_key=bytes([7]) * 32)
            summaries = report["summaries"]
            self.assertEqual(len(summaries), 2 * 3 * 3 * 2)
            self.assertEqual(len(report["clustering_gains"]), 2 * 3 * 3)
            for gain in report["clustering_gains"]:
                if gain["budget"] == 0:
                    self.assertEqual(gain["query_accuracy_gain_mean"], 0)
                    self.assertEqual(gain["session_accuracy_gain_mean"], 0)
            for row in summaries:
                self.assertEqual(row["runs"], 2)
                if row["level"] == "Volume" or row["budget"] == 0:
                    self.assertEqual(row["query_accuracy_mean"], .5)
                    self.assertEqual(row["session_accuracy_mean"], .5)
            plain = json.loads((Path(directory) / "plaintext/attacker_view.json").read_text())
            cloaked = json.loads((Path(directory) / "cloaked_1.5_key_00/attacker_view.json").read_text())
            self.assertEqual([e["query_id"] for e in plain["challenges"]],
                             [e["query_id"] for e in cloaked["challenges"]])


if __name__ == "__main__":
    unittest.main()
