import csv
import json
import tempfile
import unittest
from pathlib import Path

from defense_experiment import (client_quality, filter_fetch_batch, full_download_view,
                                pad_fetches, run_defense_study)
from evaluate_project import load_workload, make_sessions
from leakage_experiment import prepare
from log_attack import run_attack, validate_view


class DefenseTests(unittest.TestCase):
    def base(self):
        return {"level": "AP-ordered", "known_labels": {},
                "context": [{"num_results": 3, "chunk_ids": ["b", "a", "c"]}],
                "challenges": [{"query_id": "q", "num_results": 3, "chunk_ids": ["c", "a", "b"]}]}

    def test_padding_is_unique_nested_and_keeps_all_real_results(self):
        base = self.base()
        small = pad_fetches(base, list("abcdef"), 1, 7)
        large = pad_fetches(base, list("abcdef"), 3, 7)
        self.assertEqual(small, pad_fetches(base, list("abcdef"), 1, 7))
        for section in ("context", "challenges"):
            self.assertEqual(small[section][0]["num_results"], 4)
            self.assertEqual(len(set(small[section][0]["chunk_ids"])), 4)
            self.assertLessEqual(set(base[section][0]["chunk_ids"]), set(small[section][0]["chunk_ids"]))
            self.assertLessEqual(set(small[section][0]["chunk_ids"]), set(large[section][0]["chunk_ids"]))
        self.assertEqual(base["context"][0]["chunk_ids"], ["b", "a", "c"])
        self.assertEqual(filter_fetch_batch(small["context"][0]["chunk_ids"], ["b", "a", "c"]),
                         ["b", "a", "c"])
        validate_view(small)

    def test_invalid_padding_and_missing_records_are_rejected(self):
        for count in [-1, 4]:
            with self.assertRaises(ValueError):
                pad_fetches(self.base(), list("abcdef"), count, 1)
        with self.assertRaises(ValueError):
            filter_fetch_batch(["a"], ["a", "b", "c"])

    def test_full_download_hides_result_identity_in_the_observed_view(self):
        base = self.base()
        base["challenges"].append({"query_id": "r", "num_results": 3, "chunk_ids": ["d", "e", "f"]})
        view = full_download_view(base, list("abcdef"))
        view["known_labels"] = {"a": "CS", "b": "CS", "c": "CS", "d": "Bio", "e": "Bio", "f": "Bio"}
        self.assertEqual(view["challenges"][0]["chunk_ids"], view["challenges"][1]["chunk_ids"])
        for method in ("known_only", "clustering"):
            predictions = run_attack(view, method)["predictions"]
            self.assertEqual(predictions["q"], predictions["r"])

    def test_client_filters_padding_and_full_download_uses_local_ranking(self):
        private = {"retrieval_utility": {"q": {"plain_topk": ["a", "b", "c"],
                         "backend_topk": ["c", "a", "b"], "overlap_at_3": 1}}}
        observed = pad_fetches(self.base(), list("abcdef"), 3, 1)
        padded = client_quality(private, observed)
        self.assertEqual(padded, private)
        local = client_quality(private, observed, full_download=True)
        self.assertEqual(local["retrieval_utility"]["q"]["backend_topk"], ["a", "b", "c"])
        self.assertEqual(private["retrieval_utility"]["q"]["backend_topk"], ["c", "a", "b"])

    def test_study_pairs_controls_costs_and_utility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "plaintext"
            _, private = prepare(source, workload=load_workload(), level="AP-ordered", known_per_topic=0)
            sessions, truth = make_sessions(private["query_topics"], 14, 2)
            private["session_topics"] = truth
            (source / "evaluator_only.json").write_text(json.dumps(private))
            (source / "session_view.json").write_text(json.dumps(sessions))
            result = run_defense_study(root / "source", root / "output", dummy_counts=[0, 3, 21],
                                      cover_seeds=1, label_seeds=2)
            with (root / "output/runs.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 24)
            baseline = next(r for r in rows if r["dummy_count"] == "0")
            for row in rows:
                self.assertEqual(float(row["fetch_records_per_query"]), 3 + int(row["dummy_count"]))
                self.assertEqual(float(row["fetch_multiplier"]), (3 + int(row["dummy_count"])) / 3)
                if row["observer"] == "search_result_control":
                    self.assertEqual(float(row["query_accuracy_change"]), 0)
                    self.assertEqual(float(row["session_accuracy_change"]), 0)
                if row["protocol"] == "server_search_plus_padding":
                    self.assertEqual(row["client_topk_preserved"], "True")
                    self.assertEqual(row["ndcg_at_3"], baseline["ndcg_at_3"])
                if row["observer"] == "server_no_query":
                    self.assertEqual(row["query_sent_to_server"], "False")
                    self.assertEqual(float(row["query_accuracy"]), .5)
                    self.assertEqual(float(row["session_accuracy"]), .5)
            self.assertTrue(result["summaries"])


if __name__ == "__main__":
    unittest.main()
