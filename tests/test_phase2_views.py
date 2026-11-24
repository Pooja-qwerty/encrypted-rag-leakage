import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import phase2_export
from ciphertext_math import rank_scores, score_vectors
from evaluate_project import load_workload
from phase2_views import VIEWS, project_view, replay, validate_view


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name) / "export"
        workload = load_workload()
        workload["context"] = [workload["context"][0]]  # Guaranteed repeated text.
        cls.captured_queries = {}
        cls.captured_index = []
        encrypt_query = phase2_export.encrypt_query_vectors
        encrypt_index = phase2_export.encrypt_index

        async def capture_query(*args, **kwargs):
            result = await encrypt_query(*args, **kwargs)
            cls.captured_queries[kwargs["query_id"]] = list(result[0].encrypted_vector)
            return result

        async def capture_index(*args, **kwargs):
            result = await encrypt_index(*args, **kwargs)
            cls.captured_index.extend([list(record.encrypted_vector) for record in result])
            return result

        with patch("phase2_export.load_workload", return_value=workload), \
             patch("phase2_export.encrypt_query_vectors", side_effect=capture_query) as query_calls, \
             patch("phase2_export.encrypt_index", side_effect=capture_index) as index_calls, \
             patch("phase2_export.secrets.randbelow", side_effect=[0, 1, 0, 1]):
            cls.manifest = phase2_export.export_development(
                cls.directory, context_count=3, history_count=4, history_length=5,
                key_bytes=bytes([13]) * 32)
            cls.query_call_count = query_calls.call_count
            cls.index_call_count = index_calls.call_count
        cls.views = {name: json.loads((cls.directory / "views" / name / "attacker_view.json").read_text())
                     for name in VIEWS}
        cls.private = json.loads((cls.directory / "evaluator_only.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_raw_exports_are_actual_sdk_ciphertexts(self):
        combined = self.views["combined"]
        ids = combined["setup"]["record_ids"]
        self.assertEqual([combined["encrypted_index"][cid] for cid in ids], self.captured_index)
        for event in combined["context"] + [e for h in combined["histories"] for e in h["events"]]:
            self.assertEqual(event["encrypted_query"], self.captured_queries[event["event_id"]])
        self.assertEqual(self.index_call_count, 1)
        self.assertEqual(self.query_call_count, 23)

    def test_repeats_get_new_ids_and_new_sdk_calls(self):
        context = self.views["combined"]["context"]
        self.assertEqual(len({e["event_id"] for e in context}), 3)
        self.assertEqual(len({self.private["events"][e["event_id"]]["source_query_id"] for e in context}), 1)
        self.assertTrue(all(e["event_id"] in self.captured_queries for e in context))

    def test_views_have_matched_inputs_and_no_private_fields(self):
        combined = self.views["combined"]
        event_ids = replay(combined)["events"].keys()
        for view in self.views.values():
            validate_view(view)
            self.assertEqual(view["setup"], combined["setup"])
            self.assertEqual(view["known_labels"], combined["known_labels"])
            self.assertEqual(replay(view)["events"].keys(), event_ids)
            text = json.dumps(view)
            for forbidden in ["source_query_id", "challenge_bit", "majority_topic", "workload_seed",
                              "relevant_ids", "plain_topk", "plaintext_vector", "CLOAK_VECTOR_SECRET"]:
                self.assertNotIn(forbidden, text)
            self.assertNotIn((bytes([13]) * 32).hex(), text)
        self.assertEqual(self.private["evaluation_role"], "development_only")

    def test_history_shapes_and_challenge_labels_stay_private(self):
        self.assertEqual([h["challenge_bit"] for h in self.private["histories"].values()], [0, 1, 0, 1])
        for history in self.views["combined"]["histories"]:
            truth = self.private["histories"][history["history_id"]]
            labels = [self.private["events"][e["event_id"]]["query_topic"] for e in history["events"]]
            self.assertEqual(labels.count(truth["majority_topic"]), 4)
            self.assertEqual(len({e["event_id"] for e in history["events"]}), 5)

    def test_scores_and_rankings_replay_without_evaluator(self):
        reconstructed = replay(self.views["ciphertexts"])["events"]
        explicit = replay(self.views["combined"])["events"]
        self.assertEqual(reconstructed, explicit)
        for name in ("volume", "ids", "ranked_ids", "scores"):
            result = replay(self.views[name])
            self.assertEqual(result["summary"]["recomputed_events"], 0)
            self.assertEqual(result["summary"]["scope"], "schema_validation_only")

    def test_extra_fields_rejected_at_every_boundary(self):
        for mutate in [lambda v: v.update({"private": {}}),
                       lambda v: v["setup"].update({"seed": 42}),
                       lambda v: v["context"][0].update({"topic": "CS"}),
                       lambda v: v["histories"][0].update({"challenge_bit": 0}),
                       lambda v: v["histories"][0]["events"][0].update({"plaintext_vector": []}),
                       lambda v: v["encrypted_index"].update({"secret_key": []})]:
            bad = copy.deepcopy(self.views["combined"])
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_view(bad)
        bad = copy.deepcopy(self.views["volume"])
        bad["encrypted_index"] = self.views["combined"]["encrypted_index"]
        with self.assertRaises(ValueError):
            validate_view(bad)

    def test_tampering_shapes_and_scores_rejected(self):
        for mutate in [lambda v: v["context"][0]["all_scores"].__setitem__(0, 42.),
                       lambda v: v.update({"schema_version": 2}),
                       lambda v: v["setup"].update({"metric": []}),
                       lambda v: v["context"][0]["result_ids"].reverse(),
                       lambda v: v["context"][0]["encrypted_query"].pop(),
                       lambda v: v["context"][0]["encrypted_query"].__setitem__(0, float("nan")),
                       lambda v: v["context"][1].update({"event_id": v["context"][0]["event_id"]})]:
            bad = copy.deepcopy(self.views["combined"])
            mutate(bad)
            with self.assertRaises(ValueError):
                replay(bad)

    def test_isolated_cli_needs_only_view_and_public_math(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = Path(__file__).resolve().parents[1]
            for filename in ("phase2_views.py", "ciphertext_math.py"):
                shutil.copy(repo / filename, root / filename)
            (root / "view.json").write_text(json.dumps(self.views["ciphertexts"]))
            isolated = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'ironcore_alloy', 'cloak', 'sklearn', 'game3', 'phase2_export', 'evaluate_project'}:
        raise ImportError('Private/exporter dependency unavailable in replay')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = ['phase2_views.py', 'view.json', '--output', 'replay.json']
runpy.run_path('phase2_views.py', run_name='__main__')
"""
            subprocess.run([sys.executable, "-c", isolated],
                           cwd=root, check=True, capture_output=True, text=True)
            result = json.loads((root / "replay.json").read_text())
            self.assertEqual(result, replay(self.views["ciphertexts"]))
            self.assertFalse((root / "evaluator_only.json").exists())

    def test_search_arithmetic_and_public_tie_rule(self):
        index, query = [[2., 0.], [0., 3.], [-4., 0.]], [1., 0.]
        for metric, expected in [("cosine", [1., 0., -1.]), ("dot_product", [2., 0., -4.]),
                                  ("l2", [-1., -10., -25.])]:
            np.testing.assert_allclose(score_vectors(index, query, metric), expected)
        self.assertEqual(rank_scores([0., 0., 0.], ["z", "a", "m"], 2), [1, 2])

    def test_refuse_to_replace_saved_randomized_export(self):
        with self.assertRaises(ValueError):
            phase2_export.export_development(self.directory, key_bytes=bytes([13]) * 32)


if __name__ == "__main__":
    unittest.main()
