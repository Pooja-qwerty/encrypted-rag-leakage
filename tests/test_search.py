import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cloak.retrieval import top_k_cloaked
from cloak.sdk import build_standalone_sdk
from sweep_experiment import derive_key, summarize, sweep, validate_metrics


def encrypted(values):
    return SimpleNamespace(encrypted_vector=values)


class SearchTests(unittest.TestCase):
    def test_ciphertext_norms_change_dot_rank_but_not_cosine(self):
        index = [encrypted([10., 10.]), encrypted([1., 0.])]
        query = [encrypted([1., 0.])]
        self.assertEqual(top_k_cloaked(index, query, 1, "cosine"), [1])
        self.assertEqual(top_k_cloaked(index, query, 1, "dot_product"), [0])
        self.assertEqual(top_k_cloaked(index, query, 1, "l2"), [1])

    def test_unknown_metrics_and_multiple_keys_are_rejected(self):
        index, query = [encrypted([1., 0.])], [encrypted([1., 0.])]
        for queries, k, metric in [(query * 2, 1, "cosine"), (query, 0, "cosine"),
                                    (query, 1, "typo")]:
            with self.assertRaises(ValueError):
                top_k_cloaked(index, queries, k, metric)
        self.assertEqual(top_k_cloaked(index, query, 100), [0])

    def test_known_neighbor_control(self):
        report = asyncio.run(validate_metrics(bytes([5]) * 32, 3))
        self.assertEqual(len(report["checks"]), 9)
        self.assertTrue(all(row["passed"] for row in report["checks"]))

    def test_invalid_factors_fail_before_sdk_use(self):
        for factor in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                build_standalone_sdk(bytes([5]) * 32, factor)

    def test_sweep_records_runs_without_exporting_keys(self):
        root_key = bytes([5]) * 32
        self.assertNotEqual(derive_key(root_key, 0), derive_key(root_key, 1))
        self.assertEqual(derive_key(root_key, 0), derive_key(root_key, 0))
        with tempfile.TemporaryDirectory() as directory:
            report = sweep(directory, factors=[1.0], key_count=2, context_count=8,
                           root_key=root_key)
            self.assertEqual(report["factors"][0]["keys"], 2)
            self.assertEqual(report["plaintext"]["mean_overlap_at_3"], 1)
            for path in Path(directory).rglob("*.json"):
                contents = path.read_text()
                self.assertNotIn(root_key.hex(), contents)
                self.assertNotIn(derive_key(root_key, 0).hex(), contents)
            saved = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(saved, report)

    def test_summary_uses_per_run_spread(self):
        rows = [{"approximation_factor": 1, "query_accuracy": x, "advantage": 0.,
                 "mean_overlap_at_3": 1., "query_signal_fraction": 1.} for x in [.5, 1.]]
        summary = summarize(rows)[0]
        self.assertEqual(summary["query_accuracy_mean"], .75)
        self.assertEqual(summary["query_accuracy_min"], .5)
        self.assertEqual(summary["query_accuracy_max"], 1.)


if __name__ == "__main__":
    unittest.main()
