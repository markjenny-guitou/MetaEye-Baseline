"""Regression checks for metric definitions and the Windows clustering backend."""

import ast
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaeye.clustering import jaccard_distance
from metaeye.data import PseudoIdentityBatchSampler, audit_regdb
from metaeye.evaluation import retrieval_metrics
from metaeye.model import ROOT
from metaeye.artifacts import new_archive, retain_file, write_json, write_npz
from metaeye.evaluation import retrieval_results


class ExactIndex:
    def add(self, values):
        self.values = values

    def search(self, query, count):
        distance = 2 - 2 * query @ self.values.T
        order = np.argsort(distance, axis=1)[:, :count]
        return np.take_along_axis(distance, order, axis=1), order


class BaselineChecks(unittest.TestCase):
    def test_retrieval_metrics(self):
        result = retrieval_metrics([[0.9, 0.8, 0.1, 0], [0.9, 0.8, 0.7, 0.6]], [0, 1], [0, 1, 0, 1])
        self.assertAlmostEqual(result["rank1"], 0.5)
        self.assertAlmostEqual(result["mAP"], 2 / 3)
        self.assertAlmostEqual(result["mINP"], 7 / 12)

    def test_retrieval_evidence_matches_metrics(self):
        metrics, evidence = retrieval_results([[0.9, 0.8, 0.1, 0], [0.9, 0.8, 0.7, 0.6]], [0, 1], [0, 1, 0, 1])
        self.assertAlmostEqual(evidence["cmc"][0], metrics["rank1"])
        self.assertAlmostEqual(evidence["ap"].mean(), metrics["mAP"])
        self.assertAlmostEqual(evidence["inp"].mean(), metrics["mINP"])
        np.testing.assert_array_equal(evidence["first_positive_rank"], [1, 2])
        np.testing.assert_array_equal(evidence["last_positive_rank"], [3, 4])

    def test_archived_files_survive_replacement_and_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = new_archive(root, "epoch_0001")
            second = new_archive(root, "epoch_0001")
            self.assertNotEqual(first, second)
            latest = root / "checkpoint.pt"
            latest.write_bytes(b"original checkpoint")
            retain_file(latest, first / "checkpoint.pt")
            staging = root / "new.partial"
            staging.write_bytes(b"new checkpoint")
            staging.replace(latest)
            self.assertEqual((first / "checkpoint.pt").read_bytes(), b"original checkpoint")
            with self.assertRaises(FileExistsError):
                retain_file(latest, first / "checkpoint.pt")
            write_json(second / "summary.json", {"status": "completed"})
            write_npz(second / "clustering.npz", visible=np.array([0, -1, 2]))
            np.testing.assert_array_equal(np.load(second / "clustering.npz")["visible"], [0, -1, 2])

    def test_sampler_uses_only_valid_pseudo_labels(self):
        labels = np.array([-1, 0, 0, 1, 1, 2])
        sampler = PseudoIdentityBatchSampler(labels, batch_size=4, instances=2, iterations=4, seed=4)
        for batch in sampler:
            sampled = labels[batch]
            self.assertTrue(np.all(sampled >= 0))
            self.assertEqual(sorted(np.unique(sampled, return_counts=True)[1].tolist()), [2, 2])

    def test_jaccard_matches_upstream_reference(self):
        # Execute the two upstream mathematical functions with an exact CPU index.
        source = ROOT / "third_party/ADCA/clustercontrast/utils/faiss_rerank.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("k_reciprocal_neigh", "compute_jaccard_distance")]
        namespace = {"np": np, "torch": torch, "F": F, "time": time,
                     "faiss": SimpleNamespace(get_num_gpus=lambda: 0),
                     "index_init_cpu": lambda _: ExactIndex()}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        generator = torch.Generator().manual_seed(10)
        features = F.normalize(torch.randn(32, 16, generator=generator), dim=1)
        reference = namespace["compute_jaccard_distance"](features, k1=8, k2=3,
                                                         print_flag=False, search_option=3)
        actual = jaccard_distance(features, k1=8, k2=3)
        np.testing.assert_allclose(actual, reference, atol=1e-6)

    def test_regdb_all_ten_splits(self):
        # Train and test have separately reindexed labels, so leakage is checked by folder IDs.
        report = audit_regdb(ROOT / "data/RegDB")
        self.assertEqual(len(report), 10)
        self.assertTrue(all(row["train_images"] == 4120 for row in report))


if __name__ == "__main__":
    unittest.main()
