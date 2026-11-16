"""Formula and synthetic integration checks. Does not train a ReID model or use RegDB images."""

import copy
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaeye.model import ROOT, ReIDEncoder
from metaeye.rahsm import (RAHSMBank, associate_shared_labels, hard_positives,
                          iteration_lambda, progressive_lambda, validate_rahsm)
import metaeye.training as training


def configuration():
    config = json.loads((ROOT / "configs/rahsm_regdb_laptop.json").read_text(encoding="utf-8"))
    config["temperature"] = 0.7
    return config


def example_features():
    visible = torch.tensor([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [8., 0.], [8., 2.]])
    thermal = visible + torch.tensor([0.3, 4.])
    labels = np.array([0, 0, 1, 1, 2, 2])
    return {"visible": visible, "thermal": thermal}, {"visible": labels, "thermal": labels.copy()}


def example_bank(config=None):
    features, labels = example_features()
    return RAHSMBank.from_features(features, labels, config or configuration(), seed=3)


class FormulaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_chp_midpoint_and_gap_equations_7_8(self):
        centers = example_bank().centers.double()
        magnitude = 0.3
        generated = hard_positives(centers, magnitude)
        torch.testing.assert_close(generated.mean(dim=0), centers.mean(dim=0))
        torch.testing.assert_close(generated[0] - generated[1], (1 + 2 * magnitude) * (centers[0] - centers[1]))
        torch.testing.assert_close(hard_positives(centers, 0), centers)

    def test_means_are_not_implicitly_normalized(self):
        bank = example_bank()
        torch.testing.assert_close(bank.centers[0, 0], torch.tensor([0., 1.]))
        torch.testing.assert_close(bank.centers[1, 0], torch.tensor([0.3, 5.]))

    def test_pts_endpoints_and_monotonicity(self):
        values = [progressive_lambda(step, 100, 0.3) for step in range(101)]
        self.assertEqual(values[0], 0)
        self.assertAlmostEqual(values[-1], 0.3)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[50], 0.3 * math.log(1 + (math.e - 1) / 2))
        self.assertEqual(progressive_lambda(0, 100, 0.3, enabled=False), 0.3)
        with self.assertRaises(ValueError):
            progressive_lambda(1, 0, 0.3)

    def test_pts_phase_and_resume_index(self):
        config = configuration()
        self.assertEqual(iteration_lambda(config, 29, 400), 0)
        self.assertAlmostEqual(iteration_lambda(config, 30, 1), progressive_lambda(1, 8000, 0.3))
        self.assertAlmostEqual(iteration_lambda(config, 49, 400), 0.3)
        config["rahsm"]["pts"] = False
        self.assertEqual(iteration_lambda(config, 30, 1), 0.3)

    def test_khn_equation_9_and_self_exclusion(self):
        bank = example_bank()
        candidates, valid, neighbors = bank.negative_candidates(0.3)
        self.assertTrue(bool((neighbors != bank.identities[:, None]).all()))
        for index in range(len(bank.subcenters)):
            owner = int(bank.identities[index])
            modal = int(bank.modalities[index])
            neighbor = int(neighbors[index, 0])
            slot = int(bank.slots[index])
            expected = bank.subcenters[index] + 0.3 * (bank.centers[modal, neighbor] - bank.centers[modal, owner])
            torch.testing.assert_close(candidates[owner, slot], expected)
            distances = (bank.centers[modal] - bank.subcenters[index]).square().sum(dim=1)
            distances[owner] = float("inf")
            self.assertEqual(neighbor, int(distances.argmin()))
            self.assertTrue(bool(valid[owner, slot]))

    def test_small_and_duplicate_clusters(self):
        bank = example_bank()
        self.assertTrue(bool((bank.subcluster_counts == 2).all()))
        features, labels = example_features()
        features["visible"][1] = features["visible"][0]
        bank = RAHSMBank.from_features(features, labels, configuration(), seed=3)
        self.assertEqual(int(bank.subcluster_counts[0, 0]), 1)

    def test_multiple_neighbors_and_padding(self):
        config = configuration()
        config["rahsm"]["neighbors"] = 8
        bank = example_bank(config)
        self.assertEqual(bank.neighbor_count, 2)
        candidates, valid, neighbors = bank.negative_candidates(0)
        self.assertTrue(bool((neighbors != bank.identities[:, None]).all()))
        for index in range(len(bank.subcenters)):
            owner, slot = int(bank.identities[index]), int(bank.slots[index])
            for neighbor in range(2):
                torch.testing.assert_close(candidates[owner, slot * 2 + neighbor], bank.subcenters[index])
        self.assertEqual(int(valid.sum()), len(bank.subcenters) * 2)

    def test_padded_candidates_are_never_selected(self):
        features, labels = example_features()
        features["visible"][1] = features["visible"][0]
        bank = RAHSMBank.from_features(features, labels, configuration(), seed=3)
        negatives = bank.negative_candidates(0.2)
        _, details = bank.loss(torch.tensor([[0.1, 0.2], [1.4, 1.6]]), torch.tensor([0, 1]), 0, 0.2, negatives)
        chosen_valid = negatives[1][torch.arange(bank.class_count)[None], details["negative_slot"]]
        self.assertTrue(bool(chosen_valid.all()))

    def test_loss_matches_independent_equations_11_12(self):
        bank = example_bank().double()
        queries = torch.tensor([[0.2, 0.8], [2.3, 1.4], [7.6, 3.3]], dtype=torch.float64)
        targets = torch.tensor([0, 1, 2])
        magnitude = 0.21
        negatives = bank.negative_candidates(magnitude)
        candidates, valid, _ = negatives
        positive_bank = hard_positives(bank.centers, magnitude)
        oracle = []
        for query, label in zip(queries, targets):
            row = []
            for identity in range(bank.class_count):
                if identity == int(label):
                    choices = positive_bank[:, identity]
                    index = (choices - query).square().sum(dim=1).argmax()
                else:
                    choices = candidates[identity, valid[identity]]
                    index = (choices - query).square().sum(dim=1).argmin()
                row.append(query @ choices[index])
            oracle.append(torch.stack(row))
        expected = F.cross_entropy(torch.stack(oracle) / bank.temperature, targets)
        actual, _ = bank.loss(queries, targets, 0, magnitude, negatives)
        torch.testing.assert_close(actual, expected)

    def test_euclidean_selection_not_dot_product_selection(self):
        bank = example_bank().double()
        bank.centers[:, 0] = torch.tensor([[2., 0.], [1., 2.]], dtype=torch.float64)
        query = torch.tensor([[1., 0.]], dtype=torch.float64)
        _, details = bank.loss(query, torch.tensor([0]), 0, magnitude=0)
        self.assertEqual(int(details["positive_modality"][0]), 1)

    def test_query_gradients(self):
        bank = example_bank().double()
        queries = torch.tensor([[0.27, 0.83], [2.31, 1.47]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([0, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda value: bank.loss(value, targets, 0, 0.19)[0], (queries,)))

    def test_batch_mean_momentum_equations_3_4(self):
        bank = example_bank()
        original = bank.centers.clone()
        visible = torch.tensor([[10., 3.], [2., 9.]])
        thermal = torch.tensor([[4., 8.], [8., 2.]])
        labels = torch.tensor([0, 0])
        bank.update(visible, labels, thermal, labels)
        torch.testing.assert_close(bank.centers[0, 0], 0.2 * original[0, 0] + 0.8 * visible.mean(0))
        torch.testing.assert_close(bank.centers[1, 0], 0.2 * original[1, 0] + 0.8 * thermal.mean(0))
        torch.testing.assert_close(bank.centers[:, 1:], original[:, 1:])

    def test_all_module_switch_combinations(self):
        for chp, khn, pts in itertools.product((False, True), repeat=3):
            config = configuration()
            config["rahsm"].update(chp=chp, khn=khn, pts=pts)
            bank = example_bank(config)
            queries = torch.tensor([[0.2, 0.8], [2.3, 1.4]], requires_grad=True)
            targets = torch.tensor([0, 1])
            magnitude = progressive_lambda(20, 100, 0.3, pts)
            loss, _ = bank.loss(queries, targets, 0, magnitude)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(bool(torch.isfinite(queries.grad).all()))
            if not chp and not khn:
                expected = F.cross_entropy(queries @ bank.centers[0].T / bank.temperature, targets)
                torch.testing.assert_close(loss, expected)

    def test_association_handles_outliers_and_sparse_cluster_ids(self):
        features = {"visible": torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [-1., 0.]]),
                    "thermal": torch.tensor([[0., 1.], [0., 1.], [1., 0.], [1., 0.], [-1., 0.]])}
        labels = {"visible": np.array([10, 10, 22, 22, -1]), "thermal": np.array([8, 8, 3, 3, -1])}
        shared, metadata = associate_shared_labels(features, labels, 1, 1)
        np.testing.assert_array_equal(shared["visible"], [0, 0, 1, 1, -1])
        np.testing.assert_array_equal(shared["thermal"], [1, 1, 0, 0, -1])
        self.assertEqual(metadata["thermal_id_for_shared_id"], [3, 8])

    def test_unmatched_identities_fail_explicitly(self):
        features = {"visible": torch.tensor([[1., 0.], [1., 0.]]),
                    "thermal": torch.tensor([[1., 0.], [1., 0.]])}
        labels = {"visible": np.array([0, 0]), "thermal": np.array([0, 0])}
        with self.assertRaises(RuntimeError):
            associate_shared_labels(features, labels, 1, 1)

    def test_config_validation(self):
        config = configuration()
        validate_rahsm(config)
        config["rahsm"]["neighbors"] = 0
        with self.assertRaises(ValueError):
            validate_rahsm(config)


class SyntheticTrainingEntryChecks(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA backbone check")
    def test_real_backbone_backward_with_synthetic_images(self):
        config = configuration()
        model = ReIDEncoder(config, pretrained=False).cuda().train()
        before = model.backbone.patch_embed.proj.weight.detach().clone()
        images = torch.randn(6, 3, config["height"], config["width"], device="cuda")
        with torch.autocast("cuda"):
            embeddings = model(images)
        features = {"visible": embeddings[:3].detach().cpu(), "thermal": embeddings[3:].detach().cpu()}
        labels = {"visible": np.array([0, 1, 2]), "thermal": np.array([0, 1, 2])}
        bank = RAHSMBank.from_features(features, labels, config, seed=3).cuda()
        targets = torch.arange(3, device="cuda")
        negatives = bank.negative_candidates(0.15)
        loss = bank.loss(embeddings[:3], targets, 0, 0.15, negatives)[0]
        loss = loss + bank.loss(embeddings[3:], targets, 1, 0.15, negatives)[0]
        loss.backward()
        gradient = model.backbone.patch_embed.proj.weight.grad
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.norm()), 0)
        torch.testing.assert_close(before, model.backbone.patch_embed.proj.weight)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-only training entry")
    def test_phase_switch_archives_and_resume_with_synthetic_tensors(self):
        # Tiny 2-D tensors and a 2x2 linear layer exercise the real training entry.
        # ReID images, pretrained weights and actual experiment directories are unused.
        values = {"visible": torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [-1., 0.], [-1., 0.]]),
                  "thermal": torch.tensor([[1., 0.1], [1., 0.1], [0.1, 1.], [0.1, 1.], [-1., 0.1], [-1., 0.1]])}
        labels = np.array([0, 0, 1, 1, 2, 2])
        records = {modal: [SimpleNamespace(path=Path("synthetic") / modal / f"{i}.png") for i in range(6)]
                   for modal in values}

        class Encoder(nn.Module):
            def __init__(self, config, pretrained):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, images):
                return F.normalize(self.linear(images), dim=1)

        class Images(torch.utils.data.Dataset):
            def __init__(self, records, labels, height, width, visible):
                self.labels = labels
                self.visible = visible

            def __len__(self):
                return 6

            def __getitem__(self, index):
                value = values["visible" if self.visible else "thermal"][index]
                label = int(self.labels[index])
                return (value, value.clone(), label) if self.visible else (value, label)

        with tempfile.TemporaryDirectory(prefix="rahsm-unit-") as temporary:
            config = configuration()
            config.update(output_dir=temporary, epochs=2, warmup_epochs=1, iterations=1,
                          batch_size=4, instances=2, workers=0, amp=False)
            config["rahsm"].update(cross_neighbors=1, smoothing_neighbors=1)
            with patch.object(training, "audit_regdb"), patch.object(training, "split_fingerprint", return_value="synthetic"), \
                 patch.object(training, "ReIDEncoder", Encoder), patch.object(training, "TrainingImages", Images), \
                 patch.object(training, "capture_run", return_value=Path(temporary) / "provenance"), \
                 patch.object(training, "read_split", side_effect=lambda root, split, modal, trial: records[modal]), \
                 patch.object(training, "extract_features", side_effect=lambda model, rows, cfg, device: values[rows[0].path.parent.name].clone()), \
                 patch.object(training, "cluster_features", side_effect=lambda features, eps, cfg: (labels.copy(), torch.stack([features[labels == i].mean(0) for i in range(3)]))):
                checkpoint = training.train(config)
                root = Path(temporary)
                epochs = sorted((root / "epochs").iterdir())
                self.assertEqual(len(epochs), 2)
                warm = json.loads((epochs[0] / "summary.json").read_text())
                cross = json.loads((epochs[1] / "summary.json").read_text())
                self.assertFalse(warm["rahsm_active"])
                self.assertTrue(cross["rahsm_active"])
                self.assertAlmostEqual(cross["lambda_last"], 0.3)
                self.assertTrue((epochs[1] / "rahsm_last_batch_roles.npz").is_file())
                training.train(config, resume=checkpoint)
                self.assertEqual(len(list((root / "epochs").iterdir())), 2)
                wrong = copy.deepcopy(config)
                wrong["rahsm"]["pts"] = False
                with self.assertRaises(ValueError):
                    training.train(wrong, resume=checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
