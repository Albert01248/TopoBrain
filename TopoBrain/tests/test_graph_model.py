"""Synthetic invariant checks only, never EEG performance evidence."""
import unittest

import numpy as np
import torch

from topobrain.graphs import build_graph, graph_diagnostics
from topobrain.model import ModelConfig, TopoBrain


def coordinates(channels=12):
    angle = np.arange(channels) * 2 * np.pi / channels
    return np.stack([np.cos(angle), np.sin(angle), np.zeros(channels)], axis=1)


class GraphTests(unittest.TestCase):
    def test_graph_contract_and_degrees(self):
        coords = coordinates()
        geometry = build_graph(coords, k=2)
        for kind in ("geometry", "identity", "misaligned", "random_degree", "dense"):
            a = build_graph(coords, kind=kind, k=2, seed=7)
            self.assertTrue(np.array_equal(a, a.T))
            self.assertTrue(np.isin(a, [0, 1]).all())
            self.assertTrue(np.all(np.diag(a) == 0))
            self.assertTrue(np.array_equal(a, build_graph(coords, kind=kind, k=2, seed=7)))
            diag = graph_diagnostics(a, geometry)
            if kind == "random_degree":
                self.assertTrue(diag["same_labeled_node_degrees"])
                self.assertTrue(diag["topology_changed"])
                self.assertGreater(diag["changed_edge_fraction"], 0)
        self.assertTrue(np.all(build_graph(coords, "identity") == 0))

    def test_impossible_randomization_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "random_degree"):
            build_graph(coordinates(5), "random_degree", k=4)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            build_graph(coordinates(5), "misaligned", k=4)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.books = (torch.randn(16, 7, requires_grad=True), torch.randn(16, 9, requires_grad=True))
        self.coords = coordinates(8)
        self.a = torch.from_numpy(build_graph(self.coords, k=2))
        self.t = torch.randint(0, 16, (2, 8, 7))
        self.f = torch.randint(0, 16, (2, 8, 7))

    def model(self, adjacency=None, **overrides):
        options = {"num_classes": 3, "hidden_per_branch": 4, "dropout": 0.0}
        options.update(overrides)
        return TopoBrain(*self.books, self.a if adjacency is None else adjacency, ModelConfig(**options))

    def test_coordinated_channel_permutation_equivariance(self):
        first = self.model().eval()
        permutation = torch.tensor([4, 1, 7, 0, 3, 5, 2, 6])
        changed_a = self.a[permutation][:, permutation]
        second = self.model(changed_a).eval()
        parameters = {k: v for k, v in first.state_dict().items() if k != "adjacency"}
        second.load_state_dict(parameters, strict=False)
        observed = torch.ones((2, 8), dtype=torch.bool)
        observed[0, 2] = False
        with torch.no_grad():
            original_h = first.encode(self.t, self.f, observed)
            changed_h = second.encode(self.t[:, permutation], self.f[:, permutation], observed[:, permutation])
            torch.testing.assert_close(changed_h, original_h[:, permutation], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(first(self.t, self.f, observed), second(self.t[:, permutation], self.f[:, permutation], observed[:, permutation]), rtol=1e-5, atol=1e-6)

    def test_missing_node_values_cannot_pollute_predictions(self):
        model = self.model().eval()
        observed = torch.ones((2, 8), dtype=torch.bool)
        observed[:, [1, 5]] = False
        changed_t, changed_f = self.t.clone(), self.f.clone()
        changed_t[:, [1, 5]] = (changed_t[:, [1, 5]] + 7) % 16
        changed_f[:, [1, 5]] = (changed_f[:, [1, 5]] + 9) % 16
        with torch.no_grad():
            torch.testing.assert_close(model(self.t, self.f, observed), model(changed_t, changed_f, observed), rtol=0, atol=0)
            norm = model._normalized_adjacency(observed)
            self.assertEqual(float(norm[:, [1, 5]].sum()), 0)
            self.assertEqual(float(norm[:, :, [1, 5]].sum()), 0)
        with self.assertRaisesRegex(ValueError, "all-missing"):
            model(self.t, self.f, torch.zeros_like(observed))

    def test_frozen_books_and_trainable_head_gradients(self):
        model = self.model()
        loss = model(self.t, self.f).square().mean()
        loss.backward()
        self.assertIsNone(self.books[0].grad)
        self.assertIsNone(self.books[1].grad)
        self.assertFalse(model.codebook_t.requires_grad)
        self.assertNotIn("codebook_t", dict(model.named_parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.classifier.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_temporal_axis_is_local_and_does_not_mix_channels(self):
        model = self.model(torch.zeros_like(self.a), temporal_window=3).eval()
        changed_t, changed_f = self.t.clone(), self.f.clone()
        changed_t[:, 0, 5] = (changed_t[:, 0, 5] + 1) % 16
        changed_f[:, 0, 5] = (changed_f[:, 0, 5] + 1) % 16
        with torch.no_grad():
            before = model.encode(self.t, self.f)
            after = model.encode(changed_t, changed_f)
            torch.testing.assert_close(before[:, 1:], after[:, 1:], rtol=0, atol=0)
            torch.testing.assert_close(before[:, 0, :4], after[:, 0, :4], rtol=0, atol=0)
            self.assertGreater(float((before[:, 0, 5] - after[:, 0, 5]).abs().sum()), 0)

    def test_supported_domains_depths_and_parameter_matched_window(self):
        for domain in ("both", "temporal", "frequency"):
            for depth in (0, 1, 2):
                model = self.model(domain=domain, graph_depth=depth)
                self.assertEqual(model(self.t, self.f).shape, (2, 3))
        one = self.model(temporal_window=1)
        three = self.model(temporal_window=3)
        self.assertEqual(one.parameter_report()["trainable_parameters"], three.parameter_report()["trainable_parameters"])
        identity = self.model(torch.eye(8)).eval()
        empty = self.model(torch.zeros_like(self.a)).eval()
        empty.load_state_dict(identity.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(identity(self.t, self.f), empty(self.t, self.f), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
