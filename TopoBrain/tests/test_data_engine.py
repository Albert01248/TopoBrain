"""Protocol checks use explicit synthetic fixtures, never paper-performance claims."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from topobrain.data import load_cache, subject_split
from topobrain.engine import (
    _evidence_fields, _source_report_fields,
    classification_metrics, paired_subject_bootstrap,
)


def workspace_tempdir():
    allowed = Path(__file__).resolve().parents[1] / "tmp" / "engine-tests"
    allowed.mkdir(parents=True, exist_ok=True)
    directory = tempfile.TemporaryDirectory(prefix="synthetic-", dir=allowed)
    # Cleanup recursively removes only this verified workspace child.
    if not Path(directory.name).resolve().is_relative_to(allowed.resolve()):
        raise RuntimeError("test temporary directory escaped the workspace")
    return directory


def synthetic_arrays():
    rng = np.random.default_rng(105)
    n, c, p, k, d = 12, 4, 3, 8, 4
    return {"codes_t": rng.integers(0, k, (n, c, p), dtype=np.int16),
            "codes_f": rng.integers(0, k, (n, c, p), dtype=np.int16),
            "labels": np.tile(np.array([0, 1], dtype=np.int64), n // 2),
            "subject_ids": np.repeat(np.array(["train_a", "train_b", "validation", "test_a", "test_b", "test_c"]), 2),
            "channel_ids": np.array(["F3", "F4", "C3", "C4"]),
            "coords": rng.normal(size=(c, 3)).astype(np.float32),
            "codebook_t": rng.normal(size=(k, d)).astype(np.float32),
            "codebook_f": rng.normal(size=(k, d)).astype(np.float32),
            "valid_channel_mask": np.ones((n, c), dtype=bool),
            "sample_ids": np.array([f"SYNTHETIC_{i}" for i in range(n)]),
            "metadata_json": np.array(json.dumps({"synthetic": True, "provenance": "unit-test synthetic fixture; not EEG"}))}


def split_spec():
    return {"train": ["train_a", "train_b"], "val": ["validation"], "test": ["test_a", "test_b", "test_c"]}


class DataProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = workspace_tempdir()
        self.addCleanup(self.temp.cleanup)
        self.cache_path = Path(self.temp.name) / "synthetic.npz"

    def write(self, arrays):
        np.savez_compressed(self.cache_path, **arrays)
        return self.cache_path

    def test_load_preserves_codes_and_source_books_without_fitting(self):
        arrays = synthetic_arrays()
        loaded = load_cache(self.write(arrays))
        np.testing.assert_array_equal(loaded["codes_t"], arrays["codes_t"])
        np.testing.assert_array_equal(loaded["codebook_t"], arrays["codebook_t"])
        self.assertTrue(loaded["metadata"]["synthetic"])
        self.assertNotIn("metadata_json", loaded)
        self.assertNotIn("vocabulary", loaded)
        # A codebook entry used only by test remains a lookup entry; no fitting happens.
        arrays["codes_t"][-1] = 7
        second = load_cache(self.write(arrays))
        np.testing.assert_array_equal(second["codebook_t"], loaded["codebook_t"])

    def test_split_disjoint_and_exact_coverage(self):
        subjects = synthetic_arrays()["subject_ids"]
        split = subject_split(subjects, split_spec())
        self.assertEqual(set(np.concatenate(list(split.values()))), set(range(len(subjects))))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertEqual(set(subjects[split["test"]]), {"test_a", "test_b", "test_c"})

    def test_leaks_missing_unknown_and_duplicate_subjects_rejected(self):
        subjects = synthetic_arrays()["subject_ids"]
        bads = []
        overlap = split_spec(); overlap["test"].append("train_a"); bads.append(overlap)
        omitted = split_spec(); omitted["test"].pop(); bads.append(omitted)
        unknown = split_spec(); unknown["test"].append("unknown"); bads.append(unknown)
        duplicate = split_spec(); duplicate["train"].append("train_a"); bads.append(duplicate)
        empty = split_spec(); empty["val"] = []; bads.append(empty)
        for bad in bads:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                subject_split(subjects, bad)

    def test_invalid_ids_codes_coordinates_metadata_and_masks_rejected(self):
        bads = []
        for mutate in (
            lambda x: x["codes_t"].__setitem__((0, 0, 0), 8),
            lambda x: x["coords"].__setitem__((0, 0), np.nan),
            lambda x: x["valid_channel_mask"].__setitem__(0, False),
            lambda x: x["sample_ids"].__setitem__(1, x["sample_ids"][0]),
            lambda x: x.update(metadata_json=np.array('{"provenance":"missing synthetic"}')),
            lambda x: x.update(subject_ids=x["subject_ids"].astype(object)),
        ):
            arrays = synthetic_arrays(); mutate(arrays); bads.append(arrays)
        for arrays in bads:
            with self.subTest(keys=arrays.keys()), self.assertRaises(ValueError):
                load_cache(self.write(arrays))

    def test_fixed_class_metrics_and_undefined_kappa_are_explicit(self):
        result = classification_metrics(np.array([0, 0]), predictions=np.array([0, 0]), classes=np.array([0, 1]))
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertIsNone(result["kappa"])
        self.assertEqual(result["support"], [2, 0])

    def test_paired_bootstrap_samples_people_reproducibly(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        people = np.array(["a", "a", "b", "b", "c", "c"])
        a, b = y.copy(), 1 - y
        first = paired_subject_bootstrap(y, a, b, people, [0, 1], repeats=30, seed=7)
        second = paired_subject_bootstrap(y, a, b, people, [0, 1], repeats=30, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["ci95"], [1, 1])
        self.assertEqual(first["unit"], "subject")
        same = paired_subject_bootstrap(y, a, a, people, [0, 1], repeats=30)
        self.assertEqual(same["ci95"], [0, 0])

    def test_real_pilot_and_smoke_limits_propagate_without_using_epoch_count(self):
        expected = "REAL EEG PIPELINE PILOT ONLY; NOT POPULATION/BENCHMARK EVIDENCE"
        for flag in ("pilot_only", "smoke_only"):
            metadata = {"synthetic": False, flag: True, "provenance": "unit-test schema fixture, not EEG",
                        "label_mapping": {"W": 0, "N1": 1},
                        "diagnostics": {"code_usage_t": 1, "code_usage_f": 396}}
            evidence = _evidence_fields(metadata)
            self.assertTrue(evidence["pilot_only"])
            self.assertFalse(evidence["synthetic"])
            self.assertEqual(evidence["evidence_status"], expected)
            source = _source_report_fields(metadata)
            self.assertEqual(source["diagnostics"], {"code_usage_t": 1, "code_usage_f": 396})
            self.assertEqual(source["label_mapping"], metadata["label_mapping"])
            self.assertEqual(source["cache_metadata"], metadata)
        self.assertFalse(_evidence_fields({"synthetic": False})["pilot_only"])

    def test_synthetic_status_takes_precedence_and_pilot_flags_require_boolean(self):
        fields = _evidence_fields({"synthetic": True, "pilot_only": True, "smoke_only": True})
        self.assertTrue(fields["pilot_only"])
        self.assertEqual(fields["evidence_status"], "SYNTHETIC ENGINEERING CHECK ONLY; NOT EEG PAPER RESULTS")
        with self.assertRaises(ValueError):
            _evidence_fields({"synthetic": False, "pilot_only": "false"})


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch installation is required for engine checks")
class EngineProtocolTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temp = workspace_tempdir()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache_path = self.root / "synthetic.npz"
        np.savez_compressed(self.cache_path, **synthetic_arrays())
        self.split_path = self.root / "split.json"
        self.split_path.write_text(json.dumps(split_spec()), encoding="utf-8")
        self.model_options = {"hidden_per_branch": 4, "graph_depth": 1, "temporal_window": 3, "dropout": 0.0}
        self.graph_options = {"kind": "geometry", "k": 1, "seed": 3}
        self.training = {"epochs": 2, "batch_size": 2, "lr": 0.001, "seed": 11, "device": "cpu", "patience": 2}

    def train(self, name, **kwargs):
        from topobrain.engine import train_run
        return train_run(self.cache_path, self.split_path, self.root / name,
                         self.model_options, self.graph_options, self.training, **kwargs)

    def test_evaluation_batches_respect_mask_and_keep_original_missingness(self):
        import torch
        from topobrain.engine import evaluate_model
        cache = load_cache(self.cache_path)
        cache["valid_channel_mask"][0, 0] = False
        class MaskRecorder(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.seen = []
            def forward(self, t, f, valid_mask):
                self.seen.append(valid_mask.detach().cpu().numpy())
                score = valid_mask.float().sum(1)
                return torch.stack((score, -score), dim=1)
        model = MaskRecorder()
        override = np.ones(cache["codes_t"].shape[:2], dtype=bool)
        override[:, 1] = False
        metrics, pred = evaluate_model(model, cache, np.array([0, 2, 4]), "cpu", 2, override)
        self.assertEqual(metrics["n_samples"], 3)
        self.assertEqual(pred["logits"].shape, (3, 2))
        self.assertEqual(len(model.seen), 2)
        self.assertFalse(model.seen[0][0, 0])
        self.assertFalse(np.concatenate(model.seen)[:, 1].any())
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            evaluate_model(model, cache, np.array([0]), "cpu", 2, np.zeros_like(override))

    def test_val_only_selection_checkpoint_provenance_and_overwrite_guard(self):
        import torch
        from topobrain import engine
        original = engine.evaluate_model
        cache = load_cache(self.cache_path)
        groups = subject_split(cache["subject_ids"], split_spec())
        states, calls = [], []
        def monitored(model, cache, indices, device, batch_size, **kwargs):
            metrics, pred = original(model, cache, indices, device, batch_size, **kwargs)
            if np.array_equal(indices, groups["val"]):
                states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                metrics["macro_f1"] = 0.9 if len(states) == 1 else 0.1
                calls.append("val")
            elif np.array_equal(indices, groups["test"]):
                self.assertTrue((self.root / "selection" / "best.pt").is_file())
                self.assertGreaterEqual(calls.count("val"), 2)
                calls.append("test")
            return metrics, pred
        with patch.object(engine, "evaluate_model", side_effect=monitored):
            report = self.train("selection")
        self.assertEqual(report["best_epoch"], 1)
        self.assertEqual(calls.count("test"), 1)
        checkpoint = torch.load(self.root / "selection" / "best.pt", weights_only=True)
        for name, tensor in states[0].items():
            torch.testing.assert_close(checkpoint["state_dict"][name], tensor, rtol=0, atol=0)
        self.assertEqual(len(checkpoint["history"]), 2)
        manifest = json.loads((self.root / "selection" / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(len(manifest["cache_sha256"]), 64)
        self.assertEqual(manifest["status"], "complete")
        with self.assertRaises(FileExistsError):
            self.train("selection")

    def test_seed_replay_reload_and_schema_mismatch(self):
        from topobrain.engine import evaluate_model, load_run_model
        first, second = self.train("first"), self.train("second")
        with np.load(self.root / "first" / "predictions.npz", allow_pickle=False) as a, np.load(self.root / "second" / "predictions.npz", allow_pickle=False) as b:
            np.testing.assert_array_equal(a["logits"], b["logits"])
            self.assertEqual(set(a["subject_ids"]), set(split_spec()["test"]))
            expected = a["logits"].copy()
        cache = load_cache(self.cache_path)
        model = load_run_model(self.root / "first", cache, "cpu")
        indices = subject_split(cache["subject_ids"], split_spec())["test"]
        _, prediction = evaluate_model(model, cache, indices, "cpu", 2)
        np.testing.assert_array_equal(expected, prediction["logits"])
        changed = copy.deepcopy(cache); changed["codebook_t"][0, 0] += 1
        with self.assertRaises(ValueError):
            load_run_model(self.root / "first", changed, "cpu")
        self.training["seed"] = 12
        self.train("different")
        with np.load(self.root / "different" / "predictions.npz", allow_pickle=False) as other:
            self.assertFalse(np.array_equal(expected, other["logits"]))

    def test_train_must_cover_held_out_classes(self):
        arrays = synthetic_arrays(); arrays["labels"][:4] = 0
        np.savez_compressed(self.cache_path, **arrays)
        with self.assertRaises(ValueError):
            self.train("bad_coverage")
        self.assertFalse((self.root / "bad_coverage").exists())

    def test_pilot_limits_are_saved_in_manifest_report_and_checkpoint(self):
        import torch
        arrays = synthetic_arrays()
        # The input is still an artificial test fixture. Only its metadata schema
        # exercises the real-pilot branch; no scientific result is asserted.
        metadata = {"synthetic": False, "pilot_only": True,
                    "provenance": "artificial unit-test fixture for real-pilot flag propagation, NOT EEG",
                    "label_mapping": {"artificial_A": 0, "artificial_B": 1},
                    "diagnostics": {"code_usage_t": 1, "code_usage_f": 396}}
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(self.cache_path, **arrays)
        self.training["epochs"] = 2
        returned = self.train("pilot_flags")
        manifest = json.loads((self.root / "pilot_flags" / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((self.root / "pilot_flags" / "metrics.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(self.root / "pilot_flags" / "best.pt", weights_only=True)
        for saved in (returned, manifest, report, checkpoint):
            self.assertTrue(saved["pilot_only"])
            self.assertEqual(saved["evidence_status"], "REAL EEG PIPELINE PILOT ONLY; NOT POPULATION/BENCHMARK EVIDENCE")
            self.assertEqual(saved["diagnostics"], metadata["diagnostics"])
            self.assertEqual(saved["label_mapping"], metadata["label_mapping"])
            self.assertEqual(saved["cache_metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
