"""Annotation, bipolar-geometry, and filter tests with artificial input only."""
import importlib.util
import unittest

import numpy as np

from scripts.prepare_sleepedf_sample import (
    annotation_epoch_plan, bipolar_midpoints, preprocess_signal,
    resolve_leads, select_smoke_epochs,
)


class AnnotationTests(unittest.TestCase):
    def test_true_labels_unknown_movement_and_n3_n4_merging(self):
        plan = annotation_epoch_plan(
            [0, 60, 90, 120, 150, 180, 210, 240],
            [60, 30, 30, 30, 30, 30, 30, 30],
            ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
             "Sleep stage 4", "Sleep stage R", "Movement time", "Sleep stage ?"], 270)
        self.assertEqual([row["label"] for row in plan], [0, 0, 1, 2, 3, 3, 4])
        self.assertEqual([row["epoch_index"] for row in plan], list(range(7)))
        self.assertEqual(plan[5]["annotation"], "Sleep stage 4")

    def test_partial_annotation_epochs_never_receive_guessed_labels(self):
        # [5,65] fully covers only the PSG-grid epoch [30,60].
        plan = annotation_epoch_plan([5], [60], ["Sleep stage 2"], 100)
        self.assertEqual([row["onset_seconds"] for row in plan], [30])
        clipped = annotation_epoch_plan([0], [90], ["Sleep stage W"], 65)
        self.assertEqual(len(clipped), 2)

    def test_overlapping_known_stage_labels_rejected(self):
        with self.assertRaises(ValueError):
            annotation_epoch_plan([0, 30], [60, 30], ["Sleep stage W", "Sleep stage 1"], 60)
        with self.assertRaises(ValueError):
            annotation_epoch_plan([0], [-1], ["Sleep stage W"], 60)

    def test_smoke_selection_is_deterministic_label_stratified_and_limited(self):
        plan = annotation_epoch_plan([0, 90, 150], [90, 60, 60],
                                     ["Sleep stage W", "Sleep stage 2", "Sleep stage R"], 210)
        selected = select_smoke_epochs(plan, 4)
        self.assertEqual(selected, select_smoke_epochs(plan, 4))
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["label"] for row in selected}, {0, 2, 4})
        self.assertEqual([row["epoch_index"] for row in selected], sorted(row["epoch_index"] for row in selected))

    def test_bipolar_lead_resolution_and_midpoints(self):
        self.assertEqual(resolve_leads(["EEG Pz-Oz", "EOG", "EEG Fpz-Cz"]), ["EEG Fpz-Cz", "EEG Pz-Oz"])
        positions = {"Fpz": [0, 0, 2], "Cz": [0, 0, 0], "Pz": [2, 2, 2], "Oz": [2, 0, 0]}
        np.testing.assert_allclose(bipolar_midpoints(positions), [[0, 0, 1], [2, 1, 1]])
        with self.assertRaises(ValueError):
            resolve_leads(["Fpz", "Cz", "Pz-Oz"])


@unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "SciPy is required for signal preprocessing")
class SignalPreprocessingTests(unittest.TestCase):
    def test_scaling_resampling_passband_and_rejection(self):
        time = np.arange(120 * 100) / 100
        # 10 uV 10Hz signal plus 100 uV DC and 10 uV near-Nyquist noise.
        base = 10e-6 * np.sin(2 * np.pi * 10 * time)
        contaminated = base + 100e-6 + 10e-6 * np.sin(2 * np.pi * 48 * time)
        clean = preprocess_signal(base[None], 100)
        mixed = preprocess_signal(contaminated[None], 100)
        self.assertEqual(mixed.shape, (1, 120 * 200))
        interior = slice(20 * 200, 100 * 200)
        self.assertLess(abs(float(mixed[0, interior].mean())), 0.001)
        self.assertAlmostEqual(float(np.sqrt(np.mean(clean[0, interior] ** 2))), 0.1 / np.sqrt(2), delta=0.003)
        residual = mixed[0, interior] - clean[0, interior]
        self.assertLess(float(np.sqrt(np.mean(residual ** 2))), 0.003)
        self.assertEqual(mixed.dtype, np.float32)

    def test_nonfinite_and_wrong_source_rate_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_signal(np.full((2, 1000), np.nan), 100)
        with self.assertRaises(ValueError):
            preprocess_signal(np.zeros((2, 1000)), 200)


if __name__ == "__main__":
    unittest.main()
