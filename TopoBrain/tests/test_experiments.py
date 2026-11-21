"""Design and missing-channel guards; no scientific performance assertions."""
import json
from pathlib import Path
import unittest

import numpy as np

from topobrain.experiments import (
    diagnostic_masks, evidence_flags, expand_runs, read_config, validate_reencoded_variant,
)


ROOT = Path(__file__).resolve().parents[1]


class ExperimentDesignTests(unittest.TestCase):
    def config(self, number):
        return read_config(ROOT / "configs" / f"experiment_{number}.json")

    def test_all_five_configs_and_expansion_counts(self):
        expected = {1: 15, 2: 15, 3: 12, 4: 36, 5: 3}
        for number, count in expected.items():
            with self.subTest(number=number):
                self.assertEqual(len(expand_runs(self.config(number))), count)

    def test_graph_controls_preserve_time_head_and_training_budget(self):
        runs = expand_runs(self.config(1), smoke=True)
        self.assertEqual({r["graph"]["kind"] for r in runs},
                         {"geometry", "identity", "misaligned", "random_degree", "dense"})
        self.assertTrue(all(r["model"] == runs[0]["model"] for r in runs))
        self.assertTrue(all(r["training"] == runs[0]["training"] for r in runs))

    def test_smoke_reduces_only_epochs_and_seeds(self):
        original = self.config(4)
        snapshot = json.dumps(original, sort_keys=True)
        runs = expand_runs(original, smoke=True)
        self.assertEqual(len(runs), 12)
        self.assertEqual({r["training"]["epochs"] for r in runs}, {1})
        self.assertEqual({r["training"]["seed"] for r in runs}, {42})
        self.assertEqual(json.dumps(original, sort_keys=True), snapshot)

    def test_domain_ablation_keeps_total_width(self):
        runs = expand_runs(self.config(4), smoke=True)
        for run in runs:
            model = run["model"]
            # TopoBrain allocates twice hidden_per_branch in either mode.
            total = model["hidden_per_branch"] * 2
            self.assertEqual(total, 64)
        self.assertEqual({r["model"]["temporal_window"] for r in runs}, {1, 3})

    def test_real_pilot_not_promoted_by_disabling_smoke(self):
        for marker in ("pilot_only", "smoke_only"):
            with self.subTest(marker=marker):
                flags = evidence_flags({"synthetic": False, marker: True}, smoke=False)
                self.assertTrue(flags["pilot_only"])
                self.assertFalse(flags["scientific_results"])
                self.assertIn("PILOT ONLY", flags["evidence_status"])

    def test_evidence_status_respects_all_engineering_flags(self):
        self.assertFalse(evidence_flags({"synthetic": True})["scientific_results"])
        self.assertFalse(evidence_flags({"synthetic": False}, smoke=True)["scientific_results"])
        self.assertTrue(evidence_flags({"synthetic": False}, smoke=False)["scientific_results"])
        self.assertFalse(evidence_flags({"synthetic": False, "pilot_only": False, "smoke_only": True})["scientific_results"])


class MissingChannelGuardTests(unittest.TestCase):
    def setUp(self):
        self.cache = {"valid_channel_mask": np.ones((4, 4), dtype=bool),
                      "sample_ids": np.array(["a", "b", "c", "d"]),
                      "subject_ids": np.array(["s1", "s1", "s2", "s3"]),
                      "labels": np.array([0, 1, 0, 1]),
                      "channel_ids": np.array(["C3", "C4", "O1", "O2"])}
        self.meta = {"tokenizer_checkpoint_hash": "hash", "preprocessing_version": "p1",
                     "split_manifest_version": "s1"}

    def test_missing_regions_are_skipped_without_anatomy_guessing(self):
        masks, skips = diagnostic_masks(self.cache, {}, {"random_missing_rates": []})
        self.assertEqual(masks, {})
        self.assertEqual(skips[0]["condition"], "regional_missing")

    def test_shared_mask_deterministic_and_full_missing_refused(self):
        design = {"mask_seed": 1, "random_missing_rates": [0.5]}
        first, _ = diagnostic_masks(self.cache, {}, design)
        second, _ = diagnostic_masks(self.cache, {}, design)
        np.testing.assert_array_equal(first["random_0.5"], second["random_0.5"])
        for row in first["random_0.5"]:
            np.testing.assert_array_equal(row, first["random_0.5"][0])
        with self.assertRaisesRegex(ValueError, "strictly"):
            diagnostic_masks(self.cache, {}, {"random_missing_rates": [1.0]})

    def test_region_full_missing_skipped(self):
        _, skips = diagnostic_masks(self.cache, {"regions": {"all": self.cache["channel_ids"].tolist()}},
                                    {"random_missing_rates": [], "regions": ["all"]})
        self.assertIn("every valid channel", skips[0]["reason"])

    def test_reencoded_provenance_and_identity_required(self):
        variant = {**self.cache, "valid_channel_mask": self.cache["valid_channel_mask"].copy()}
        variant["valid_channel_mask"][:, 0] = False
        with self.assertRaisesRegex(ValueError, "raw_missing_reencoded"):
            validate_reencoded_variant(self.cache, variant, self.meta, self.meta)
        variant_meta = {**self.meta, "raw_missing_reencoded": True}
        validate_reencoded_variant(self.cache, variant, self.meta, variant_meta)
        variant["sample_ids"] = np.array(["b", "a", "c", "d"])
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            validate_reencoded_variant(self.cache, variant, self.meta, variant_meta)


class PipelineProfileAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.eeg = np.zeros((2, 3, 2, 200), dtype=np.float32)
        self.arrays = {"sample_ids": np.array(["a", "b"]), "subject_ids": np.array(["s1", "s2"]),
                       "labels": np.array([0, 1]), "channel_ids": np.array(["C3", "C4", "Cz"]),
                       "coords": np.eye(3, dtype=np.float32)}
        self.meta = {"synthetic": True}
        self.cache = {**self.arrays, "codes_t": np.zeros((2, 3, 2), dtype=np.int64),
                      "metadata": {"synthetic": True, "tokenizer_checkpoint_hash": "checkpoint",
                                   "raw_source_sha256": "raw"}}

    def test_pipeline_alignment_validates_identity_and_hash(self):
        from scripts.profile_pipeline import validate_alignment
        self.assertEqual(validate_alignment(self.eeg, self.arrays, self.meta, self.cache, "raw", "checkpoint"), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
            validate_alignment(self.eeg, self.arrays, self.meta, self.cache, "raw", "wrong")
        changed = {**self.arrays, "sample_ids": np.array(["b", "a"])}
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            validate_alignment(self.eeg, changed, self.meta, self.cache, "raw", "checkpoint")

    def test_pipeline_rejects_missing_channels(self):
        from scripts.profile_pipeline import validate_alignment
        self.cache["valid_channel_mask"] = np.ones((2, 3), dtype=bool)
        self.cache["valid_channel_mask"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "fully observed"):
            validate_alignment(self.eeg, self.arrays, self.meta, self.cache, "raw", "checkpoint")


if __name__ == "__main__":
    unittest.main()
