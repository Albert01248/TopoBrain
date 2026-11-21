"""Finite synthetic BDF/range protocol tests; no network or EEG findings."""
import unittest

import numpy as np

from scripts.prepare_faced_pilot import decode_bdf_records, parse_bdf_header, request_range, select_trials, EMOTIONS


def synthetic_header(units=("uV", "mV", "V"), samples=(4, 4, 4)):
    channels = len(units)
    fixed = bytearray(b" " * 256)
    fixed[:8] = b"\xffBIOSEMI"
    for start, width, value in ((184, 8, 256 + channels * 256), (236, 8, 1), (244, 8, 1), (252, 4, channels)):
        fixed[start:start + width] = str(value).encode().ljust(width)
    fields = ((16, ["F3", "C3", "O1"][:channels]), (80, [""] * channels), (8, units),
              (8, [-1] * channels), (8, [1] * channels), (8, [-8388608] * channels), (8, [8388607] * channels),
              (80, [""] * channels), (8, samples), (32, [""] * channels))
    return bytes(fixed) + b"".join(str(value).encode("latin-1").ljust(width) for width, values in fields for value in values)


def signed24(values):
    values = np.asarray(values, dtype=np.int64) & 0xFFFFFF
    return np.stack([values & 255, (values >> 8) & 255, (values >> 16) & 255], axis=-1).astype(np.uint8).tobytes()


class FakeResponse:
    def __init__(self, status, content_range="bytes 10-13/100", payload=b"abcd", length="4"):
        self.status = status
        self.headers = {"Content-Range": content_range, "Content-Length": length}
        self.payload = payload
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, bound):
        self.read_calls += 1
        return self.payload[:bound]


class BDFRangeTests(unittest.TestCase):
    def test_signed24_extremes_and_physical_units(self):
        header = parse_bdf_header(synthetic_header())
        digital = np.array([[-8388608, -1, 0, 8388607]] * 3)
        decoded = decode_bdf_records(signed24(digital), header)
        expected_base = (digital.astype(np.float64) + 8388608) * 2 / 16777215 - 1
        expected = expected_base * np.array([1, 1000, 1e6])[:, None]
        np.testing.assert_allclose(decoded, expected, rtol=1e-9, atol=1e-9)
        self.assertEqual(decoded[0, 0], -1)
        self.assertEqual(decoded[0, -1], 1)
        self.assertLess(decoded[0, 1], 0)
        self.assertGreater(decoded[0, 2], 0)

    def test_invalid_headers_and_partial_records(self):
        with self.assertRaisesRegex(ValueError, "mixed"):
            parse_bdf_header(synthetic_header(samples=(4, 3, 4)))
        with self.assertRaisesRegex(ValueError, "unit"):
            parse_bdf_header(synthetic_header(units=("bad", "mV", "V")))
        with self.assertRaises(ValueError):
            parse_bdf_header(b"notbdf")
        with self.assertRaisesRegex(ValueError, "complete"):
            decode_bdf_records(b"x", parse_bdf_header(synthetic_header()))

    def test_range200_rejected_before_body_read(self):
        response = FakeResponse(200)
        with self.assertRaisesRegex(ValueError, "refusing full-file"):
            request_range("https://example.test/data", 10, 13, opener=lambda *_args, **_kwargs: response)
        self.assertEqual(response.read_calls, 0)

    def test_range_bounds_and_lengths_checked(self):
        valid = FakeResponse(206)
        self.assertEqual(request_range("https://example.test/data", 10, 13, opener=lambda *_args, **_kwargs: valid), b"abcd")
        for response in (FakeResponse(206, "bytes 11-14/100"), FakeResponse(206, "missing"), FakeResponse(206, length="5"), FakeResponse(206, payload=b"abc")):
            with self.assertRaises(ValueError):
                request_range("https://example.test/data", 10, 13, opener=lambda *_args, **_kwargs: response)

    def test_trial_labels_and_nonoverlap_come_from_events(self):
        events = [{"emotion_label": emotion, "onset": str(100 + i * 100), "duration": "60", "video_index": str(i + 1)} for i, emotion in enumerate(EMOTIONS)]
        plans = select_trials(events, 2)
        self.assertEqual([p["label"] for p in plans], list(range(9)))
        for plan in plans:
            starts = plan["window_starts"]
            self.assertGreaterEqual(starts[0] - plan["onset"], 5)
            self.assertGreaterEqual(plan["onset"] + plan["duration"] - starts[-1] - 10, 5)
            self.assertGreaterEqual(starts[1] - starts[0], 10)
        with self.assertRaisesRegex(ValueError, "no trial"):
            select_trials(events[:-1], 2)

    def test_exact_neutral_source_mapping_preserves_original_label(self):
        events = [{"emotion_label": emotion, "binary_label": "positive", "onset": str(100 + i * 100), "duration": "60", "video_index": str(i + 1)} for i, emotion in enumerate(EMOTIONS)]
        neutral = next(row for row in events if row["emotion_label"] == "Neutral")
        neutral["emotion_label"] = "\\"
        neutral["binary_label"] = "neutral"
        plans = select_trials(events, 2)
        normalized = next(plan for plan in plans if plan["emotion"] == "Neutral")
        self.assertEqual(normalized["source_emotion_label"], "\\")
        self.assertEqual(normalized["source_binary_label"], "neutral")
        self.assertTrue(normalized["source_label_normalization_applied"])
        self.assertEqual(neutral["emotion_label"], "\\")
        for source, binary in (("n/a", "neutral"), ("", "neutral"), ("\\", "positive"), ("\\", "n/a"), ("\\", "Neutral")):
            neutral["emotion_label"], neutral["binary_label"] = source, binary
            with self.assertRaisesRegex(ValueError, "class Neutral has no trial"):
                select_trials(events, 2)


if __name__ == "__main__":
    unittest.main()
