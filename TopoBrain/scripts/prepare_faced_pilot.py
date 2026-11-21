"""Download bounded public FACED BDF record ranges for a pipeline-only pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np


BASE = "https://data.nemar.org/nm000112/v1.1.4"
EMOTIONS = tuple(sorted(("Anger", "Disgust", "Fear", "Sadness", "Neutral", "Amusement", "Inspiration", "Joy", "Tenderness")))
USER_AGENT = "TopoBrain-public-data-pilot/0.1"


def sha256(block):
    return hashlib.sha256(block).hexdigest()


def request_range(url, start, end, *, timeout=30, retries=3, opener=None):
    """Reject HTTP200 BEFORE reading, exact Content-Range and bounded body only."""
    if start < 0 or end < start or retries not in (1, 2, 3):
        raise ValueError("invalid byte bounds or retries (allowed 1..3)")
    opener = urllib.request.urlopen if opener is None else opener
    expected = end - start + 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    for attempt in range(retries):
        try:
            with opener(request, timeout=timeout) as response:
                if response.status != 206:
                    raise ValueError(f"Range server returned HTTP {response.status}, expected 206; refusing full-file response")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if match is None or (int(match[1]), int(match[2])) != (start, end) or int(match[3]) <= end:
                    raise ValueError("invalid/mismatched Content-Range")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected:
                    raise ValueError("Content-Length differs from requested range")
                block = response.read(expected + 1)
                if len(block) != expected:
                    raise ValueError("range payload has unexpected length")
                return block
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def parse_bdf_header(block):
    if len(block) < 256 or block[:8] != b"\xffBIOSEMI":
        raise ValueError("not a supported 24-bit BioSemi BDF header")
    try:
        count = int(block[252:256].decode("ascii").strip())
        header_bytes = int(block[184:192].decode("ascii").strip())
        records = int(block[236:244].decode("ascii").strip())
        duration = float(block[244:252].decode("ascii").strip())
    except ValueError as exc:
        raise ValueError("invalid fixed BDF header fields") from exc
    if count < 1 or count > 256 or header_bytes != 256 + 256 * count or len(block) < header_bytes:
        raise ValueError("invalid channel/header length or incomplete BDF header")
    if records <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("BDF needs known positive record count and duration")
    widths = (("labels", 16), ("transducers", 80), ("units", 8), ("physical_min", 8), ("physical_max", 8),
              ("digital_min", 8), ("digital_max", 8), ("prefilters", 80), ("samples_per_record", 8), ("reserved", 32))
    offset = 256
    result = {"channels": count, "header_bytes": header_bytes, "records": records, "record_duration": duration}
    for field, width in widths:
        result[field] = [block[offset + i * width:offset + (i + 1) * width].decode("latin-1").strip() for i in range(count)]
        offset += count * width
    for field in ("physical_min", "physical_max", "digital_min", "digital_max"):
        result[field] = np.asarray(result[field], dtype=np.float64)
        if not np.isfinite(result[field]).all():
            raise ValueError("nonfinite BDF physical/digital scale")
    result["samples_per_record"] = np.asarray(result["samples_per_record"], dtype=np.int64)
    samples = result["samples_per_record"]
    if np.any(samples <= 0) or not np.all(samples == samples[0]):
        raise ValueError("mixed channel sampling rates are unsupported; no silent resampling")
    if np.any(result["physical_max"] <= result["physical_min"]) or np.any(result["digital_max"] <= result["digital_min"]):
        raise ValueError("invalid BDF scale range")
    if np.any(result["digital_min"] < -8388608) or np.any(result["digital_max"] > 8388607):
        raise ValueError("digital bounds exceed signed 24-bit range")
    conversion = []
    for unit in result["units"]:
        normalized = unit.replace("\xb5", "u").replace("\u03bc", "u").strip().lower()
        if normalized not in {"v", "mv", "uv"}:
            raise ValueError("unsupported physical unit in BDF: " + repr(unit))
        conversion.append({"v": 1e6, "mv": 1e3, "uv": 1.0}[normalized])
    result["unit_to_microvolt"] = np.asarray(conversion)
    result["record_bytes"] = int(samples.sum() * 3)
    result["sample_rate"] = float(samples[0] / duration)
    return result


def decode_bdf_records(payload, header):
    """Decode complete records -> physical microvolts [C,T], not raw integers."""
    record_bytes = header["record_bytes"]
    if not payload or len(payload) % record_bytes:
        raise ValueError("payload must contain complete BDF records")
    records = len(payload) // record_bytes
    channels = header["channels"]
    samples = int(header["samples_per_record"][0])
    triples = np.frombuffer(payload, dtype=np.uint8).reshape(records, channels, samples, 3).astype(np.int32)
    unsigned = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
    digital = (unsigned ^ 0x800000) - 0x800000
    slope = (header["physical_max"] - header["physical_min"]) / (header["digital_max"] - header["digital_min"])
    intercept = header["physical_min"] - header["digital_min"] * slope
    physical = (digital.astype(np.float64) * slope[None, :, None] + intercept[None, :, None])
    physical *= header["unit_to_microvolt"][None, :, None]
    return physical.transpose(1, 0, 2).reshape(channels, records * samples)


class DownloadCache:
    def __init__(self, folder, allow_download):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.folder / "manifest.json"
        self.entries = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        self.allow_download = allow_download

    def get(self, url, filename, start=None, end=None):
        path = self.folder / filename
        identity = {"url": url, "byte_range": None if start is None else [start, end]}
        entry = self.entries.get(filename)
        if path.exists():
            payload = path.read_bytes()
            if entry is None or entry["url"] != url or entry["byte_range"] != identity["byte_range"] or entry["sha256"] != sha256(payload):
                raise ValueError("cache checksum/provenance mismatch: " + filename)
            return payload
        if not self.allow_download:
            raise RuntimeError("missing cached public data; rerun with --download: " + filename)
        if start is not None:
            payload = request_range(url, start, end)
        else:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if response.status != 200:
                            raise ValueError("sidecar response must be HTTP200")
                        payload = response.read(4 * 1024 * 1024 + 1)
                        if len(payload) > 4 * 1024 * 1024:
                            raise ValueError("sidecar exceeds bounded 4MiB limit")
                    break
                except (urllib.error.URLError, TimeoutError, OSError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        path.write_bytes(payload)
        self.entries[filename] = {**identity, "bytes": len(payload), "sha256": sha256(payload)}
        self.manifest_path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        return payload


def table(block):
    return list(csv.DictReader(io.StringIO(block.decode("utf-8-sig")), delimiter="\t"))


def clean_channel_name(name):
    cleaned = re.sub(r"^EEG\s+", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"(?:[-_ ]REF)$", "", cleaned, flags=re.IGNORECASE).strip()


def standard_coordinates(channel_names):
    import mne
    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "A1": "M1", "A2": "M2"}
    result, resolved = [], []
    for original in channel_names:
        name = clean_channel_name(original)
        if name not in positions:
            name = aliases.get(name, name)
        if name not in positions:
            raise ValueError("no standard_1020 coordinate for channel " + original)
        result.append(positions[name])
        resolved.append(name)
    coords = np.asarray(result, dtype=np.float32)
    if coords.shape != (len(channel_names), 3) or not np.isfinite(coords).all():
        raise ValueError("invalid montage coordinates")
    return coords, resolved


def select_trials(events, windows_per_class, context_seconds=5.0):
    plans = []
    for emotion in EMOTIONS:
        candidates = []
        for row in events:
            source_emotion = row.get("emotion_label", "")
            source_binary = row.get("binary_label", "")
            # NEMAR v1.1.4 uses a literal backslash for the neutral category.
            # Normalize only this exact, independently confirmed conjunction.
            corrected_neutral = source_emotion == "\\" and source_binary == "neutral"
            event_emotion = "Neutral" if corrected_neutral else source_emotion.strip()
            if event_emotion != emotion:
                continue
            try:
                onset, duration = float(row["onset"]), float(row["duration"])
            except (ValueError, KeyError):
                continue
            needed = windows_per_class * 10 + 2 * context_seconds
            if math.isfinite(onset) and math.isfinite(duration) and onset >= 0 and duration >= needed:
                candidates.append((onset, duration, row))
        if not candidates:
            raise ValueError(f"class {emotion} has no trial with {windows_per_class} nonoverlapping 10s windows and >=5s context")
        onset, duration, row = min(candidates, key=lambda item: item[0])
        first_window = onset + (duration - windows_per_class * 10) / 2
        plans.append({"emotion": emotion, "label": EMOTIONS.index(emotion), "onset": onset,
                      "duration": duration, "video_index": row.get("video_index", ""),
                      "source_emotion_label": row.get("emotion_label", ""),
                      "source_binary_label": row.get("binary_label", ""),
                      "source_label_normalization_applied": row.get("emotion_label", "") == "\\" and row.get("binary_label", "") == "neutral",
                      "window_starts": [first_window + i * 10 for i in range(windows_per_class)]})
    return plans


def preprocess_block(eeg_microvolt, source_rate):
    from scipy.signal import butter, resample_poly, sosfiltfilt
    from fractions import Fraction
    if source_rate <= 90 or not math.isfinite(source_rate):
        raise ValueError("source rate cannot support 45Hz lowpass")
    sos = butter(4, [0.3, 45.0], btype="bandpass", fs=source_rate, output="sos")
    filtered = sosfiltfilt(sos, eeg_microvolt, axis=-1)
    ratio = Fraction(200 / source_rate).limit_denominator(10000)
    if not np.isclose(ratio.numerator / ratio.denominator, 200 / source_rate, rtol=0, atol=1e-10):
        raise ValueError("source rate not representable by supported rational resampling")
    return resample_poly(filtered, ratio.numerator, ratio.denominator, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/faced_pilot/raw.npz")
    parser.add_argument("--subjects", nargs="+", default=["sub-000", "sub-001", "sub-002"])
    parser.add_argument("--windows-per-class", type=int, default=2)
    parser.add_argument("--download", action="store_true", help="authorize bounded public network fetches; otherwise use verified cache only")
    args = parser.parse_args()
    if len(args.subjects) != 3 or len(set(args.subjects)) != 3 or any(re.fullmatch(r"sub-\d{3}", x) is None for x in args.subjects):
        parser.error("this pilot requires exactly three distinct sub-NNN subjects")
    if args.windows_per_class < 1:
        parser.error("windows-per-class must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = DownloadCache(output.parent / "downloads", args.download)
    subject_info = []
    failures = {}
    for subject in args.subjects:
        stem = f"{subject}_task-watchingVideoClips_"
        prefix = f"{BASE}/{subject}/eeg/{stem}"
        sidecars = {suffix: cache.get(prefix + suffix, stem + suffix) for suffix in ("eeg.json", "channels.tsv", "events.tsv")}
        fixed = cache.get(prefix + "eeg.bdf", stem + "fixed_header.bin", 0, 255)
        header_length = int(fixed[184:192].decode("ascii").strip())
        if header_length < 512 or header_length > 256 + 256 * 256:
            raise ValueError("unsupported BDF header length")
        full_header = cache.get(prefix + "eeg.bdf", stem + "header.bin", 0, header_length - 1)
        header = parse_bdf_header(full_header)
        channel_rows = table(sidecars["channels.tsv"])
        names = [row["name"] for row in channel_rows]
        if len(names) != 32 or names != header["labels"] or any(row.get("type") != "EEG" for row in channel_rows):
            raise ValueError(subject + ": BDF/sidecar must agree on exactly 32 EEG channels")
        try:
            plans = select_trials(table(sidecars["events.tsv"]), args.windows_per_class)
        except ValueError as exc:
            failures[subject] = str(exc)
            continue
        subject_info.append((subject, prefix, stem, header, names, plans, json.loads(sidecars["eeg.json"].decode("utf-8-sig"))))
    if failures:
        raise RuntimeError("incomplete subject/class coverage; no dataset fabricated: " + json.dumps(failures))
    eeg, labels, subject_ids, record_ids, sample_ids, sample_manifest = [], [], [], [], [], []
    canonical_names = subject_info[0][4]
    coords, coordinate_names = standard_coordinates(canonical_names)
    for subject, prefix, stem, header, names, plans, sidecar in subject_info:
        if names != canonical_names:
            raise ValueError("channel order differs across subjects; explicit harmonization required")
        rate = header["sample_rate"]
        if not np.isclose(rate, float(sidecar["SamplingFrequency"])):
            raise ValueError("BDF and sidecar sampling frequency disagree")
        for plan in plans:
            block_start = max(0, math.floor((plan["window_starts"][0] - 5) / header["record_duration"]))
            block_stop = math.ceil((plan["window_starts"][-1] + 10 + 5) / header["record_duration"])
            if block_stop > header["records"]:
                raise ValueError("trial context extends past BDF recording")
            byte_start = header["header_bytes"] + block_start * header["record_bytes"]
            byte_end = header["header_bytes"] + block_stop * header["record_bytes"] - 1
            filename = stem + f"records_{block_start}_{block_stop}.bin"
            payload = cache.get(prefix + "eeg.bdf", filename, byte_start, byte_end)
            microvolt = decode_bdf_records(payload, header)
            processed = preprocess_block(microvolt, rate)
            for window_index, start in enumerate(plan["window_starts"]):
                offset = round((start - block_start * header["record_duration"]) * 200)
                window = processed[:, offset:offset + 2000]
                if window.shape != (32, 2000) or not np.isfinite(window).all():
                    raise ValueError("processed trial did not produce a finite 10s window")
                identifier = f"{subject}_{plan['emotion']}_video{plan['video_index']}_window{window_index}"
                eeg.append((window / 100.0).reshape(32, 10, 200).astype(np.float32))
                labels.append(plan["label"])
                subject_ids.append(subject)
                record_ids.append(subject + "_task-watchingVideoClips")
                sample_ids.append(identifier)
                sample_manifest.append({"sample_id": identifier, "subject": subject, "emotion": plan["emotion"],
                                        "source_emotion_label": plan["source_emotion_label"], "source_binary_label": plan["source_binary_label"],
                                        "source_label_normalization_applied": plan["source_label_normalization_applied"],
                                        "video_index": plan["video_index"], "trial_onset": plan["onset"], "trial_duration": plan["duration"],
                                        "requested_window_start": start, "actual_resampled_window_start": block_start * header["record_duration"] + offset / 200,
                                        "window_seconds": 10, "source_rate": rate, "source_url": prefix + "eeg.bdf",
                                        "source_byte_range": [byte_start, byte_end], "source_chunk_sha256": sha256(payload)})
            print(f"Prepared {subject} {plan['emotion']} ({args.windows_per_class} windows)", flush=True)
    expected = len(args.subjects) * len(EMOTIONS) * args.windows_per_class
    if len(eeg) != expected:
        raise RuntimeError("unexpected sample count; refusing incomplete pilot")
    coverage = {subject: {emotion: sum(s == subject and y == index for s, y in zip(subject_ids, labels))
                          for index, emotion in enumerate(EMOTIONS)} for subject in args.subjects}
    metadata = {"synthetic": False, "pilot_only": True, "dataset": "FACED NEMAR BIDS nm000112 v1.1.4",
                "provenance": {"official_nemar": "https://nemar.org/dataset/nm000112", "original_paper": "https://www.nature.com/articles/s41597-023-02650-w", "source_base": BASE},
                "license": {"identifier": "CC-BY-4.0", "source": "https://nemar.org/dataset/nm000112"},
                "label_mapping": {emotion: i for i, emotion in enumerate(EMOTIONS)}, "class_coverage": coverage,
                "source_label_normalization": {"rule_id": "nemar_nm000112_v1.1.4_neutral_backslash", "source_emotion_label": "\\", "required_binary_label": "neutral",
                                               "normalized_label": "Neutral", "condition": "exact source emotion_label is one literal backslash AND exact binary_label is neutral",
                                               "exclusions": "n/a, empty, or other event labels are not mapped to Neutral"},
                "preprocessing": {"bandpass_hz": [0.3, 45], "filter": "4th-order Butterworth SOS, zero-phase sosfiltfilt", "context_seconds_minimum": 5,
                                  "resample": "scipy.signal.resample_poly to 200Hz", "output_sample_rate": 200, "window_seconds": 10,
                                  "physical_scaling": "per-channel BDF physical/digital min/max and dimension -> microvolts", "input_normalization": "microvolts divided by 100", "eeg_unit": "microvolt/100"},
                "coordinate_source": "MNE standard_1020 template, not participant-measured electrode coordinates", "coordinate_names": coordinate_names,
                "source_checksums": cache.entries, "source_checksums_scope": "retrieved headers, metadata and byte-range chunks only; not full source-file hashes",
                "samples": sample_manifest, "limitations": ["one subject per split is pipeline testing, not population validation", "NEMAR BIDS conversion is not a claim of exact original CodeBrain preprocessing", "event labels are video-target emotion labels, not inferred from subject identity", "source reference metadata may be n/a; no unreported rereferencing"]}
    np.savez_compressed(output, eeg=np.stack(eeg), labels=np.asarray(labels, dtype=np.int64), subject_ids=np.asarray(subject_ids),
                        channel_ids=np.asarray(canonical_names), coords=coords, record_ids=np.asarray(record_ids), sample_ids=np.asarray(sample_ids),
                        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    split = {"train": [args.subjects[0]], "val": [args.subjects[1]], "test": [args.subjects[2]]}
    (output.parent / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (output.parent / "pilot_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved real FACED pipeline-only pilot: {len(eeg)} samples; no population-level claim.")


if __name__ == "__main__":
    main()
