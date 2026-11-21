"""Prepare a few genuinely annotated Sleep-EDF epochs for forward checks only.

This is NOT the ISRUC preprocessing protocol, a benchmark split, or a source of
paper-performance evidence. The two bipolar lead midpoints are approximate
geometry, not electrode nodes or evidence of physiological brain connectivity.
The script converts existing local files and never downloads data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PREPROCESSING_VERSION = "sleepedf-forward-smoke-v1"
DATASET_BASE = "https://physionet.org/files/sleep-edfx/1.0.0/"
LABEL_MAPPING = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
                 "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}
CANONICAL_LEADS = ("Fpz-Cz", "Pz-Oz")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_epoch_plan(onsets, durations, descriptions, recording_duration,
                          epoch_seconds=30.0):
    """Select only complete PSG-grid epochs covered by one known annotation.

    The grid is anchored at PSG time zero. A partial epoch is never assigned the
    label of its first sample; unknown/movement annotations do not supply labels.
    All onsets supplied here must already be aligned to the PSG measurement time.
    """
    if not (len(onsets) == len(durations) == len(descriptions)):
        raise ValueError("annotation arrays must have equal lengths")
    if (not np.isfinite(recording_duration) or recording_duration <= 0
            or not np.isfinite(epoch_seconds) or epoch_seconds <= 0):
        raise ValueError("recording and epoch durations must be finite and positive")
    tolerance = 1e-6
    rows, starts = [], set()
    for onset, duration, description in zip(onsets, durations, descriptions):
        onset, duration = float(onset), float(duration)
        if not np.isfinite(onset) or not np.isfinite(duration) or duration < 0:
            raise ValueError("annotation onsets/durations must be finite and durations nonnegative")
        description = str(description).strip()
        if description not in LABEL_MAPPING:
            continue
        begin = max(0.0, onset)
        end = min(recording_duration, onset + duration)
        first = int(np.ceil((begin - tolerance) / epoch_seconds))
        last = int(np.floor((end + tolerance) / epoch_seconds))
        for epoch in range(max(0, first), last):
            if epoch in starts:
                raise ValueError("overlapping known annotations assign the same PSG epoch more than once")
            starts.add(epoch)
            rows.append({"epoch_index": epoch, "onset_seconds": epoch * epoch_seconds,
                         "duration_seconds": epoch_seconds, "label": LABEL_MAPPING[description],
                         "annotation": description})
    return sorted(rows, key=lambda row: row["epoch_index"])


def select_smoke_epochs(plan, max_epochs=8):
    """Deterministically interleave true stage labels, then restore time order."""
    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError("max_epochs must be a positive integer")
    groups = {label: [row for row in plan if row["label"] == label]
              for label in sorted({row["label"] for row in plan})}
    selected, offset = [], 0
    while len(selected) < max_epochs:
        added = False
        for rows in groups.values():
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == max_epochs:
                    break
        if not added:
            break
        offset += 1
    return sorted(selected, key=lambda row: row["epoch_index"])


def preprocess_signal(signal_volts, source_sample_rate, target_sample_rate=200):
    """0.3--45 Hz zero-phase Butterworth, resample, then microvolts / 100.

    Filtering operates on the full recording before selected epochs are cut, so
    individual epoch boundaries do not create repeated filter transients.
    """
    from scipy.signal import butter, resample_poly, sosfiltfilt
    values = np.asarray(signal_volts, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite [channels,samples] array in SI volts")
    if not np.isclose(source_sample_rate, 100.0) or target_sample_rate != 200:
        raise ValueError("this explicitly scoped Sleep-EDF smoke protocol requires 100 Hz input and 200 Hz output")
    if values.shape[1] < 100:
        raise ValueError("recording is too short for this preprocessing check")
    sos = butter(4, (0.3, 45.0), btype="bandpass", fs=float(source_sample_rate), output="sos")
    filtered = sosfiltfilt(sos, values, axis=-1)
    resampled = resample_poly(filtered, up=2, down=1, axis=-1)
    scaled = resampled * (1e6 / 100.0)
    if not np.isfinite(scaled).all():
        raise ValueError("preprocessing produced nonfinite signal values")
    return scaled.astype(np.float32)


def _lead_name(value):
    return re.sub(r"^EEG\s+", "", str(value).strip(), flags=re.IGNORECASE).lower()


def resolve_leads(channel_names):
    result = []
    for lead in CANONICAL_LEADS:
        matches = [name for name in channel_names if _lead_name(name) == lead.lower()]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one EEG {lead} lead; found {matches}")
        result.append(matches[0])
    return result


def bipolar_midpoints(channel_positions):
    """Approximate standard-head lead midpoints, never true EEG sensor nodes."""
    positions = []
    for lead in CANONICAL_LEADS:
        left, right = lead.split("-")
        if left not in channel_positions or right not in channel_positions:
            raise ValueError(f"standard montage is missing a bipolar lead endpoint: {lead}")
        a, b = np.asarray(channel_positions[left], float), np.asarray(channel_positions[right], float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
            raise ValueError("montage endpoint coordinates must be finite [3] vectors")
        positions.append((a + b) / 2)
    return np.asarray(positions, dtype=np.float32)


def _source_license(psg_path, supplied=None):
    if supplied is not None:
        path = Path(supplied).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        path = next((psg_path.parent / name for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")
                     if (psg_path.parent / name).is_file()), None)
    return {"url": DATASET_BASE + "LICENSE.txt", "local_path": str(path.resolve()) if path else None,
            "sha256": sha256(path) if path else None, "local_file_present": path is not None,
            "note": "Review the supplied source license; this script does not grant additional data rights."}


def prepare(psg_path, hypnogram_path, output_path, max_epochs=8, license_path=None, overwrite=False):
    import mne
    psg_path, hypnogram_path = Path(psg_path).resolve(), Path(hypnogram_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output must have an explicit .npz suffix")
    for source in (psg_path, hypnogram_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output_path in (psg_path, hypnogram_path):
        raise ValueError("output must not overwrite a source EDF")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite explicitly: {output_path}")
    subject_match = re.match(r"^(SC\d{3})", psg_path.name)
    if not subject_match or not hypnogram_path.name.startswith(subject_match.group(1)):
        raise ValueError("this smoke converter expects matching Sleep Cassette SCxxx PSG/hypnogram files")
    if psg_path.name[:6] != hypnogram_path.name[:6]:
        raise ValueError("PSG and hypnogram filenames refer to different recording nights")
    subject_id = subject_match.group(1)
    recording_id = psg_path.stem.removesuffix("-PSG")
    raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose="ERROR")
    source_rate = float(raw.info["sfreq"])
    if not np.isclose(source_rate, 100.0):
        raise ValueError(f"expected 100 Hz Sleep-EDF EEG; got {source_rate}")
    source_channels = resolve_leads(raw.ch_names)
    raw.pick(source_channels)
    annotations = mne.read_annotations(str(hypnogram_path))
    raw.set_annotations(annotations, emit_warning=False)
    # MNE aligns annotation orig_time to meas_date. Subtract first_time so that
    # the onsets below use the same zero as the data returned by get_data().
    onsets = raw.annotations.onset - float(raw.first_time)
    plan = annotation_epoch_plan(onsets, raw.annotations.duration, raw.annotations.description,
                                 raw.n_times / source_rate)
    selected = select_smoke_epochs(plan, max_epochs)
    if not selected:
        raise ValueError("no complete known-stage EEG epochs are available")
    signal = preprocess_signal(raw.get_data(), source_rate)
    frames = []
    for row in selected:
        start = int(round(row["onset_seconds"] * 200))
        epoch = signal[:, start:start + 6000]
        if epoch.shape != (2, 6000):
            raise ValueError("aligned epoch is incomplete after resampling")
        frames.append(epoch.reshape(2, 30, 200))
    eeg = np.stack(frames).astype(np.float32)
    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    coords = bipolar_midpoints(positions)
    source_urls = {"psg": DATASET_BASE + "sleep-cassette/" + psg_path.name,
                   "hypnogram": DATASET_BASE + "sleep-cassette/" + hypnogram_path.name,
                   "dataset": "https://physionet.org/content/sleep-edfx/1.0.0/"}
    metadata = {"synthetic": False, "smoke_only": True,
                "provenance": {"dataset": "Sleep-EDF Expanded 1.0.0", "source_urls": source_urls,
                               "source_paths": {"psg": str(psg_path), "hypnogram": str(hypnogram_path)},
                               "source_sha256": {"psg": sha256(psg_path), "hypnogram": sha256(hypnogram_path)},
                               "purpose": "real annotated forward check; no benchmark or paper result"},
                "source_urls": source_urls, "source_license": _source_license(psg_path, license_path),
                "preprocessing_version": PREPROCESSING_VERSION, "sample_rate": 200,
                "source_sample_rate": source_rate, "shape": list(eeg.shape),
                "unit": "microvolts divided by 100; MNE SI volts multiplied by 10000",
                "reference": "existing bipolar derivations Fpz-Cz and Pz-Oz; no rereferencing",
                "source_channel_names": source_channels,
                "filter": {"type": "fourth-order Butterworth bandpass", "low_hz": 0.3, "high_hz": 45,
                           "phase": "forward-backward zero phase", "implementation": "scipy.signal.sosfiltfilt",
                           "scope": "full selected-channel recording before epoch slicing"},
                "resampling": "scipy.signal.resample_poly up=2 down=1",
                "geometry": {"kind": "approximate_bipolar_geometry", "coordinates_unit": "meters",
                             "source": "MNE standard_1020 ch_pos; midpoint of each lead's two endpoints",
                             "caution": "Bipolar leads are differences, not point electrodes. Two approximate midpoints cannot support spatial/topological scientific conclusions."},
                "label_mapping": LABEL_MAPPING,
                "selection": "deterministic round-robin across available true stages, restored chronological order",
                "selected_epochs": selected, "max_epochs_requested": max_epochs,
                "available_complete_known_epochs": len(plan),
                "subject_id": subject_id, "source_recording_id": recording_id,
                "caution": "Single-record label-stratified smoke subset. No train/test split. Not ISRUC preprocessing or a reproduction of CodeBrain benchmarks; no 60Hz notch is possible on original 100Hz data."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, eeg=eeg,
                        labels=np.asarray([row["label"] for row in selected], dtype=np.int64),
                        subject_ids=np.full(len(selected), subject_id),
                        channel_ids=np.asarray(CANONICAL_LEADS), coords=coords,
                        record_ids=np.full(len(selected), recording_id),
                        sample_ids=np.asarray([f"{recording_id}:epoch{row['epoch_index']:06d}" for row in selected]),
                        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    return {"output": str(output_path), "shape": list(eeg.shape), "labels": sorted(set(int(row["label"]) for row in selected)),
            "synthetic": False, "smoke_only": True, "geometry": "approximate_bipolar_geometry"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--hypnogram", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/sleepedf_prepared.npz"), type=Path)
    parser.add_argument("--max-epochs", default=8, type=int)
    parser.add_argument("--license", dest="license_path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare(args.psg, args.hypnogram, args.output, args.max_epochs, args.license_path, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
