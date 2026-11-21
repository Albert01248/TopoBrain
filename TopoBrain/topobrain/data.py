"""Strict, pickle-free loading and subject-isolated splits for EEG code caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(values: np.ndarray, name: str, shape: tuple, unique: bool = False) -> None:
    if values.shape != shape or values.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode array with shape {shape}")
    strings = values.reshape(-1).tolist()
    if any(not x or x.strip() != x for x in strings):
        raise ValueError(f"{name} contains empty IDs or surrounding whitespace")
    if unique and len(set(strings)) != len(strings):
        raise ValueError(f"{name} must contain unique IDs")


def validate_cache(cache: dict) -> None:
    required = {"codes_t", "codes_f", "labels", "subject_ids", "channel_ids",
                "coords", "codebook_t", "codebook_f", "metadata"}
    missing = required - cache.keys()
    if missing:
        raise ValueError("cache is missing fields: " + ", ".join(sorted(missing)))
    t, f = cache["codes_t"], cache["codes_f"]
    if t.ndim != 3 or t.shape != f.shape or min(t.shape) < 1:
        raise ValueError("codes_t and codes_f must share a nonempty [N,C,P] shape")
    n, c, _ = t.shape
    for name, codes, book_name in (("codes_t", t, "codebook_t"),
                                    ("codes_f", f, "codebook_f")):
        book = cache[book_name]
        if book.ndim != 2 or min(book.shape) < 1 or book.dtype.kind != "f":
            raise ValueError(f"{book_name} must be a nonempty floating [K,D] array")
        if not np.isfinite(book).all():
            raise ValueError(f"{book_name} contains nonfinite values")
        if codes.dtype.kind not in "iu" or np.any(codes < 0) or np.any(codes >= len(book)):
            raise ValueError(f"{name} must contain integer indices within its codebook")
    if cache["codebook_t"].shape[1] != cache["codebook_f"].shape[1]:
        raise ValueError("time and frequency codebooks must have the same embedding dimension")
    y = cache["labels"]
    if y.shape != (n,) or y.dtype.kind not in "iu" or np.any(y < 0):
        raise ValueError("labels must be nonnegative integer [N]")
    _strings(cache["subject_ids"], "subject_ids", (n,))
    _strings(cache["channel_ids"], "channel_ids", (c,), unique=True)
    coords = cache["coords"]
    if coords.shape != (c, 3) or coords.dtype.kind != "f" or not np.isfinite(coords).all():
        raise ValueError("coords must be finite floating [C,3] coordinates")
    if "valid_channel_mask" in cache:
        mask = cache["valid_channel_mask"]
        if mask.dtype.kind != "b" or mask.shape != (n, c):
            raise ValueError("valid_channel_mask must be boolean [N,C]")
        if not np.all(mask.any(axis=1)):
            raise ValueError("each sample must have at least one observed channel")
    for name in ("record_ids", "sample_ids"):
        if name in cache:
            _strings(cache[name], name, (n,), unique=(name == "sample_ids"))
    metadata = cache["metadata"]
    if not isinstance(metadata, dict) or type(metadata.get("synthetic")) is not bool:
        raise ValueError("metadata must declare synthetic as a JSON boolean")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, (str, dict)) or not provenance:
        raise ValueError("metadata must contain nonempty string/object provenance")


def load_cache(path: str | Path) -> dict:
    """Read arrays, validate their identities, and parse scalar JSON metadata.

    No data-dependent dictionary, statistic, or normalization is fitted here.
    Numerical code IDs are categorical lookup indices, never ordered features.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    raw = arrays.pop("metadata_json", None)
    if raw is None or raw.shape != () or raw.dtype.kind != "U":
        raise ValueError("metadata_json must be a scalar Unicode JSON string")
    try:
        arrays["metadata"] = json.loads(str(raw.item()))
    except (ValueError, TypeError) as exc:
        raise ValueError("metadata_json is invalid JSON") from exc
    validate_cache(arrays)
    return arrays


def subject_split(subject_ids: np.ndarray, specification: dict) -> dict[str, np.ndarray]:
    """Require exact subject coverage and disjoint nonempty train/val/test lists."""
    subjects = np.asarray(subject_ids)
    _strings(subjects, "subject_ids", subjects.shape)
    if subjects.ndim != 1 or len(subjects) == 0:
        raise ValueError("subject_ids must be nonempty one-dimensional Unicode IDs")
    if not isinstance(specification, dict) or set(specification) != {"train", "val", "test"}:
        raise ValueError("split must contain exactly train, val, and test")
    groups = {}
    for name, values in specification.items():
        if not isinstance(values, list) or not values or any(type(x) is not str for x in values):
            raise ValueError(f"{name} must be a nonempty list of string subject IDs")
        if any(not x or x.strip() != x for x in values) or len(set(values)) != len(values):
            raise ValueError(f"{name} contains duplicate/empty/whitespace IDs")
        groups[name] = set(values)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"subject leakage between {left} and {right}")
    observed = set(subjects.tolist())
    declared = set.union(*groups.values())
    if observed != declared:
        raise ValueError("split subject lists must exactly cover the cache; no omitted/unknown IDs")
    result = {name: np.flatnonzero(np.isin(subjects, sorted(values)))
              for name, values in groups.items()}
    if any(len(indices) == 0 for indices in result.values()):
        raise ValueError("each split must contain samples")
    return result
