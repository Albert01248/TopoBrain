"""Generate a tiny, explicitly synthetic cache for engineering checks only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def create_fixture(output_dir: str | Path, seed: int = 7301, overwrite: bool = False):
    destination = Path(output_dir).resolve()
    cache_path, split_path = destination / "synthetic_codes.npz", destination / "synthetic_split.json"
    if not overwrite and (cache_path.exists() or split_path.exists()):
        raise FileExistsError("Synthetic fixture exists; choose a new directory or explicitly use --overwrite")
    rng = np.random.default_rng(seed)
    subjects = [f"synthetic_subject_{i:02d}" for i in range(8)]
    labels = np.tile(np.array([0, 1, 2, 0, 1, 2], dtype=np.int64), 8)
    channel_ids = np.array([f"SYNTHETIC_CH_{i:02d}" for i in range(12)])
    # Fibonacci points on a unit sphere, not real electrodes or a clinical montage.
    position = np.arange(12, dtype=np.float64)
    z = 1 - 2 * (position + 0.5) / 12
    angle = position * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    coords = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z)).astype(np.float32)
    regions = {
        "left_hemisphere": channel_ids[coords[:, 0] < 0].tolist(),
        "right_hemisphere": channel_ids[coords[:, 0] >= 0].tolist(),
        "occipital": channel_ids[coords[:, 1] < -0.4].tolist(),
        "midline": channel_ids[np.argsort(np.abs(coords[:, 0]))[:2]].tolist(),
    }
    # Check that both requested k values support changed degree-matched graphs.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from topobrain.data import validate_cache, subject_split
    from topobrain.graphs import build_graph
    for k in (2, 4):
        reference = build_graph(coords, kind="geometry", k=k)
        for graph_seed in (42, 43, 44):
            randomized = build_graph(coords, kind="random_degree", k=k, seed=graph_seed)
            if np.array_equal(reference, randomized) or not np.array_equal(reference.sum(1), randomized.sum(1)):
                raise RuntimeError("Synthetic fixture does not support the declared graph control")
    metadata = {
        "synthetic": True,
        "evidence_status": "SYNTHETIC ENGINEERING CHECK ONLY; NOT EEG PAPER RESULTS",
        "provenance": {"generator": "scripts/make_synthetic_fixture.py", "seed": seed,
                       "labels": "Balanced synthetic classes, independent random categorical codes",
                       "coordinates": "12 artificial Fibonacci-sphere points; no anatomical meaning"},
        "regions": regions,
        "region_note": "Artificial coordinate groups with familiar condition names for CLI tests only; not verified human anatomy",
        "tokenizer_checkpoint_hash": "SYNTHETIC_NO_REAL_TOKENIZER",
        "preprocessing_version": "synthetic_fixture_v1",
        "split_manifest_version": "synthetic_subject_split_v1",
        "raw_missing_reencoded": False,
    }
    arrays = {
        "codes_t": rng.integers(0, 32, size=(48, 12, 6), dtype=np.int64),
        "codes_f": rng.integers(0, 32, size=(48, 12, 6), dtype=np.int64),
        "codebook_t": rng.standard_normal((32, 32)).astype(np.float32),
        "codebook_f": rng.standard_normal((32, 32)).astype(np.float32),
        "labels": labels,
        "subject_ids": np.repeat(np.array(subjects), 6),
        "channel_ids": channel_ids,
        "coords": coords,
        "valid_channel_mask": np.ones((48, 12), dtype=bool),
        "sample_ids": np.array([f"synthetic_epoch_{i:03d}" for i in range(48)]),
        "record_ids": np.repeat(np.array([f"synthetic_record_{i:02d}" for i in range(8)]), 6),
    }
    split = {"train": subjects[:4], "val": subjects[4:6], "test": subjects[6:]}
    validate_cache({**arrays, "metadata": metadata})
    subject_split(arrays["subject_ids"], split)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays, metadata_json=np.array(json.dumps(metadata)))
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return {"cache": str(cache_path), "split": str(split_path), "synthetic": True,
            "shape": [48, 12, 6], "subjects": 8,
            "evidence_status": metadata["evidence_status"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(create_fixture(args.output_dir, args.seed, args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
