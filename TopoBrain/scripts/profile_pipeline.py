"""Measure prepared EEG in host memory -> frozen codes -> trained head logits.

This deliberately does not call itself an EDF-to-prediction or complete raw-file
pipeline: EDF parsing, filtering, resampling, file I/O and model setup are excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def validate_alignment(eeg, raw_arrays, raw_metadata, cache, raw_hash, checkpoint_hash):
    import numpy as np
    if eeg.shape[:3] != cache["codes_t"].shape:
        raise ValueError("Prepared EEG and cache sample/channel/patch dimensions differ")
    for name in ("sample_ids", "subject_ids", "labels", "channel_ids", "coords"):
        if name not in raw_arrays or name not in cache:
            raise ValueError(f"Profiling requires explicit aligned {name} in raw and cache")
        if not np.array_equal(raw_arrays[name], cache[name]):
            raise ValueError(f"Prepared EEG and cache differ in {name}")
    raw_valid = raw_arrays.get("valid_channel_mask", np.ones(eeg.shape[:2], dtype=bool))
    cache_valid = cache.get("valid_channel_mask", np.ones(eeg.shape[:2], dtype=bool))
    if raw_valid.shape != eeg.shape[:2] or cache_valid.shape != eeg.shape[:2]:
        raise ValueError("Invalid observation mask shape")
    if raw_valid.dtype.kind != "b" or cache_valid.dtype.kind != "b":
        raise ValueError("Observation masks must be boolean")
    if not raw_valid.all() or not cache_valid.all() or cache["metadata"].get("raw_missing_reencoded", False):
        raise ValueError("This profile supports clean, fully observed prepared EEG only")
    if cache["metadata"].get("tokenizer_checkpoint_hash") != checkpoint_hash:
        raise ValueError("Checkpoint SHA256 differs from the cache tokenizer_checkpoint_hash")
    if cache["metadata"].get("raw_source_sha256") != raw_hash:
        raise ValueError("Prepared source SHA256 differs from the cache raw_source_sha256")
    if bool(raw_metadata["synthetic"]) != bool(cache["metadata"]["synthetic"]):
        raise ValueError("Raw/cache synthetic provenance differs")
    positions = raw_metadata.get("tokenizer_channel_positions", list(range(1, eeg.shape[1] + 1)))
    cached_positions = cache["metadata"].get("tokenizer_channel_positions", list(range(1, eeg.shape[1] + 1)))
    if list(positions) != list(cached_positions):
        raise ValueError("Raw/cache tokenizer channel positions differ")
    return list(positions)


def profile_pipeline(raw_path, cache_path, run_dir, repo_path, checkpoint_path, output_path,
                     device="auto", warmup=2, iterations=5, batch_size=1):
    if warmup < 0 or iterations < 1 or batch_size < 1:
        raise ValueError("Require warmup>=0, iterations>=1, batch_size>=1")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing pipeline profile")
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import numpy as np
    import torch
    from topobrain.data import file_sha256, load_cache
    from topobrain.engine import load_run_model
    from topobrain.experiments import evidence_flags
    from topobrain.tokenizer import FrozenTokenizer, load_prepared

    eeg, arrays, raw_metadata = load_prepared(raw_path)
    cache = load_cache(cache_path)
    raw_hash, checkpoint_hash = file_sha256(raw_path), file_sha256(checkpoint_path)
    positions = validate_alignment(eeg, arrays, raw_metadata, cache, raw_hash, checkpoint_hash)
    adapter = FrozenTokenizer(repo_path, checkpoint_path, device)
    actual_device = adapter.device
    for name, actual in (("codebook_t", adapter.book_t), ("codebook_f", adapter.book_f)):
        if not np.array_equal(cache[name], actual.detach().cpu().numpy()):
            raise ValueError(f"Cache {name} differs from the frozen tokenizer")
    if cache["metadata"].get("tokenizer_source_sha256") != adapter.source_hash:
        raise ValueError("Tokenizer source SHA256 differs from the extraction source")
    model = load_run_model(run_dir, cache, str(actual_device))
    model.eval()
    adapter.model.eval()
    selected = min(batch_size, len(eeg))
    batch = np.ascontiguousarray(eeg[:selected], dtype=np.float32)
    host_mask = np.ones(batch.shape[:2], dtype=bool)

    def forward():
        # encode performs the actual numpy host -> device transfer every call.
        temporal, frequency = adapter.encode(batch, channel_positions=positions)
        observed = torch.as_tensor(host_mask, dtype=torch.bool, device=actual_device)
        return model(temporal, frequency, valid_mask=observed)

    with torch.inference_mode():
        for _ in range(warmup):
            logits = forward()
        if actual_device.type == "cuda":
            torch.cuda.synchronize(actual_device)
            torch.cuda.reset_peak_memory_stats(actual_device)
        durations = []
        for _ in range(iterations):
            if actual_device.type == "cuda":
                torch.cuda.synchronize(actual_device)
            start = time.perf_counter()
            logits = forward()
            if actual_device.type == "cuda":
                torch.cuda.synchronize(actual_device)
            durations.append(time.perf_counter() - start)
    peak_allocated = torch.cuda.max_memory_allocated(actual_device) if actual_device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(actual_device) if actual_device.type == "cuda" else None
    if not torch.isfinite(logits).all():
        raise RuntimeError("Pipeline returned nonfinite logits")
    # Check implementation/data alignment, not prediction quality. This extra
    # check is outside the timer and cannot affect the measured latency.
    with torch.inference_mode():
        temporal, frequency = adapter.encode(batch, channel_positions=positions)
    if not np.array_equal(temporal.cpu().numpy(), cache["codes_t"][:selected]):
        raise ValueError("Fresh temporal codes differ from cached clean codes")
    if not np.array_equal(frequency.cpu().numpy(), cache["codes_f"][:selected]):
        raise ValueError("Fresh frequency codes differ from cached clean codes")
    metadata = {**cache["metadata"],
                "pilot_only": bool(raw_metadata.get("pilot_only", False) or cache["metadata"].get("pilot_only", False)),
                "smoke_only": bool(raw_metadata.get("smoke_only", False) or cache["metadata"].get("smoke_only", False))}
    flags = evidence_flags(metadata)
    latency = np.asarray(durations, dtype=float)
    result = {"status": "completed", "scope": "prepared_eeg_to_logits", **flags,
        "device": str(actual_device), "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(actual_device) if actual_device.type == "cuda" else None,
        "warmup": warmup, "iterations": iterations,
        "requested_batch_size": batch_size, "actual_batch_size": selected,
        "input_shape": list(batch.shape), "dtype": "float32",
        "latency_ms_per_batch_mean": float(latency.mean() * 1000),
        "latency_ms_per_batch_median": float(np.median(latency) * 1000),
        "latency_ms_per_batch_min": float(latency.min() * 1000),
        "latency_ms_per_batch_max": float(latency.max() * 1000),
        "latency_ms_per_batch_samples": (latency * 1000).tolist(),
        "throughput_samples_per_second": float(selected * iterations / latency.sum()),
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "input_transfer_included": True, "encoder_included": True,
        "quantization_included": True, "classification_head_included": True,
        "preprocessing_included": False, "file_io_included": False,
        "model_setup_included": False, "output_transfer_to_host_included": False,
        "boundary": "In-memory already-prepared EEG to device logits, including host-to-device input transfer. Excludes EDF parsing, filtering, resampling, disk I/O, model/checkpoint loading, and output transfer to host; not a complete EDF pipeline.",
        "raw_path": str(Path(raw_path).resolve()), "raw_sha256": raw_hash,
        "cache_path": str(Path(cache_path).resolve()), "cache_sha256": file_sha256(cache_path),
        "run_dir": str(Path(run_dir).resolve()), "checkpoint_sha256": checkpoint_hash,
        "tokenizer_source_sha256": adapter.source_hash, "source_repository_commit": adapter.source_commit,
        "code_alignment_verified": True, "profiled_sample_ids": arrays["sample_ids"][:selected].tolist()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    result = profile_pipeline(args.raw, args.cache, args.run_dir, args.repo, args.checkpoint, args.output,
                              args.device, args.warmup, args.iterations, args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
