"""The five teacher experiments, with explicit scientific measurement boundaries.

This module consumes a real cache by default. Synthetic smoke fixtures are supplied
by the caller and remain labelled synthetic in every suite report.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any


GRAPH_CONTROLS = ("geometry", "identity", "misaligned", "random_degree", "dense")


def evidence_flags(metadata: dict, smoke: bool = False) -> dict:
    """Do not promote a real-data pilot to scientific results by dropping --smoke."""
    synthetic = bool(metadata.get("synthetic", metadata.get("is_synthetic", False)))
    pilot_only = bool(metadata.get("pilot_only", False) or metadata.get("smoke_only", False))
    if synthetic:
        status = "SYNTHETIC ENGINEERING CHECK ONLY; NOT EEG PAPER RESULTS"
    elif pilot_only:
        status = "REAL DATA PILOT ONLY; NOT CONFIRMATORY EEG PAPER RESULTS"
    elif smoke:
        status = "SMOKE ENGINEERING CHECK ONLY; NOT CONFIRMATORY EEG PAPER RESULTS"
    else:
        status = "INPUT-SPECIFIC RESEARCH RUN; AUDIT PROVENANCE AND LOCKED PROTOCOL BEFORE SCIENTIFIC CLAIMS"
    return {"synthetic": synthetic, "pilot_only": pilot_only,
            "smoke_only": bool(metadata.get("smoke_only", False)), "smoke": bool(smoke),
            "scientific_results": not synthetic and not pilot_only and not smoke,
            "evidence_status": status}


def read_config(config_path: str | Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("experiment_id") not in range(1, 6):
        raise ValueError("experiment_id must be an integer from 1 to 5")
    return config


def expand_runs(config: dict, smoke: bool = False) -> list[dict]:
    """Expand a declared design without touching data or launching training."""
    experiment_id = config["experiment_id"]
    design = config.get("experiment", {})
    model = deepcopy(config.get("model", {}))
    graph = deepcopy(config.get("graph", {}))
    training = deepcopy(config.get("training", {}))
    seeds = list(training.pop("seeds", [42, 43, 44]))
    if not seeds:
        raise ValueError("At least one training seed is required")
    if smoke:
        seeds = seeds[:1]
        training["epochs"] = 1
    cells = []
    if experiment_id in (1, 2):
        cells = [(kind, {}, {"kind": kind}) for kind in design.get("graph_kinds", GRAPH_CONTROLS)]
    elif experiment_id == 3:
        cells = [(f"k{k}_depth{depth}", {"graph_depth": depth}, {"k": k})
                 for k in design.get("neighbor_counts", [2, 4])
                 for depth in design.get("graph_depths", [1, 2])]
    elif experiment_id == 4:
        for domain in design.get("domains", ["both", "temporal", "frequency"]):
            if domain not in ("both", "temporal", "frequency"):
                raise ValueError(f"Unsupported domain: {domain}")
            for spatial in (False, True):
                for temporal in (False, True):
                    # ModelConfig automatically doubles the active single branch
                    # to retain the same total width as two-domain concatenation.
                    width = model.get("hidden_per_branch", 32)
                    cells.append((f"{domain}_S{int(spatial)}T{int(temporal)}",
                                  {"domain": domain, "hidden_per_branch": width,
                                   "temporal_window": model.get("temporal_window", 3) if temporal else 1},
                                  {"kind": "geometry" if spatial else "identity"}))
    else:
        cells = [("profile_geometry", {}, {"kind": "geometry"})]
    runs = []
    for name, model_update, graph_update in cells:
        for seed in seeds:
            runs.append({"name": f"{name}_seed{seed}",
                         "model": {**model, **model_update},
                         "graph": {**graph, **graph_update, "seed": seed},
                         "training": {**training, "seed": seed}})
    return runs


def _unpack_cache(path: str | Path):
    from . import data
    loaded = data.load_cache(path)
    if isinstance(loaded, tuple):
        cache, metadata = loaded
    else:
        cache = loaded
        metadata = cache.get("metadata", {})
    return cache, metadata


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _cache_key(cache: dict, *names):
    for name in names:
        if name in cache:
            return cache[name]
    raise ValueError(f"Cache is missing required field (one of {names})")


def _valid_mask(cache: dict):
    import numpy as np
    value = cache.get("valid_channel_mask", cache.get("valid_mask"))
    if value is None:
        return np.ones(cache["codes_t"].shape[:2], dtype=bool)
    return np.asarray(value, dtype=bool)


def validate_reencoded_variant(reference: dict, variant: dict, reference_meta: dict, variant_meta: dict):
    """Require identical sample identity, order, labels and provenance before pairing."""
    import numpy as np
    for names in (("sample_ids",), ("subject_ids",), ("labels", "y"), ("channel_ids", "channel_names")):
        a = np.asarray(_cache_key(reference, *names))
        b = np.asarray(_cache_key(variant, *names))
        if not np.array_equal(a, b):
            raise ValueError(f"Reencoded cache does not match reference {names[0]}")
    for field in ("coords", "codebook_t", "codebook_f", "record_ids"):
        if field in reference or field in variant:
            if field not in reference or field not in variant or not np.array_equal(reference[field], variant[field]):
                raise ValueError(f"Reencoded cache does not match reference {field}")
    if "codes_t" in reference and reference["codes_t"].shape != variant["codes_t"].shape:
        raise ValueError("Reencoded cache must retain aligned sample/channel/patch dimensions")
    for field in ("tokenizer_checkpoint_hash", "preprocessing_version", "split_manifest_version"):
        if not reference_meta.get(field) or reference_meta.get(field) != variant_meta.get(field):
            raise ValueError(f"Reencoded caches require the same explicit {field}")
    if not variant_meta.get("raw_missing_reencoded", False):
        raise ValueError("Variant must explicitly declare raw_missing_reencoded=true")
    mask = _valid_mask(variant)
    if mask.shape != _valid_mask(reference).shape:
        raise ValueError("Reencoded cache mask shape differs from reference")
    if np.any(mask & ~_valid_mask(reference)):
        raise ValueError("A raw-missing variant cannot resurrect originally absent channels")
    if not mask.any(axis=1).all():
        raise ValueError("Full missing-channel samples are not a valid evaluation condition")
    if np.array_equal(mask, _valid_mask(reference)):
        raise ValueError("Raw missing variant must contain an actual missing-channel intervention")


def diagnostic_masks(cache: dict, metadata: dict, design: dict):
    """Create masks once per suite; all trained graph models receive identical masks."""
    import numpy as np
    valid = _valid_mask(cache)
    if valid.ndim != 2 or not valid.any(axis=1).all():
        raise ValueError("Each sample requires at least one originally valid channel")
    masks, skips = {}, []
    rng = np.random.default_rng(design.get("mask_seed", 7301))
    for rate in design.get("random_missing_rates", [0.25, 0.5]):
        if not 0 < rate < 1:
            raise ValueError("Random missing rates must be strictly between 0 and 1")
        mask = valid.copy()
        # One channel subset per condition, shared across samples and models.
        count = max(1, int(round(valid.shape[1] * rate)))
        count = min(count, valid.shape[1] - 1)
        if count <= 0:
            skips.append({"condition": f"random_{rate:g}", "reason": "Only one channel exists"})
            continue
        removed = rng.choice(valid.shape[1], count, replace=False)
        mask[:, removed] = False
        if not mask.any(axis=1).all():
            skips.append({"condition": f"random_{rate:g}", "reason": "Would remove every valid channel from a sample"})
            continue
        masks[f"random_{rate:g}"] = mask
    regions = metadata.get("regions")
    if not isinstance(regions, dict) or not regions:
        skips.append({"condition": "regional_missing", "reason": "No verified regions mapping in cache metadata; channel anatomy is not guessed"})
        return masks, skips
    channels = [str(x) for x in _cache_key(cache, "channel_ids", "channel_names")]
    for region in design.get("regions", list(regions)):
        if region not in regions:
            skips.append({"condition": f"region_{region}", "reason": "Region absent from verified metadata"})
            continue
        members = regions[region]
        if not isinstance(members, list) or not members:
            skips.append({"condition": f"region_{region}", "reason": "Region has no verified channel list"})
            continue
        if any(str(member) not in channels for member in members):
            raise ValueError(f"Region {region} includes an unknown channel")
        mask = valid.copy()
        mask[:, [channels.index(str(member)) for member in members]] = False
        if not mask.any(axis=1).all():
            skips.append({"condition": f"region_{region}", "reason": "Would remove every valid channel from a sample"})
            continue
        masks[f"region_{region}"] = mask
    return masks, skips


def _test_indices(cache: dict, split_path: str | Path):
    from . import data
    specification = json.loads(Path(split_path).read_text(encoding="utf-8"))
    return data.subject_split(cache["subject_ids"], specification)["test"]


def _profile_head(model, cache: dict, indices, device, batch_sizes, warmup: int, iterations: int):
    import numpy as np
    import torch
    if warmup < 0 or iterations < 1:
        raise ValueError("Profile requires warmup >= 0 and iterations >= 1")
    device = torch.device(device)
    codes_t = _cache_key(cache, "codes_t")
    codes_f = _cache_key(cache, "codes_f")
    valid = _valid_mask(cache)
    rows = []
    model.eval()
    for requested_batch in batch_sizes:
        if requested_batch < 1:
            raise ValueError("Profile batch sizes must be positive")
        selected = np.asarray(indices)[:requested_batch]
        if not len(selected):
            raise ValueError("Test split is empty")
        t = torch.as_tensor(codes_t[selected], dtype=torch.long, device=device)
        f = torch.as_tensor(codes_f[selected], dtype=torch.long, device=device)
        mask = torch.as_tensor(valid[selected], dtype=torch.bool, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(t, f, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(iterations):
                model(t, f, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        rows.append({"scope": "head_only", "device": str(device), "requested_batch_size": requested_batch,
                     "actual_batch_size": len(selected), "iterations": iterations, "warmup": warmup,
                     "latency_ms_per_batch": elapsed * 1000 / iterations,
                     "throughput_samples_per_second": len(selected) * iterations / elapsed,
                     "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                     "input_transfer_included": False, "tokenizer_included": False,
                     "preprocessing_included": False})
    return {"scope": "head_only", "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "measurements": rows,
            "full_pipeline": {"status": "not_measured", "reason": "This suite receives cached codes, not raw EEG plus a runnable tokenizer"}}


def run_experiment(config_path, cache_path, split_path, output_root, smoke=False) -> dict:
    from . import engine
    import numpy as np
    import torch

    config = read_config(config_path)
    cache_path, split_path = Path(cache_path).resolve(), Path(split_path).resolve()
    if not cache_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("A supplied cache file and locked subject split file are required")
    cache, metadata = _unpack_cache(cache_path)
    experiment_id = config["experiment_id"]
    suite_dir = Path(output_root).resolve() / f"experiment_{experiment_id:02d}"
    if (suite_dir / "summary.json").exists():
        raise FileExistsError(f"Suite already exists; choose a new output root: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    design = config.get("experiment", {})
    device = config.get("training", {}).get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    summary: dict[str, Any] = {"experiment_id": experiment_id, "name": config.get("name"),
        "status": "running", **evidence_flags(metadata, smoke),
        "cache_path": str(cache_path), "split_path": str(split_path), "config": config,
        "external_baselines_pending": ["full_CodeBrain", "CBraMod"],
        "notes": ["A cache input is mandatory; real experiments never silently generate synthetic data.",
                  "Smoke runs only reduce epochs and seeds; the supplied subject split is retained."],
        "runs": [], "skipped": []}
    summary_path = suite_dir / "summary.json"
    _write_json(summary_path, summary)
    indices = _test_indices(cache, split_path)
    masks = {}
    variants = {}
    if experiment_id == 2:
        masks, skips = diagnostic_masks(cache, metadata, design)
        summary["skipped"].extend(skips)
        mask_path = suite_dir / "shared_diagnostic_masks.npz"
        np.savez_compressed(mask_path, **masks)
        summary["shared_diagnostic_masks_path"] = str(mask_path)
        for name, variant_path in design.get("reencoded_cache_variants", {}).items():
            path = Path(variant_path)
            if not path.is_absolute():
                path = Path(config_path).resolve().parent / path
            variant, variant_meta = _unpack_cache(path)
            validate_reencoded_variant(cache, variant, metadata, variant_meta)
            variants[name] = (variant, str(path.resolve()))
        if not variants:
            summary["skipped"].append({"condition": "raw_missing", "reason": "No verified raw-missing reencoded cache variants supplied; diagnostic masks are not raw-input results"})
    try:
        for run in expand_runs(config, smoke=smoke):
            run_dir = suite_dir / run["name"]
            report = engine.train_run(cache_path=cache_path, split_path=split_path, run_dir=run_dir,
                model_config=run["model"], graph_config=run["graph"], training_config=run["training"], overwrite=False)
            row = {"name": run["name"], "run_dir": str(run_dir), "design": run, "report": report}
            if experiment_id in (2, 5):
                model = engine.load_run_model(run_dir, cache, device)
                if experiment_id == 2:
                    conditions = []
                    batch_size = run["training"].get("batch_size", 16)
                    clean_metrics, _ = engine.evaluate_model(model, cache, indices, device, batch_size)
                    conditions.append({"condition": "clean", "scope": "cached_clean", "metrics": clean_metrics})
                    for name, mask in masks.items():
                        metrics, _ = engine.evaluate_model(model, cache, indices, device, batch_size, valid_mask_override=mask)
                        conditions.append({"condition": name, "scope": "cached_mask_diagnostic_only", "metrics": metrics})
                    for name, (variant, path) in variants.items():
                        metrics, _ = engine.evaluate_model(model, variant, indices, device, batch_size)
                        conditions.append({"condition": name, "scope": "raw_missing_reencoded", "cache_path": path, "metrics": metrics})
                    row["conditions"] = conditions
                else:
                    row["hardware_profile"] = _profile_head(model, cache, indices, device,
                        design.get("batch_sizes", [1, 8, 32]), design.get("warmup", 5),
                        3 if smoke else design.get("iterations", 30))
            summary["runs"].append(row)
            _write_json(summary_path, summary)
        summary["status"] = "completed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        _write_json(summary_path, summary)
        raise
    _write_json(summary_path, summary)
    return summary
