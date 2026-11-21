"""Small-batch training with validation-only checkpoint selection.

The module does not fit vocabularies or normalization on any split. Test labels
are used only to validate class coverage and to score the already-selected model.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
from pathlib import Path

import numpy as np

from .data import file_sha256, load_cache, subject_split


def classification_metrics(y, logits=None, predictions=None, classes=None) -> dict:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("metrics require a nonempty one-dimensional label array")
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2 or len(logits) != len(y) or not np.isfinite(logits).all():
            raise ValueError("logits must be finite [N,K]")
        if classes is None:
            classes = np.arange(logits.shape[1])
    elif predictions is None or classes is None:
        raise ValueError("provide logits or predictions plus the fixed class list")
    classes = np.asarray(classes)
    if classes.ndim != 1 or len(classes) < 1 or len(np.unique(classes)) != len(classes):
        raise ValueError("classes must be a unique one-dimensional list")
    if logits is not None and logits.shape[1] != len(classes):
        raise ValueError("logit columns must match classes")
    pred = classes[logits.argmax(axis=1)] if predictions is None else np.asarray(predictions)
    if pred.shape != y.shape or not np.isin(y, classes).all() or not np.isin(pred, classes).all():
        raise ValueError("labels and predictions must belong to the fixed class list")
    index = {int(label): i for i, label in enumerate(classes)}
    yi = np.asarray([index[int(label)] for label in y])
    pi = np.asarray([index[int(label)] for label in pred])
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (yi, pi), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = confusion.diagonal().astype(float)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(classes)), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support > 0)
    accuracy = float(tp.sum() / len(y))
    expected = float(support @ predicted / (len(y) ** 2))
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else None
    nll = None
    if logits is not None:
        shifted = logits - logits.max(axis=1, keepdims=True)
        nll = float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), yi]))
    return {"accuracy": accuracy, "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "kappa": float(kappa) if kappa is not None else None, "nll": nll,
            "confusion_matrix": confusion.tolist(), "support": support.tolist(),
            "classes": classes.astype(int).tolist(), "n_samples": len(y),
            "macro_f1_definition": "all fixed training classes; absent classes have F1=0",
            "balanced_accuracy_definition": "mean recall over classes present in this evaluation"}


def _device(requested="auto"):
    import torch
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(requested)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return result


def _validate_indices(cache, indices):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
        raise ValueError("indices must be a nonempty integer vector")
    if np.any(indices < 0) or np.any(indices >= len(cache["labels"])):
        raise ValueError("indices lie outside the cache")
    return indices


def _mask(cache, override=None):
    shape = cache["codes_t"].shape[:2]
    value = (cache.get("valid_channel_mask", np.ones(shape, dtype=bool))
             if override is None else np.asarray(override))
    if value.shape != shape or value.dtype.kind != "b":
        raise ValueError("valid mask override must be boolean [N,C] for the full cache")
    original = cache.get("valid_channel_mask")
    if original is not None:
        value = value & original
    return value


def evaluate_model(model, cache: dict, indices, device="cpu", batch_size=32,
                   valid_mask_override=None) -> tuple[dict, dict]:
    """Evaluate selected examples; override can hide, never resurrect, channels."""
    import torch
    indices = _validate_indices(cache, indices)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = _device(device)
    observed = _mask(cache, valid_mask_override)
    if not observed[indices].any(axis=1).all():
        raise ValueError("evaluation cannot contain an example with no observed channels")
    model = model.to(device)
    was_training = model.training
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            t = torch.as_tensor(cache["codes_t"][ids].astype(np.int64), device=device)
            f = torch.as_tensor(cache["codes_f"][ids].astype(np.int64), device=device)
            mask = torch.as_tensor(observed[ids], dtype=torch.bool, device=device)
            logits = model(t, f, valid_mask=mask)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise ValueError("model must return a [batch,num_classes] logits tensor")
            output.append(logits.detach().float().cpu().numpy())
    model.train(was_training)
    logits = np.concatenate(output)
    y = cache["labels"][indices]
    subjects = cache["subject_ids"][indices]
    metrics = classification_metrics(y, logits=logits)
    predictions = {"logits": logits, "labels": y.copy(),
                   "predictions": logits.argmax(axis=1),
                   "subject_ids": subjects.copy(), "subjects": subjects.copy(),
                   "indices": indices.copy()}
    for name in ("sample_ids", "record_ids"):
        if name in cache:
            predictions[name] = cache[name][indices].copy()
    return metrics, predictions


def _seed_everything(seed: int, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)
    if deterministic and hasattr(torch.backends, "cuda"):
        # Small local sequences do not need nondeterministic memory-efficient SDP.
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _structure_fingerprint(cache):
    digest = hashlib.sha256()
    for name in ("codebook_t", "codebook_f", "coords", "channel_ids"):
        value = np.ascontiguousarray(cache[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _evidence_fields(metadata):
    """Propagate source evidence limits; short training never promotes a pilot."""
    for name in ("synthetic", "pilot_only", "smoke_only"):
        if name in metadata and type(metadata[name]) is not bool:
            raise ValueError(f"metadata {name} must be a JSON boolean")
    synthetic = metadata["synthetic"]
    smoke_only = metadata.get("smoke_only", False)
    pilot_only = metadata.get("pilot_only", False) or smoke_only
    if synthetic:
        status = "SYNTHETIC ENGINEERING CHECK ONLY; NOT EEG PAPER RESULTS"
    elif pilot_only:
        status = "REAL EEG PIPELINE PILOT ONLY; NOT POPULATION/BENCHMARK EVIDENCE"
    else:
        status = "REAL EEG RESEARCH RUN; NOT INDEPENDENTLY VALIDATED"
    return {"synthetic": synthetic, "pilot_only": pilot_only,
            "smoke_only": smoke_only, "evidence_status": status}


def _source_report_fields(metadata):
    """Keep provenance/label/asset diagnostics visible beside reported metrics."""
    return {"cache_metadata": metadata, "source_provenance": metadata["provenance"],
            "label_mapping": metadata.get("label_mapping"),
            "diagnostics": metadata.get("diagnostics", metadata.get("tokenizer_diagnostics", {}))}


def train_run(cache_path, split_path, run_dir, model_config: dict, graph_config: dict,
              training_config: dict, overwrite=False) -> dict:
    """Train on train, select only by val macro-F1, then evaluate test once."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from .graphs import build_graph
    from .model import ModelConfig, TopoBrain

    options = {"epochs": 10, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4,
               "seed": 0, "device": "auto", "patience": 5, "amp": False,
               "grad_clip": 1.0, "num_workers": 0, "deterministic": True}
    unknown = set(training_config) - options.keys()
    if unknown:
        raise ValueError("unknown training options: " + ", ".join(sorted(unknown)))
    options.update(training_config)
    for key in ("epochs", "batch_size", "patience"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if options["lr"] <= 0 or options["weight_decay"] < 0 or options["grad_clip"] < 0:
        raise ValueError("invalid learning rate, weight decay, or gradient clipping")
    cache = load_cache(cache_path)
    evidence = _evidence_fields(cache["metadata"])
    source_fields = _source_report_fields(cache["metadata"])
    specification = json.loads(Path(split_path).read_text(encoding="utf-8-sig"))
    groups = subject_split(cache["subject_ids"], specification)
    classes = np.unique(cache["labels"][groups["train"]])
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) < 2:
        raise ValueError("training labels must contain every consecutive class 0..K-1, K>=2")
    if not np.isin(cache["labels"], classes).all():
        raise ValueError("train labels must cover every validation/test class")
    model_options = dict(model_config)
    if "num_classes" in model_options and model_options["num_classes"] != len(classes):
        raise ValueError("num_classes does not match training label classes")
    model_options["num_classes"] = int(len(classes))
    config = ModelConfig(**model_options)
    graph_options = dict(graph_config)
    adjacency = build_graph(cache["coords"], **graph_options)
    run_dir = Path(run_dir)
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"run directory already exists; set overwrite explicitly: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(options["seed"]), options["deterministic"])
    device = _device(options["device"])
    model = TopoBrain(codebook_t=torch.as_tensor(cache["codebook_t"], dtype=torch.float32),
                      codebook_f=torch.as_tensor(cache["codebook_f"], dtype=torch.float32),
                      adjacency=torch.as_tensor(adjacency, dtype=torch.float32), config=config).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=options["lr"], weight_decay=options["weight_decay"])
    amp_enabled = bool(options["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    observed = _mask(cache)

    class Samples(Dataset):
        def __len__(self):
            return len(groups["train"])

        def __getitem__(self, item):
            index = groups["train"][item]
            return (torch.as_tensor(cache["codes_t"][index].astype(np.int64)),
                    torch.as_tensor(cache["codes_f"][index].astype(np.int64)),
                    torch.as_tensor(observed[index], dtype=torch.bool),
                    torch.tensor(int(cache["labels"][index]), dtype=torch.long))

    # Zero workers is portable on Windows and avoids copying a large cache into subprocesses.
    if options["num_workers"] != 0:
        raise ValueError("num_workers must remain 0 for the in-memory cache dataset")
    generator = torch.Generator().manual_seed(int(options["seed"]))
    loader = DataLoader(Samples(), batch_size=options["batch_size"], shuffle=True,
                        num_workers=0, generator=generator,
                        pin_memory=(device.type == "cuda"))
    manifest = {"status": "training", **evidence, **source_fields,
                "cache_path": str(Path(cache_path).resolve()), "cache_sha256": file_sha256(cache_path),
                "cache_structure_sha256": _structure_fingerprint(cache),
                "split_path": str(Path(split_path).resolve()), "split_sha256": file_sha256(split_path),
                "subjects": specification, "sample_counts": {k: len(v) for k, v in groups.items()},
                "model_config": model_options, "graph_config": graph_options,
                "training_config": options, "actual_device": str(device),
                "amp_enabled": amp_enabled, "checkpoint_selection": "validation macro_f1 only; earliest epoch wins ties",
                "trainable_parameters": sum(p.numel() for p in trainable),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "classes": classes.astype(int).tolist()}
    _json_write(run_dir / "manifest.json", manifest)
    history, best_state, best_score, best_epoch, stale = [], None, -float("inf"), 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, options["epochs"] + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for t, f, mask, y in loader:
            t, f, mask, y = (value.to(device, non_blocking=True) for value in (t, f, mask, y))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(t, f, valid_mask=mask)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            scaler.scale(loss).backward()
            if options["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, options["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)
        val_metrics, _ = evaluate_model(model, cache, groups["val"], device, options["batch_size"])
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "val": val_metrics}
        history.append(row)
        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= options["patience"]:
            break
    model.load_state_dict(best_state)
    checkpoint = {"state_dict": best_state, "model_config": model_options,
                  "graph_config": graph_options, "classes": classes.astype(int).tolist(),
                  "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
                  "history": history, "cache_sha256": manifest["cache_sha256"],
                  "cache_structure_sha256": manifest["cache_structure_sha256"],
                  **evidence, **source_fields}
    torch.save(checkpoint, run_dir / "best.pt")
    # Test is evaluated only after the validation-selected state has been fixed.
    val_metrics, _ = evaluate_model(model, cache, groups["val"], device, options["batch_size"])
    test_metrics, predictions = evaluate_model(model, cache, groups["test"], device, options["batch_size"])
    np.savez_compressed(run_dir / "predictions.npz", **predictions)
    report = {**evidence, **source_fields,
              "best_epoch": best_epoch, "validation": val_metrics, "test": test_metrics,
              "history": history, "test_evaluations_during_selection": 0,
              "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
              "run_dir": str(run_dir.resolve())}
    _json_write(run_dir / "metrics.json", report)
    manifest["status"] = "complete"
    manifest["best_epoch"] = best_epoch
    _json_write(run_dir / "manifest.json", manifest)
    return report


def load_run_model(run_dir, cache, device="cpu"):
    import torch
    from .graphs import build_graph
    from .model import ModelConfig, TopoBrain
    device = _device(device)
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("cache_structure_sha256") != _structure_fingerprint(cache):
        raise ValueError("cache codebooks, coordinates, or channel order differ from the saved run")
    adjacency = build_graph(cache["coords"], **checkpoint["graph_config"])
    model = TopoBrain(codebook_t=torch.as_tensor(cache["codebook_t"], dtype=torch.float32),
                      codebook_f=torch.as_tensor(cache["codebook_f"], dtype=torch.float32),
                      adjacency=torch.as_tensor(adjacency, dtype=torch.float32),
                      config=ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def paired_subject_bootstrap(y, predA, predB, subjects, classes, repeats=2000,
                             seed=0, metric="macro_f1") -> dict:
    """Paired cluster bootstrap of pooled metric(A)-metric(B), sampling people.

    Every selected person contributes all their examples each time they are drawn.
    This retains within-person dependence; the statistic itself is sample-pooled.
    """
    y, a, b, subjects = map(np.asarray, (y, predA, predB, subjects))
    if y.ndim != 1 or len(y) == 0 or any(v.shape != y.shape for v in (a, b, subjects)):
        raise ValueError("paired arrays must have matching nonempty [N] shapes")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if metric not in ("macro_f1", "balanced_accuracy", "accuracy", "kappa"):
        raise ValueError("unsupported bootstrap metric")
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("subject bootstrap requires at least two subjects")
    groups = [np.flatnonzero(subjects == name) for name in unique]
    def difference(ids):
        ma = classification_metrics(y[ids], predictions=a[ids], classes=classes)[metric]
        mb = classification_metrics(y[ids], predictions=b[ids], classes=classes)[metric]
        return None if ma is None or mb is None else ma - mb
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        delta = difference(ids)
        if delta is not None:
            estimates.append(delta)
    if not estimates:
        raise ValueError("bootstrap statistic is undefined for every resample")
    return {"metric": metric, "difference_A_minus_B": difference(np.arange(len(y))),
            "ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
            "n_subjects": len(unique), "repeats": repeats, "valid_repeats": len(estimates),
            "seed": int(seed), "unit": "subject", "statistic": "sample-pooled metric; paired subject resampling"}
