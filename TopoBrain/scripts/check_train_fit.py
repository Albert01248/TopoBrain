"""Train-only frequency-head capacity check, never a held-out evaluation.

The fixed identity graph retains the real model's spatial-layer parameters but
has no inter-channel edges. This diagnostic asks whether optimization can fit
the supplied TRAIN examples. Its output is not population or benchmark evidence.
No model weights are saved and no validation/test metric is computed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_train_fit(cache_path, split_path, output_path, steps=200, lr=0.003,
                    device="auto", seed=42):
    """Optimize on train labels only, in batches of at most 32 examples."""
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("output must have an explicit .json suffix")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing diagnostic: {output}")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")

    from topobrain.data import file_sha256, load_cache, subject_split
    from topobrain.engine import _device, _evidence_fields, _seed_everything, classification_metrics
    from topobrain.graphs import build_graph
    from topobrain.model import ModelConfig, TopoBrain
    import torch

    cache = load_cache(cache_path)
    split = json.loads(Path(split_path).read_text(encoding="utf-8-sig"))
    # Subject IDs from all splits are used only for strict disjointness/coverage
    # checks. Labels and EEG/code examples below are indexed ONLY by train IDs.
    indices = subject_split(cache["subject_ids"], split)["train"]
    y_original = cache["labels"][indices]
    label_values = np.unique(y_original)
    if len(label_values) < 2:
        raise ValueError("capacity check requires at least two label classes in train")
    y_train = np.searchsorted(label_values, y_original).astype(np.int64)
    t_train = cache["codes_t"][indices]
    f_train = cache["codes_f"][indices]
    mask_train = (cache["valid_channel_mask"][indices]
                  if "valid_channel_mask" in cache else np.ones(t_train.shape[:2], dtype=bool))
    if not mask_train.any(axis=1).all():
        raise ValueError("every training example needs at least one observed channel")
    n_train = len(indices)
    batch_size = min(32, n_train)

    _seed_everything(int(seed), deterministic=True)
    target_device = _device(device)
    config = ModelConfig(num_classes=len(label_values), domain="frequency",
                         hidden_per_branch=32, dropout=0.0, graph_depth=1,
                         temporal_window=3)
    graph_config = {"kind": "identity", "k": 4, "seed": int(seed)}
    adjacency = build_graph(cache["coords"], **graph_config)
    model = TopoBrain(codebook_t=torch.as_tensor(cache["codebook_t"], dtype=torch.float32),
                      codebook_f=torch.as_tensor(cache["codebook_f"], dtype=torch.float32),
                      adjacency=torch.as_tensor(adjacency, dtype=torch.float32),
                      config=config).to(target_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(lr), weight_decay=0.0)

    def tensors(local_indices):
        # Only this small CPU batch is copied to the GPU; the full code cache
        # never becomes a device tensor, including when train contains >32 rows.
        return (torch.as_tensor(t_train[local_indices].astype(np.int64), device=target_device),
                torch.as_tensor(f_train[local_indices].astype(np.int64), device=target_device),
                torch.as_tensor(mask_train[local_indices], dtype=torch.bool, device=target_device),
                torch.as_tensor(y_train[local_indices], dtype=torch.long, device=target_device))

    def measure_train(step):
        model.eval()
        pieces = []
        with torch.inference_mode():
            for begin in range(0, n_train, batch_size):
                local = np.arange(begin, min(begin + batch_size, n_train))
                t, f, mask, _ = tensors(local)
                pieces.append(model(t, f, valid_mask=mask).float().cpu().numpy())
        logits = np.concatenate(pieces)
        measured = classification_metrics(y_train, logits=logits, classes=np.arange(len(label_values)))
        return {"step": int(step), "train_loss": measured["nll"],
                "train_accuracy": measured["accuracy"],
                "train_macro_f1": measured["macro_f1"]}

    initial = measure_train(0)
    history = [initial]
    first_95 = 0 if initial["train_accuracy"] >= 0.95 else None
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_train)
    cursor = 0
    final = initial
    actual_steps = 0
    for step in range(1, steps + 1):
        if n_train <= batch_size:
            local = np.arange(n_train)
        else:
            if cursor >= n_train:
                order, cursor = rng.permutation(n_train), 0
            local = order[cursor:cursor + batch_size]
            cursor += len(local)
        t, f, mask, y = tensors(local)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(t, f, valid_mask=mask)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss; no diagnostic result was saved")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        # Every threshold check measures ALL training rows, batched on device.
        # There is no early stopping or hyperparameter choice on held-out data.
        final = measure_train(step)
        actual_steps = step
        if first_95 is None and final["train_accuracy"] >= 0.95:
            first_95 = step
        stopping = step >= 50 and final["train_accuracy"] >= 0.95
        if step % 20 == 0 or step == steps or stopping:
            history.append(final)
            print(json.dumps(final, ensure_ascii=False), flush=True)
        if stopping:
            break

    source_evidence = _evidence_fields(cache["metadata"])
    report = {**source_evidence,
              "criterion": "capacity_sanity_check", "status": "completed",
              "evidence_status": ("SYNTHETIC TRAIN-ONLY CAPACITY CHECK; NOT EEG PAPER RESULTS"
                                  if source_evidence["synthetic"] else
                                  "REAL EEG TRAIN-ONLY CAPACITY CHECK; NOT GENERALIZATION/BENCHMARK EVIDENCE"),
              "source_evidence_status": source_evidence["evidence_status"],
              "heldout_examples_used": 0, "validation_examples_used": 0, "test_examples_used": 0,
              "cache_path": str(Path(cache_path).resolve()), "cache_sha256": file_sha256(cache_path),
              "split_path": str(Path(split_path).resolve()), "split_sha256": file_sha256(split_path),
              "source_provenance": cache["metadata"]["provenance"],
              "train_subjects": sorted(set(cache["subject_ids"][indices].tolist())),
              "n_train_examples": n_train, "train_label_values": label_values.astype(int).tolist(),
              "label_index_policy": "consecutive class indices fitted from TRAIN labels only",
              "model_config": asdict(config), "graph_config": graph_config,
              "optimizer": {"name": "AdamW", "lr": float(lr), "weight_decay": 0.0, "grad_clip_norm": 1.0},
              "seed": int(seed), "device": str(target_device), "batch_size_max": 32,
              "actual_batch_size": batch_size, "steps_requested": steps, "actual_steps": actual_steps,
              "early_stop_rule": "current full-training accuracy >=0.95, after at least 50 optimizer steps",
              "threshold_check_frequency_steps": 1, "reached_95_step": first_95,
              "initial": initial, "final": final, "history": history,
              "weights_saved": False,
              "interpretation": ("High training accuracy would show only that this head/optimizer can fit these training examples. "
                                 "Failure can reflect insufficient steps, representation, readout capacity, or optimization and requires diagnosis. "
                                 "This check does not establish cross-subject performance, frequency-domain superiority, or useful spatial routing.")}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also refuses an output created by another process while
    # the diagnostic was running. No official model checkpoints are overwritten.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/faced_pilot/codes.npz"))
    parser.add_argument("--split", type=Path, default=Path("data/faced_pilot/split.json"))
    parser.add_argument("--output", type=Path, default=Path("results/next_step_frequency_train_fit.json"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = check_train_fit(args.cache, args.split, args.output, args.steps, args.lr, args.device, args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "criterion": report["criterion"],
                      "heldout_examples_used": report["heldout_examples_used"],
                      "actual_steps": report["actual_steps"], "initial": report["initial"],
                      "final": report["final"], "reached_95_step": report["reached_95_step"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
