"""Read-only, bounded pilot diagnostics for frozen CodeBrain branch features.

No training, network access, amplitude adjustment, architecture search, or weight
updates. The output contains summaries, never full activations or raw signals.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topobrain.tokenizer import FrozenTokenizer, load_prepared


QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)


def quantiles(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("diagnostic array is empty or nonfinite")
    return {str(q): float(value) for q, value in zip(QUANTILES, np.quantile(x, QUANTILES))}


def feature_summary(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    if not np.isfinite(x).all():
        raise ValueError("nonfinite encoder/projection activation")
    norms = np.linalg.norm(x, axis=-1)
    unit = x / np.maximum(norms[:, None], 1e-12)
    unit_mean = unit.mean(axis=0)
    return {
        "token_count": len(x), "feature_dimension": x.shape[1],
        "rms": float(np.sqrt(np.mean(x * x))),
        "cross_token_std_rms": float(np.sqrt(np.mean(np.var(x, axis=0)))),
        "cross_token_std_per_dimension_quantiles": quantiles(x.std(axis=0)),
        "token_l2_norm_quantiles": quantiles(norms),
        "zero_norm_token_count": int(np.count_nonzero(norms <= 1e-12)),
        "unit_vector_mean_norm": float(np.linalg.norm(unit_mean)),
        "unit_vector_mean_squared_dispersion": float(np.mean(np.sum((unit - unit_mean) ** 2, axis=1))),
        "dispersion_note": "Unit-vector dispersion near zero indicates nearly constant directions; raw norm changes alone do not change normalized-L2 code assignment."
    }


def code_distribution(values):
    ids, counts = np.unique(np.asarray(values).reshape(-1), return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {"used_codes": len(ids), "token_count": int(counts.sum()),
            "largest_code_fraction": float(counts.max() / counts.sum()),
            "counts_by_code": {str(int(ids[i])): int(counts[i]) for i in order}}


def safe_checkpoint_metadata(path):
    # Match the existing safe loader; do not permit arbitrary pickle execution.
    safe = [argparse.Namespace, np.dtype,
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = checkpoint.get("args", {})
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    allowed = ("model", "nhead", "n_layer", "input_size", "in_dim", "out_dim", "d_model", "seq_len",
               "codebook_n_emd_t", "codebook_n_emd_f", "codebook_emd_dim", "quantize_kmeans_init")
    result = {}
    if isinstance(args, dict):
        for key in allowed:
            value = args.get(key)
            if value is None or isinstance(value, (str, bool, int, float)):
                result[key] = value
            elif isinstance(value, np.generic):
                result[key] = value.item()
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, (int, float, np.generic)):
        result["saved_epoch"] = int(epoch)
    return result


def diagnose(raw, repo, checkpoint, output, max_samples=6, device="auto"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite existing diagnostic output")
    if not 1 <= max_samples <= 9:
        raise ValueError("max-samples must be in [1,9] for this bounded pilot diagnostic")
    torch.set_num_threads(2)
    eeg, arrays, metadata = load_prepared(raw)
    selected = np.linspace(0, len(eeg) - 1, min(max_samples, len(eeg)), dtype=int)
    sample_ids = arrays.get("sample_ids", np.asarray([f"row_{i}" for i in range(len(eeg))]))
    positions = list(metadata.get("tokenizer_channel_positions", range(1, eeg.shape[1] + 1)))
    # Official constructors print full configs. Suppress those non-diagnostic logs;
    # record explicit runtime fields and hashes below instead.
    with contextlib.redirect_stdout(io.StringIO()):
        adapter = FrozenTokenizer(repo, checkpoint, device=device)
    checkpoint_info = safe_checkpoint_metadata(checkpoint)
    runtime_heads = int(adapter.model.encoder.blocks[0].attn.num_heads)
    books = {"temporal": adapter.book_t, "frequency": adapter.book_f}
    book_snapshots = {key: book.detach().cpu().clone() for key, book in books.items()}
    book_summary = {}
    for key, tensor in book_snapshots.items():
        book = tensor.numpy()
        book_summary[key] = {"shape": list(book.shape), "norm_quantiles": quantiles(np.linalg.norm(book, axis=1)),
                             "unique_rows": int(len(np.unique(book, axis=0))), "finite": bool(np.isfinite(book).all())}
    raw_records, feature_records = [], []
    collected_h = []
    collected = {key: {"projection": [], "codes": [], "top1": [], "margin": []} for key in books}
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
    with torch.inference_mode():
        for index in selected:
            prepared = eeg[index:index + 1]
            raw_records.append({"sample_index": int(index), "sample_id": str(sample_ids[index]),
                                "shape": list(prepared.shape), "finite": bool(np.isfinite(prepared).all()),
                                "rms_in_prepared_units": float(np.sqrt(np.mean(prepared.astype(np.float64) ** 2))),
                                "abs_quantiles_in_prepared_units": quantiles(np.abs(prepared))})
            x = torch.as_tensor(prepared, dtype=torch.float32, device=adapter.device)
            h = adapter.model.encoder(x, input_chans=[0] + positions, return_patch_tokens=True)
            h_array = h.detach().cpu().numpy().reshape(-1, h.shape[-1])
            collected_h.append(h_array)
            current = {"sample_index": int(index), "sample_id": str(sample_ids[index]), "encoder": feature_summary(h_array), "branches": {}}
            for key, projection in (("temporal", adapter.model.encode_task_layer_t), ("frequency", adapter.model.encode_task_layer_f)):
                projected = projection(h)
                z = F.normalize(projected.float(), p=2, dim=-1)
                book = books[key].float()
                distances = z.square().sum(dim=-1, keepdim=True) + book.square().sum(dim=-1) - 2 * (z @ book.T)
                top2 = torch.topk(distances, k=2, dim=-1, largest=False, sorted=True).values
                ids = distances.argmin(dim=-1)
                projection_array = projected.detach().cpu().numpy().reshape(-1, projected.shape[-1])
                ids_array = ids.detach().cpu().numpy().reshape(-1)
                top1_array = top2[..., 0].detach().cpu().numpy().reshape(-1)
                margin_array = (top2[..., 1] - top2[..., 0]).detach().cpu().numpy().reshape(-1)
                current["branches"][key] = {"projection": feature_summary(projection_array), "top1_codes": code_distribution(ids_array),
                                             "nearest_distance_quantiles": quantiles(top1_array), "top2_minus_top1_distance_quantiles": quantiles(margin_array)}
                for field, value in (("projection", projection_array), ("codes", ids_array), ("top1", top1_array), ("margin", margin_array)):
                    collected[key][field].append(value)
            feature_records.append(current)
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
    unchanged = {key: bool(torch.equal(before, books[key].detach().cpu())) for key, before in book_snapshots.items()}
    if not all(unchanged.values()):
        raise RuntimeError("unexpected codebook mutation during read-only diagnosis")
    aggregate = {"encoder": feature_summary(np.concatenate(collected_h, axis=0)), "branches": {}}
    for key, values in collected.items():
        aggregate["branches"][key] = {"projection": feature_summary(np.concatenate(values["projection"], axis=0)),
                                       "top1_codes": code_distribution(np.concatenate(values["codes"])),
                                       "nearest_distance_quantiles": quantiles(np.concatenate(values["top1"])),
                                       "top2_minus_top1_distance_quantiles": quantiles(np.concatenate(values["margin"]))}
    declared_head = checkpoint_info.get("nhead")
    report = {"status": "pilot_feature_diagnostic_only", "synthetic": metadata["synthetic"], "pilot_only": True,
              "is_performance_evaluation": False, "training_performed": False, "input_rescaled": False,
              "raw_shape": list(eeg.shape), "selected_sample_indices": selected.tolist(),
              "selected_sample_ids": [str(sample_ids[i]) for i in selected], "selection": "deterministic linspace over source array indices",
              "input_unit_metadata": metadata.get("preprocessing", {}), "input_position_indices": positions,
              "position_policy": metadata.get("tokenizer_channel_position_policy", "explicit raw indices if present; otherwise ordinal channel indices, not anatomical mapping"),
              "device": str(adapter.device), "checkpoint_sha256": adapter.checkpoint_hash, "source_commit": adapter.source_commit,
              "source_sha256": adapter.source_hash, "checkpoint_arguments_subset": checkpoint_info,
              "runtime": {"encoder_heads_per_block": [int(block.attn.num_heads) for block in adapter.model.encoder.blocks],
                          "encoder_layers": len(adapter.model.encoder.blocks), "position_embedding_shape": list(adapter.model.encoder.pos_embed.shape),
                          "time_embedding_shape": list(adapter.model.encoder.time_embed.shape)},
              "head_metadata_discrepancy": None if declared_head is None else int(declared_head) != runtime_heads,
              "head_discrepancy_interpretation": "A metadata/runtime difference alone does not prove an incorrect architecture. Current official get_model does not pass args.nhead to the factory; state-dict shape parity does not establish training-time head semantics. No alternate head count was tested or selected.",
              "codebooks": book_summary, "codebooks_unchanged": unchanged,
              "raw_samples": raw_records, "per_sample_features": feature_records, "aggregate_features": aggregate,
              "interpretation_boundary": "Constant IDs on a small pilot may reflect input/source/checkpoint compatibility or directional quantization behavior. This summary is not evidence that temporal EEG information is absent, nor a clinical or performance result."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print("Saved bounded pilot feature diagnostic; no training or performance claims.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/faced_pilot/raw.npz")
    parser.add_argument("--repo", default="vendor/CodeBrain")
    parser.add_argument("--checkpoint", default="weights/CodeBrain_Tokenizer.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    diagnose(args.raw, args.repo, args.checkpoint, args.output, args.max_samples, args.device)


if __name__ == "__main__":
    main()
