"""Small command-line interface with no implicit data generation or downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _print_json(value):
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(f"Cannot encode {type(item).__name__} as JSON")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=convert))


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m topobrain",
        description="Frozen EEG-code topology adaptation. Synthetic runs are engineering checks only.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one of the five explicit experiment configurations")
    run.add_argument("--experiment", required=True, type=Path, help="Experiment JSON configuration")
    run.add_argument("--cache", required=True, type=Path, help="Validated code-cache NPZ")
    run.add_argument("--split", required=True, type=Path, help="Locked train/val/test subject-list JSON")
    run.add_argument("--output", required=True, type=Path, help="Fresh output root; existing suites are protected")
    run.add_argument("--smoke", action="store_true", help="One epoch/one seed, retaining the supplied subject split")
    inspect = commands.add_parser("inspect", help="Validate and summarize a cache without loading CUDA")
    inspect.add_argument("--cache", required=True, type=Path)
    inspect.add_argument("--split", type=Path, help="Optionally validate the locked subject split")
    extract = commands.add_parser("extract", help="Extract frozen CodeBrain tokens from explicitly prepared EEG")
    extract.add_argument("--raw", required=True, type=Path, help="Prepared EEG NPZ, not an EDF download")
    extract.add_argument("--repo", required=True, type=Path, help="Local audited CodeBrain checkout")
    extract.add_argument("--checkpoint", required=True, type=Path, help="Local tokenizer checkpoint")
    extract.add_argument("--output", required=True, type=Path, help="Output code-cache NPZ")
    extract.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--drop-channels", default="", help="Comma-separated exact channel IDs; re-encode raw-input missingness")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .experiments import run_experiment
            result = run_experiment(args.experiment, args.cache, args.split, args.output, smoke=args.smoke)
            _print_json({"status": result["status"], "experiment_id": result["experiment_id"],
                         "synthetic": result["synthetic"], "smoke": result["smoke"],
                         "pilot_only": result.get("pilot_only", False),
                         "scientific_results": result.get("scientific_results", False),
                         "completed_runs": len(result["runs"]),
                         "summary_path": str((args.output / f"experiment_{result['experiment_id']:02d}" / "summary.json").resolve()),
                         "external_baselines_pending": result["external_baselines_pending"],
                         "evidence": result.get("evidence_status", "Input-specific run; independently audit evidence limits")})
        elif args.command == "inspect":
            import numpy as np
            from .data import load_cache, subject_split
            cache = load_cache(args.cache)
            labels, counts = np.unique(cache["labels"], return_counts=True)
            result = {"cache_path": str(args.cache.resolve()), "valid": True,
                      "synthetic": cache["metadata"]["synthetic"],
                      "code_shape": list(cache["codes_t"].shape),
                      "codebook_t_shape": list(cache["codebook_t"].shape),
                      "codebook_f_shape": list(cache["codebook_f"].shape),
                      "subjects": int(len(np.unique(cache["subject_ids"]))),
                      "class_counts": {str(int(k)): int(v) for k, v in zip(labels, counts)},
                      "channel_ids": cache["channel_ids"].tolist(),
                      "metadata": cache["metadata"]}
            if args.split:
                specification = json.loads(args.split.read_text(encoding="utf-8"))
                indices = subject_split(cache["subject_ids"], specification)
                result["split"] = {name: {"samples": len(ids), "subjects": len(specification[name])}
                                   for name, ids in indices.items()}
            _print_json(result)
        else:
            from .tokenizer import extract_cache
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            channels = [name.strip() for name in args.drop_channels.split(",") if name.strip()]
            if args.drop_channels.strip() and not channels:
                raise ValueError("--drop-channels must contain at least one exact channel ID")
            if len(channels) != len(set(channels)):
                raise ValueError("--drop-channels contains duplicate IDs")
            result = extract_cache(raw_path=args.raw, repo_path=args.repo, checkpoint_path=args.checkpoint,
                                   output_path=args.output, device=args.device, batch_size=args.batch_size,
                                   drop_channels=channels or None)
            if isinstance(result, dict):
                _print_json(result)
            else:
                _print_json({"status": "completed", "output_path": str(args.output.resolve())})
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"topobrain: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
