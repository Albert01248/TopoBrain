"""Audited frozen CodeBrain encoder adapter and raw-input missingness re-encoding."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import file_sha256, validate_cache


def load_prepared(path):
    with np.load(path, allow_pickle=False) as f:
        raw = {k: f[k].copy() for k in f.files}
    key = 'eeg' if 'eeg' in raw else 'raw_eeg'
    if key not in raw:
        raise ValueError('Prepared NPZ requires eeg[N,C,P,200] and source metadata')
    x = raw.pop(key)
    if x.ndim != 4 or min(x.shape) < 1 or x.shape[-1] != 200 or not np.isfinite(x).all():
        raise ValueError('EEG must be finite nonempty [N,C,P,200], already scaled/preprocessed')
    required = {'labels', 'subject_ids', 'channel_ids', 'coords', 'metadata_json'}
    if not required.issubset(raw): raise ValueError('Prepared NPZ missing ' + str(required - raw.keys()))
    meta = json.loads(str(raw.pop('metadata_json').item()))
    if type(meta.get('synthetic')) is not bool or not meta.get('provenance'):
        raise ValueError('Raw metadata must declare synthetic and provenance')
    if raw['labels'].shape != (len(x),) or raw['subject_ids'].shape != (len(x),):
        raise ValueError('Label/subject alignment differs from raw EEG')
    if raw['channel_ids'].shape != (x.shape[1],) or raw['coords'].shape != (x.shape[1], 3):
        raise ValueError('Channel names and geometry must align to EEG channel axis')
    return x.astype(np.float32), raw, meta


def safe_checkpoint(path):
    # Only allow the simple metadata types found in the pinned official checkpoint.
    # Never fall back to arbitrary pickle execution (weights_only=False).
    safe = [argparse.Namespace, np.dtype,
            (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
            np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    if not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Checkpoint model must be a tensor-only state dictionary')
    return state


class FrozenTokenizer:
    def __init__(self, repo_path, checkpoint_path, device='auto'):
        repo = Path(repo_path).resolve()
        source = repo / 'Models' / 'modeling_tokenizer.py'
        if not source.is_file(): raise ValueError('CodeBrain repo lacks Models/modeling_tokenizer.py')
        if 'Models.modeling_tokenizer' in sys.modules:
            origin = Path(sys.modules['Models.modeling_tokenizer'].__file__).resolve()
            if origin != source: raise ValueError('A different Models package is already loaded')
        if str(repo) not in sys.path: sys.path.insert(0, str(repo))
        official = importlib.import_module('Models.modeling_tokenizer')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        state = safe_checkpoint(checkpoint_path)
        self.model = official.tfdual_vq(pretrained=False, as_tokenzer=False,
                                       EEG_size=6000, n_code_t=4096, n_code_f=4096,
                                       code_dim=32, quantize_kmeans_init=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.book_t = self.model.quantize_t.embedding.weight.detach().clone()
        self.book_f = self.model.quantize_f.embedding.weight.detach().clone()
        # Decoders are not used for code extraction and need no GPU memory.
        for name in ('decoder_t', 'decoder_f', 'decode_task_layer_temp',
                     'decode_task_layer_xrec', 'decode_task_layer_angle'):
            setattr(self.model, name, torch.nn.Identity())
        self.model.to(self.device)
        self.book_t = self.book_t.to(self.device)
        self.book_f = self.book_f.to(self.device)
        self.checkpoint_hash = file_sha256(checkpoint_path)
        manifest = repo.parent / 'codebrain_source.json'
        self.source_commit = json.loads(manifest.read_text(encoding='utf-8')).get('source_commit') if manifest.exists() else None
        self.source_hash = file_sha256(source)

    @staticmethod
    def nearest(features, book):
        z = F.normalize(features.float(), p=2, dim=-1)
        d = z.square().sum(dim=-1, keepdim=True) + book.float().square().sum(dim=-1) - 2 * (z @ book.float().T)
        return d.argmin(dim=-1)

    @torch.inference_mode()
    def encode(self, eeg, channel_positions=None):
        x = torch.as_tensor(eeg, dtype=torch.float32, device=self.device)
        b, c, patches, samples = x.shape
        if samples != 200 or patches > 60 or c > 128:
            raise ValueError('Pinned tokenizer supports 200 samples/patch, <=60 patches and <=128 channel positions')
        positions = list(range(1, c + 1)) if channel_positions is None else list(channel_positions)
        if len(positions) != c or any(i < 1 or i > 128 for i in positions):
            raise ValueError('Explicit tokenizer channel positions must align to observed channels')
        h = self.model.encoder(x, input_chans=[0] + positions, return_patch_tokens=True)
        t = self.nearest(self.model.encode_task_layer_t(h), self.book_t)
        f = self.nearest(self.model.encode_task_layer_f(h), self.book_f)
        return t.reshape(b, c, patches), f.reshape(b, c, patches)


def extract_cache(raw_path, repo_path, checkpoint_path, output_path, device='auto', batch_size=1, drop_channels=None):
    output = Path(output_path)
    if output.exists(): raise FileExistsError('Refusing to overwrite existing code cache: ' + str(output))
    if batch_size < 1: raise ValueError('batch_size must be positive')
    eeg, arrays, metadata = load_prepared(raw_path)
    channel_names = arrays['channel_ids'].tolist()
    dropped = list(drop_channels or [])
    if len(set(dropped)) != len(dropped) or not set(dropped).issubset(channel_names):
        raise ValueError('drop_channels must name unique existing channels')
    keep = np.array([name not in dropped for name in channel_names], dtype=bool)
    if not keep.any(): raise ValueError('Cannot remove every input channel')
    original_valid = arrays.get('valid_channel_mask', np.ones(eeg.shape[:2], dtype=bool))
    if not original_valid.all():
        raise ValueError('Prepared source must be fully observed; use explicit drop_channels to re-encode')
    adapter = FrozenTokenizer(repo_path, checkpoint_path, device)
    n, c, patches, _ = eeg.shape
    positions = metadata.get('tokenizer_channel_positions', list(range(1, c + 1)))
    if len(positions) != c: raise ValueError('Raw tokenizer_channel_positions length does not match channels')
    observed_positions = [positions[i] for i in np.flatnonzero(keep)]
    t_all = np.zeros((n, c, patches), dtype=np.uint16)
    f_all = np.zeros_like(t_all)
    if adapter.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(adapter.device)
        torch.cuda.synchronize(adapter.device)
    start = time.perf_counter()
    for first in range(0, n, batch_size):
        t, f = adapter.encode(eeg[first:first+batch_size, keep], observed_positions)
        t_all[first:first+batch_size, keep] = t.cpu().numpy().astype(np.uint16)
        f_all[first:first+batch_size, keep] = f.cpu().numpy().astype(np.uint16)
    if adapter.device.type == 'cuda': torch.cuda.synchronize(adapter.device)
    elapsed = time.perf_counter() - start
    metadata = {**metadata, 'tokenizer_checkpoint_hash': adapter.checkpoint_hash,
                'source_repository_commit': adapter.source_commit,
                'tokenizer_source_sha256': adapter.source_hash,
                'tokenizer_channel_positions': positions,
                'tokenizer_channel_position_policy': 'explicit positions if supplied; otherwise preserved original ordinal indices, not an anatomical mapping',
                'preprocessing_version': metadata.get('preprocessing_version', 'explicit_prepared_input'),
                'split_manifest_version': metadata.get('split_manifest_version', 'subject_ids_locked_v1'),
                'raw_source_sha256': file_sha256(raw_path), 'raw_missing_reencoded': bool(dropped),
                'dropped_channels': dropped, 'missing_code_placeholder': 'zero is ignored ONLY where explicit valid_channel_mask=false',
                'quantizer_adapter': 'official frozen encoder/projections and normalized-L2 nearest code; no EMA/statistics mutation',
                'encoding_device': str(adapter.device), 'encoding_seconds': elapsed,
                'tokenizer_diagnostics': {
                    'temporal_used_codes': int(len(np.unique(t_all[:, keep]))),
                    'frequency_used_codes': int(len(np.unique(f_all[:, keep]))),
                    'temporal_constant_on_this_cache': bool(len(np.unique(t_all[:, keep])) == 1),
                    'frequency_constant_on_this_cache': bool(len(np.unique(f_all[:, keep])) == 1)},
                'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(adapter.device) if adapter.device.type == 'cuda' else None}
    arrays.update(codes_t=t_all, codes_f=f_all,
                  codebook_t=adapter.book_t.cpu().numpy(), codebook_f=adapter.book_f.cpu().numpy(),
                  valid_channel_mask=np.broadcast_to(keep, (n, c)).copy())
    arrays['metadata'] = metadata
    validate_cache(arrays)
    arrays.pop('metadata')
    arrays['metadata_json'] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report = {'status': 'extracted', 'output': str(output.resolve()), 'synthetic': metadata['synthetic'],
              'pilot_only': metadata.get('pilot_only', metadata.get('smoke_only', False)),
              'shape': list(t_all.shape), 'dropped_channels': dropped,
              'seconds': elapsed, 'device': str(adapter.device),
              'code_usage_t': int(len(np.unique(t_all[:, keep]))), 'code_usage_f': int(len(np.unique(f_all[:, keep]))),
              'checkpoint_sha256': adapter.checkpoint_hash,
              'peak_cuda_memory_bytes': metadata['peak_cuda_memory_bytes']}
    output.with_suffix('.extraction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
