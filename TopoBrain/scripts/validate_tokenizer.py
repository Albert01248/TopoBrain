"""Compare the frozen extraction adapter to the official encoder/quantizer path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topobrain.tokenizer import FrozenTokenizer, load_prepared
from topobrain.data import load_cache


def validate(raw_path, cache_path, repo, checkpoint, output):
    torch.set_num_threads(2)
    x, arrays, metadata = load_prepared(raw_path)
    cache = load_cache(cache_path)
    adapter = FrozenTokenizer(repo, checkpoint)
    positions = metadata.get('tokenizer_channel_positions', list(range(1, x.shape[1]+1)))
    batch = x[:min(2, len(x))]
    before_t, before_f = adapter.book_t.clone(), adapter.book_f.clone()
    with torch.inference_mode():
        ours_t, ours_f = adapter.encode(batch, positions)
        source = adapter.model.encode(torch.as_tensor(batch, device=adapter.device), input_chans=[0]+positions)
        official_t = source[1].reshape(ours_t.shape)
        official_f = source[4].reshape(ours_f.shape)
    equality = {'temporal': bool(torch.equal(ours_t, official_t)), 'frequency': bool(torch.equal(ours_f, official_f))}
    unchanged = bool(torch.equal(before_t, adapter.book_t) and torch.equal(before_f, adapter.book_f))
    if not all(equality.values()) or not unchanged: raise RuntimeError('Official quantizer parity/frozen book validation failed')
    branches = {}
    for name in ('t', 'f'):
        ids, count = np.unique(cache['codes_'+name], return_counts=True)
        book = cache['codebook_'+name]
        norm = np.linalg.norm(book, axis=1)
        branches[name] = {'used_codes': len(ids), 'most_frequent_code': int(ids[count.argmax()]),
                          'largest_code_fraction': float(count.max()/count.sum()),
                          'collapsed_on_this_cache': bool(len(ids)==1),
                          'codebook_rows': len(book), 'unique_codebook_rows': len(np.unique(book, axis=0)),
                          'codebook_norm_min_max': [float(norm.min()),float(norm.max())]}
    report = {'status': 'adapter_matches_official_quantizer', 'adapter_official_parity': equality,
              'frozen_codebooks_unchanged': unchanged, 'synthetic': metadata['synthetic'],
              'pilot_only': metadata.get('pilot_only',metadata.get('smoke_only',False)),
              'cache_shape': list(cache['codes_t'].shape), 'code_usage': branches,
              'interpretation': 'Constant IDs are an observed asset/input diagnostic, not proof that temporal information is absent in all EEG. Do not hide or fabricate this outcome.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True); p.add_argument('--cache',required=True)
    p.add_argument('--repo',default='vendor/CodeBrain'); p.add_argument('--checkpoint',default='weights/CodeBrain_Tokenizer.pth')
    p.add_argument('--output',required=True)
    a=p.parse_args(); validate(a.raw,a.cache,a.repo,a.checkpoint,a.output)
