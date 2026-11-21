"""Execute all five explicit designs on a supplied, subject-isolated cache."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topobrain.data import load_cache
from topobrain.experiments import run_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True); p.add_argument('--split',required=True)
    p.add_argument('--output',required=True); p.add_argument('--smoke',action='store_true')
    p.add_argument('--neighbors',type=int,default=None)
    p.add_argument('--raw-missing-cache',default=None)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    out=Path(a.output).resolve()
    if out.exists(): raise FileExistsError('Choose a fresh output root; previous evidence is protected')
    out.mkdir(parents=True)
    cache=load_cache(a.cache)
    meta=cache['metadata']
    counts={domain:len(set(cache['codes_'+domain][cache.get('valid_channel_mask',__import__('numpy').ones(cache['codes_t'].shape[:2],dtype=bool))].reshape(-1).tolist())) for domain in ['t','f']}
    blockers=[]
    if counts['t']==1: blockers.append('Temporal codes constant on this cache: no sample-level temporal-code information; do not claim time-frequency complementarity')
    if counts['f']==1: blockers.append('Frequency codes constant on this cache: investigate before scientific interpretation')
    record={'cache':str(Path(a.cache).resolve()),'synthetic':meta['synthetic'],
            'pilot_only':meta.get('pilot_only',meta.get('smoke_only',False)), 'smoke':a.smoke,
            'code_usage':counts,'scientific_claims_blocked':blockers,'experiments':[]}
    config_dir=out/'executed_configs'; config_dir.mkdir()
    for i in range(1,6):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs'/f'experiment_{i}.json').read_text(encoding='utf-8'))
        if a.neighbors is not None: config['graph']['k']=a.neighbors
        if i==2 and a.raw_missing_cache:
            config['experiment']['reencoded_cache_variants']={'template_left_raw_removed':str(Path(a.raw_missing_cache).resolve())}
        path=config_dir/f'experiment_{i}.json'
        path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        print(f'Running experiment {i}/5',flush=True)
        result=run_experiment(path,a.cache,a.split,out,smoke=a.smoke)
        record['experiments'].append({'id':i,'status':result['status'],'runs':len(result['runs']),
                                      'evidence_status':result['evidence_status'],
                                      'summary':str(out/f'experiment_{i:02d}'/'summary.json')})
        (out/'run_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
