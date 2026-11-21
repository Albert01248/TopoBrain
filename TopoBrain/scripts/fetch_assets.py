"""Fetch pinned official CodeBrain assets and an optional public EEG interface sample."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '22d350caf68246d2fda4f630ef837420db3fb130'
TOKENIZER_SHA256 = 'e9560b670d64ea4712fd99a48dc2131326b919744c7d3eb504cf57b1ef3af999'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def download(url, target, expected=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected is None or digest(target) == expected):
        return {'url': url, 'path': str(target.relative_to(ROOT)), 'sha256': digest(target), 'bytes': target.stat().st_size}
    part = target.with_suffix(target.suffix + '.part')
    print('Downloading:', target.name, flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'EEGResearch/0.1'})
    with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as f:
        shutil.copyfileobj(response, f, length=1024*1024)
    actual = digest(part)
    if expected and actual != expected:
        raise ValueError('Official asset checksum mismatch: ' + target.name)
    part.replace(target)
    return {'url': url, 'path': str(target.relative_to(ROOT)), 'sha256': actual, 'bytes': target.stat().st_size}

def fetch(source=True, weights=True, sample=False):
    assets = []
    if source:
        zip_path = ROOT / 'vendor' / 'codebrain_source.zip'
        url = f'https://codeload.github.com/jingyingma01/CodeBrain/zip/{SOURCE_COMMIT}'
        item = download(url, zip_path)
        destination = (ROOT / 'vendor' / 'CodeBrain').resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for entry in z.infolist():
                parts = Path(entry.filename).parts[1:]
                if not parts: continue
                target = destination.joinpath(*parts).resolve()
                if not target.is_relative_to(destination): raise ValueError('Invalid archive member')
                if entry.is_dir(): target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(entry) as src, target.open('wb') as dst: shutil.copyfileobj(src, dst)
        item['source_commit'] = SOURCE_COMMIT
        assets.append(item)
        (ROOT / 'vendor' / 'codebrain_source.json').write_text(json.dumps(item, indent=2), encoding='utf-8')
    if weights:
        info = json.load(urllib.request.urlopen('https://huggingface.co/api/models/YjMajy/CodeBrain', timeout=30))
        revision = info['sha']
        item = download(f'https://huggingface.co/YjMajy/CodeBrain/resolve/{revision}/CodeBrain_Tokenizer.pth',
                        ROOT / 'weights' / 'CodeBrain_Tokenizer.pth', TOKENIZER_SHA256)
        item['model_revision'] = revision
        assets.append(item)
    if sample:
        base = 'https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'
        for filename in ['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf']:
            assets.append(download(base+filename, ROOT / 'data' / 'sleepedf_sample' / filename))
    manifest = ROOT / 'data' / 'asset_manifest.json'
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps({'assets': assets, 'note': 'Public EEG sample is for interface validation, not FACED/ISRUC benchmark evidence.'}, indent=2), encoding='utf-8')
    print('Verified assets:', len(assets), flush=True)

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    fetch(weights=not args.source_only, sample=args.sample)
