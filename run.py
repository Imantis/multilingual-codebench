import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def prepare_inputs(cache=None):
    with gzip.open(ROOT/'inputs.json.gz', 'rt', encoding='utf-8') as stream:
        bundle = json.load(stream)
    manifest, tasks = bundle['manifest'], bundle['tasks']
    data = ROOT/'data'
    data.mkdir(exist_ok=True)
    task_file = data/'tasks.jsonl'
    if task_file.exists():
        if digest(task_file.read_bytes()) != manifest['tasks_sha256']:
            raise ValueError('Existing input selection differs; use a separate folder.')
        return
    (data/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    needed = {t['id'].rsplit('/', 1)[-1]:t for t in tasks if t['dataset']=='livecodebench'}
    cached = Path(cache)/'leetcode-release-v1.jsonl.gz' if cache else data/'source-cache/leetcode-release-v1.jsonl.gz'
    source = manifest['sources']
    rows = []
    if cached.exists():
        with gzip.open(cached, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['platform'].lower()=='leetcode' and str(row['question_id']) in needed:
                    rows.append(row)
    else:
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix('.partial')
        url = f'https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{source["lcb_revision"]}/{source["lcb_file"]}'
        checksum = hashlib.sha256()
        print('Downloading pinned benchmark inputs...', flush=True)
        with urllib.request.urlopen(url, timeout=300) as response, gzip.open(temporary, 'wb') as output:
            for line in response:
                checksum.update(line)
                row = json.loads(line)
                if row['platform'].lower()=='leetcode':
                    output.write(line)
                    if str(row['question_id']) in needed:
                        rows.append(row)
        if checksum.hexdigest()!=source['lcb_sha256']:
            temporary.unlink()
            raise ValueError('Upstream input checksum mismatch.')
        temporary.replace(cached)
    if len(rows)!=len(needed) or len({str(r['question_id']) for r in rows})!=len(needed):
        raise ValueError('Pinned source does not contain the required input selection.')
    for row in rows:
        if digest(row)!=needed[str(row['question_id'])]['source_hash']:
            raise ValueError('Task source checksum mismatch: '+str(row['question_id']))
    from scripts.prepare_leetcode import build_tasks
    restored, _ = build_tasks(rows, manifest['seed'], all_eligible=True)
    by_id = {t['id']:t for t in restored}
    for task in tasks:
        if task['dataset']=='livecodebench':
            actual = by_id[task['id']]
            if digest(actual['prompt'])!=task['prompt']['sha256']:
                raise ValueError('Reconstructed statement differs: '+task['id'])
            for key,value in task.items():
                if key!='prompt' and actual.get(key)!=value:
                    raise ValueError('Reconstructed task differs: '+task['id']+' '+key)
            task['prompt'] = actual['prompt']
    content = ''.join(json.dumps(t, ensure_ascii=False)+bundle['task_line_ending'] for t in tasks).encode('utf-8')
    if digest(content)!=manifest['tasks_sha256']:
        raise ValueError('Reconstructed selection checksum mismatch.')
    references = base64.b64decode(bundle['references_gzip_base64'])
    if digest(references)!=manifest['references_sha256']:
        raise ValueError('Reference checksum mismatch.')
    (data/'references.json.gz').write_bytes(references)
    (data/'leetcode-exclusions.json').write_text(json.dumps(bundle['leetcode_exclusions']), encoding='utf-8')
    task_file.write_bytes(content)
    print('Prepared 150 checksum-verified tasks.', flush=True)


def csv_records(path):
    floating = {'seconds','cc_max','halstead_volume','sloc','nesting_depth','generation_seconds','sampling_temperature'}
    integers = {'repeat','tests_run','request_seed','input_utf8_bytes','combined_prompt_utf8_bytes','input_tokens','output_tokens','total_tokens','transport_attempts','case_index'}
    rows = list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    for row in rows:
        for key,value in list(row.items()):
            if key in floating:
                row[key] = float(value) if value else None
            elif key in integers:
                row[key] = int(float(value)) if value else None
            elif key=='syntax_valid':
                row[key] = value=='True'
    return rows


def same_values(actual, expected):
    if isinstance(actual,dict) and isinstance(expected,dict):
        return actual.keys()==expected.keys() and all(same_values(actual[k], expected[k]) for k in actual)
    if isinstance(actual,list) and isinstance(expected,list):
        return len(actual)==len(expected) and all(same_values(a,b) for a,b in zip(actual,expected))
    if isinstance(actual,(int,float)) and isinstance(expected,(int,float)):
        return math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10)
    return actual==expected


def replay(output):
    output = output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(ROOT/'records.tar.gz','r:gz') as archive:
        for member in archive:
            target = output/member.name
            if not member.isfile() or not target.resolve().is_relative_to(output):
                raise ValueError('Invalid archive member.')
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as destination:
                while block:=source.read(1024*1024):destination.write(block)
    records = csv_records(output/'results.csv')
    tasks = [{**r,'id':r['id']} for r in csv.DictReader((output/'selected-tasks.csv').open(encoding='utf-8',newline=''))]
    expected = json.loads((output/'report.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    config['models'] |= config['modes']['prod']['models']
    from project.statistics import analyses, summarize, stability
    print('Recomputing saved comparisons...',flush=True)
    summary, difficulty = summarize(records,tasks,expected['metadata']['languages'],expected['metadata']['models'])
    statistics = analyses(records,tasks,expected['metadata']['languages'],expected['metadata']['models'],config)
    repeats = stability(records)
    for key,value in [('summary',summary),('difficulty',difficulty),('statistics',statistics),('stability',repeats)]:
        if not same_values(value,expected[key]):
            raise ValueError('Saved and recomputed values differ: '+key)
    from project.thesis_export import export
    export(expected,output/'figures')
    print(f'Verified {len(records)} outputs and all saved comparisons. Files: {output}',flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--saved',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--source-cache',type=Path)
    args,remaining = parser.parse_known_args()
    if args.saved:
        if remaining:parser.error('Unexpected arguments for saved-data replay.')
        replay(args.output)
        return
    if remaining in (['--help'],['-h']):
        print('Saved data: python run.py --saved\nNew collection:')
        raise SystemExit(subprocess.call([sys.executable,'-m','project.run',*remaining],cwd=ROOT))
    prepare_inputs(args.source_cache)
    if not args.prepare_only:
        raise SystemExit(subprocess.call([sys.executable,'-m','project.run',*remaining],cwd=ROOT))


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError) as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(1)
