import ast
import gzip
import json
import random
import urllib.request
from .common import ROOT, PROTOCOL, digest


LCB_ADDITIONAL_FILES = {
    'test2.jsonl': '095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3',
    'test3.jsonl': '28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79',
    'test4.jsonl': 'd711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da',
    'test5.jsonl': '7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214',
    'test6.jsonl': 'bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5',
}


def extend_livecodebench_reserves(c, store):

    import hashlib
    import time
    from .common import active_generations
    from scripts.prepare_leetcode import build_tasks
    key = PROTOCOL+':livecodebench-reserve-extension'
    if store.get('settings', key): return
    if active_generations(store):
        raise RuntimeError('LiveCodeBench reserves can only be extended before code generation')
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    revision = manifest['sources']['lcb_revision']
    cache = ROOT/'data/source-cache'
    rows = []; sources = []
    for filename, expected in LCB_ADDITIONAL_FILES.items():
        path = cache/('leetcode-'+filename+'.gz')
        receipt = path.with_suffix('.source.json')
        url = f'https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{revision}/{filename}'
        recorded = json.loads(receipt.read_text()) if receipt.exists() else {}
        if not path.exists() or recorded.get('source_sha256') != expected or recorded.get('cache_sha256') != digest(path.read_bytes()):
            temporary = path.with_suffix('.partial')
            sha = hashlib.sha256(); count = 0
            print('Downloading pinned LiveCodeBench reserve file:', filename, flush=True)
            with urllib.request.urlopen(url, timeout=180) as response, gzip.open(temporary,'wb') as output:
                for line in response:
                    sha.update(line)
                    if json.loads(line)['platform'].lower() == 'leetcode':
                        output.write(line); count += 1
            if sha.hexdigest() != expected:
                temporary.unlink()
                raise ValueError('LiveCodeBench reserve source checksum mismatch: '+filename)
            temporary.replace(path)
            recorded = {'url':url,'revision':revision,'source_sha256':expected,
                        'cache_sha256':digest(path.read_bytes()),'leetcode_rows':count}
            receipt.write_text(json.dumps(recorded,indent=2),encoding='utf-8')
        sources.append(recorded)
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            rows.extend(json.loads(line) for line in stream)
    tasks, exclusions = build_tasks(rows,c['seed'],all_eligible=True,
                                    case_prefix='source-cache/reserve-v6-cases')
    previous = reserve_tasks('livecodebench',c,store)
    excluded = {t['id'] for t in previous+store.get('settings',PROTOCOL+':selection')}
    excluded.update(json.loads(line)['id'] for line in (ROOT/'data/tasks.jsonl').read_text(encoding='utf-8').splitlines())
    excluded.update(r['old_id'] for r in manifest.get('pre_generation_replacements',[]))
    excluded.update(r[field] for r in (store.get('settings',PROTOCOL+':replacements') or []) for field in ('old_id','new_id'))
    additional = [t for t in tasks if t['id'] not in excluded]
    if len({t['id'] for t in additional}) != len(additional):
        raise ValueError('Repeated task IDs in additional LiveCodeBench files')
    additional.sort(key=lambda t:t['id']); random.Random(c['seed']).shuffle(additional)
    print('Backing up run before reserve extension:',store.backup('reserve-extension'),flush=True)
    from .relational_storage import write_record
    with store.db:
        write_record(store.db,'settings',PROTOCOL+':reserves:livecodebench-before-extension',previous)
        write_record(store.db,'settings',PROTOCOL+':reserves:livecodebench',previous+additional)
        write_record(store.db,'settings',key,{'frozen_at':time.time(),'sources':sources,
            'seed':c['seed'],'original_reserves':len(previous),'additional_reserves':len(additional),
            'order':'Original reserves first, then sorted IDs shuffled with the approved seed.',
            'eligibility':'Same JSON-compatible LeetCode function filters and all supplied Lite tests.',
            'excluded':exclusions,'task_ids':[t['id'] for t in additional]})
    print('Frozen additional LiveCodeBench reserves:',len(additional),flush=True)

def reserve_tasks(dataset, c, store):
    key = PROTOCOL+':reserves:'+dataset
    previous = store.get('settings', key)
    if previous is not None: return previous
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    sources = manifest['sources']
    cache = ROOT/'data/source-cache'; cache.mkdir(exist_ok=True)
    def download(filename, url, sha):
        path = cache/filename
        if not path.exists():
            print('Downloading pinned replacement source:', filename, flush=True)
            with urllib.request.urlopen(url, timeout=180) as response: content = response.read()
            if digest(content) != sha: raise ValueError('Replacement source checksum mismatch: '+filename)
            path.write_bytes(content)
        if digest(path.read_bytes()) != sha: raise ValueError('Replacement cache checksum mismatch: '+filename)
        return path
    tasks = []
    if dataset == 'humaneval':
        from scripts.prepare_data import humaneval_task
        path = download('humaneval.jsonl.gz', sources['urls']['humaneval.jsonl.gz'], sources['humaneval_sha256'])
        for line in gzip.open(path, 'rt', encoding='utf-8'):
            r = json.loads(line)
            tasks.append(humaneval_task(r)|{'reference':r['prompt']+r['canonical_solution']})
    elif dataset == 'bigcodebench':
        path = download('bcb.parquet', sources['urls']['bcb.parquet'], sources['bcb_sha256'])
        try: import pyarrow.parquet as pq
        except ImportError: raise ValueError('Replacement preparation needs pyarrow: install requirements-preparation.txt') from None
        from scripts.prepare_data import imports
        import sys
        allowed = set(sys.stdlib_module_names)|{'numpy','pandas','scipy','matplotlib'}
        forbidden = {'socket','urllib','http','ftplib','subprocess','multiprocessing','tkinter','turtle','venv','webbrowser'}
        for r in pq.read_table(path).to_pylist():
            ref = r['complete_prompt']+r['canonical_solution']
            try: modules = imports(ref+'\n'+r['test'])
            except SyntaxError: continue
            if not modules <= allowed or modules & forbidden: continue
            tasks.append({'id':r['task_id'], 'dataset':dataset,
                          'prompt':r['instruct_prompt']+'\n\nRequired Python interface:\n```python\n'+r['code_prompt'].rstrip()+'\n```',
                          'entry_point':r['entry_point'], 'tests':r['test'], 'source_hash':digest(r),
                          'imports':sorted(modules), 'reference':ref})
    else:
        from scripts.prepare_leetcode import build_tasks
        path = cache/'leetcode-release-v1.jsonl.gz'
        if not path.exists():
            import hashlib
            temporary = path.with_suffix('.partial')
            url=f'https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{sources["lcb_revision"]}/{sources["lcb_file"]}'
            sha=hashlib.sha256()
            print('Downloading pinned LiveCodeBench replacement source...',flush=True)
            with urllib.request.urlopen(url,timeout=180) as response, gzip.open(temporary,'wb') as out:
                for line in response:
                    sha.update(line)
                    if json.loads(line)['platform'].lower()=='leetcode': out.write(line)
            if sha.hexdigest()!=sources['lcb_sha256']:
                temporary.unlink(); raise ValueError('LiveCodeBench replacement source checksum mismatch')
            temporary.replace(path)
        with gzip.open(path, 'rt', encoding='utf-8') as stream: rows = [json.loads(x) for x in stream]
        tasks, _ = build_tasks(rows, c['seed'], all_eligible=True,case_prefix='source-cache/reserve-cases')
    tasks.sort(key=lambda t:t['id'])
    random.Random(c['seed']).shuffle(tasks)

    original = {json.loads(x)['id'] for x in (ROOT/'data/tasks.jsonl').read_text(encoding='utf-8').splitlines()}
    source_exclusions = {r['old_id'] for r in manifest.get('pre_generation_replacements', [])}
    tasks = [t for t in tasks if t['id'] not in original | source_exclusions]
    store.freeze(key, tasks)
    return tasks

def replacement(task, c, store):
    from .evaluate import evaluate
    rows = store.get('settings', PROTOCOL+':selection')
    excluded = store.get('settings', PROTOCOL+':replacements') or []
    used = {t['id'] for t in rows} | {x['old_id'] for x in excluded} | {x['new_id'] for x in excluded}
    for candidate in reserve_tasks(task['dataset'], c, store):
        if candidate['id'] in used: continue
        if store.get('settings',PROTOCOL+':reserve-exclusion:'+candidate['id']): continue
        if task['dataset'] == 'livecodebench' and candidate['difficulty'].lower() != task['difficulty'].lower(): continue
        ref = candidate.get('reference')
        checks = []
        if ref:
            checks = [evaluate(candidate, ref, c)]
            if checks[0]['status'] == 'infrastructure_error': raise RuntimeError('Reserve reference infrastructure fault: '+checks[0]['detail'])
            if checks[0]['status'] == 'pass' and task['dataset']=='bigcodebench': checks.append(evaluate(candidate, ref, c))
            if any(x['status'] != 'pass' for x in checks):
                store.put('settings', PROTOCOL+':reserve-exclusion:'+candidate['id'], {'checks':checks})
                continue
        candidate = dict(candidate); candidate.pop('reference', None)
        excluded.append({'old_id':task['id'], 'new_id':candidate['id'], 'dataset':task['dataset'],
                         'difficulty':task.get('difficulty'), 'reason':'Translation checks unresolved after two repairs', 'reference_checks':checks})
        rows = [candidate if t['id'] == task['id'] else t for t in rows]
        with store.db:
            from .relational_storage import write_record
            write_record(store.db, 'settings', PROTOCOL+':selection', rows)
            write_record(store.db, 'settings', PROTOCOL+':replacements', excluded)
            store.db.execute('INSERT OR REPLACE INTO tasks VALUES (?,?,?)', (candidate['id'],candidate['dataset'],candidate['prompt']))
        print(f'Replaced {task["id"]} with {candidate["id"]} before generation', flush=True)
        return candidate
    if task['dataset']=='livecodebench' and not store.get('settings',PROTOCOL+':livecodebench-reserve-extension'):
        extend_livecodebench_reserves(c,store)
        return replacement(task,c,store)
    raise RuntimeError('No eligible reserve remains for '+task['dataset']+' '+str(task.get('difficulty','')))
