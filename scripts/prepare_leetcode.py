import ast, base64, io, json, lzma, pickle, random, zlib
from project.common import ROOT, digest


MIX = {'easy': 10, 'medium': 20, 'hard': 20}

class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Executable object in upstream test data')

def decode_private(value):
    try:return json.loads(value)
    except (ValueError, TypeError):
        data = DataOnlyUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(value)))).load()
        return json.loads(data) if isinstance(data, str) else data

def build_tasks(rows, seed, all_eligible=False, case_prefix='cases'):
    pools = {level: [] for level in MIX}
    excluded = []
    for row in sorted(rows, key=lambda r: str(r['question_id'])):
        if row['platform'].lower() != 'leetcode':continue
        try:
            method = json.loads(row['metadata'])['func_name']
            stub = row['starter_code'].rstrip()+'\n        pass\n'
            tree = ast.parse(stub)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solution')
            func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)

            annotations = [a.annotation for a in func.args.args if a.annotation] + ([func.returns] if func.returns else [])
            names = {n.id for a in annotations for n in ast.walk(a) if isinstance(n, ast.Name)}
            if not names <= {'List', 'list', 'int', 'str', 'bool'}:
                raise ValueError('Requires non-JSON judge types or floating-point judging')
            if 'any order' in row['question_content'].lower():
                raise ValueError('Requires order-independent output judging')
            public = json.loads(row['public_test_cases'])
            cases = public + decode_private(row['private_test_cases'])
            if not cases:raise ValueError('No test cases')
            for case in cases:
                if case['testtype'] != 'functional':raise ValueError('Not a function test')
                args = [json.loads(line) for line in case['input'].splitlines() if line.strip()]
                json.loads(case['output'])
                if len(args) != len(func.args.args)-1:raise ValueError('Argument count mismatch')
            pools[row['difficulty'].lower()].append((row, method, cases, len(public)))
        except (ValueError, KeyError, SyntaxError, StopIteration, TypeError) as error:
            excluded.append({'id': row['question_id'], 'reason': str(error) or 'Unsupported starter interface'})
    rng = random.Random(seed)
    selected = []
    for level, count in MIX.items():
        if not all_eligible and len(pools[level]) < count:raise ValueError(f'Only {len(pools[level])} compatible {level} tasks')
        if all_eligible:
            part = list(pools[level]); rng.shuffle(part); selected.extend(part)
        else:
            selected.extend(rng.sample(pools[level], count))

    manifest_path = ROOT/'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if not all_eligible:
        available = {str(item[0]['question_id']): item for pool in pools.values() for item in pool}
        for change in manifest.get('pre_generation_replacements', []):
            if change['dataset'] != 'livecodebench': continue
            old_id, new_id = (change[key].rsplit('/', 1)[-1] for key in ('old_id', 'new_id'))
            if not any(str(item[0]['question_id']) == old_id for item in selected): continue
            candidate = available[new_id]
            original = next(item for item in selected if str(item[0]['question_id']) == old_id)
            if candidate[0]['difficulty'].lower() != original[0]['difficulty'].lower() or any(
                    str(item[0]['question_id']) == new_id for item in selected):
                raise ValueError('Invalid recorded LiveCodeBench replacement: '+new_id)
            selected = [candidate if str(item[0]['question_id']) == old_id else item for item in selected]
            excluded.append({'id': old_id, 'reason': change['reason'], 'replacement_id': new_id})
    tasks = []
    for row, method, cases, public_count in selected:
        filename = f'{case_prefix}/leetcode-{row["question_id"]}.json.xz'
        target = ROOT/'data'/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        with lzma.open(target, 'wt', encoding='utf-8', preset=1) as out:
            json.dump(cases, out, ensure_ascii=False)
        tasks.append({
            'id': f'LiveCodeBench/leetcode/{row["question_id"]}', 'dataset': 'livecodebench',
            'prompt': row['question_content']+'\n\nRequired Python interface:\n```python\nfrom typing import List\n\n'+row['starter_code'].rstrip()+'\n```\nImplement the method inside class Solution. Do not read stdin or print the answer.',
            'entry_point': method, 'cases_file': filename, 'cases_sha256': digest(target.read_bytes()),
            'case_count': len(cases), 'public_case_count': public_count,
            'private_case_count': len(cases)-public_count, 'difficulty': row['difficulty'],
            'platform': 'leetcode', 'contest_date': row['contest_date'], 'title': row['question_title'],
            'source_hash': digest(row),
        })
    return tasks, excluded
