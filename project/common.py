import argparse, contextlib, errno, hashlib, json, os, random, re, sqlite3, sys, time

if os.name == "nt":
    import msvcrt

    class fcntl:
        LOCK_EX = 1
        LOCK_SH = 2
        LOCK_NB = 4
        LOCK_UN = 8

        @staticmethod
        def flock(file_obj, operation):
            file_obj.seek(0)

            if operation & fcntl.LOCK_UN:
                try:
                    msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                mode = (
                    msvcrt.LK_NBLCK
                    if operation & fcntl.LOCK_NB
                    else msvcrt.LK_LOCK
                )
                try:
                    msvcrt.locking(file_obj.fileno(), mode, 1)
                except OSError as error:


                    conflicts = {
                        errno.EACCES,
                        errno.EAGAIN,
                        getattr(errno, "EDEADLK", -1),
                        getattr(errno, "EDEADLOCK", -1),
                    }
                    if operation & fcntl.LOCK_NB and error.errno in conflicts:
                        raise BlockingIOError(errno.EAGAIN, "Run lock is already held") from error
                    raise

else:
    import fcntl
from pathlib import Path
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try: stream.reconfigure(encoding='utf-8', errors='backslashreplace')
        except (ValueError,OSError): pass
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'approved-v6'
DATASETS = ('humaneval', 'livecodebench', 'bigcodebench')
LANGUAGES = ('en','de','es','sv','tr','fi','ru','pl','lv')

def digest(value):
    if not isinstance(value, bytes): value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()

def analysis_environment():
    from importlib.metadata import version
    return {'host_python':sys.version.split()[0],
            'packages':{name:version(name) for name in ('numpy','scipy','radon')}}

def config(mode=None):
    for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines() if (ROOT / '.env').exists() else []:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))
    c = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
    mode = mode or os.getenv('EXPERIMENT_MODE', c.get('default_mode','dev'))
    if mode not in ('dev','prod'): raise ValueError('EXPERIMENT_MODE must be dev or prod')
    profile = c.pop('modes')[mode]
    for key,value in profile.items():
        if isinstance(value,dict): c[key] = c.get(key,{}) | value
        else: c[key] = value
    c['mode'] = mode

    prefix = 'PROD_' if mode=='prod' else ''
    provider = os.getenv(prefix+'API_PROVIDER')
    if provider:
        if provider not in ('gemini','openrouter'): raise ValueError('API_PROVIDER must be gemini or openrouter')
        c['translation_provider'] = c['validator_provider'] = provider
        c['model_providers']['gemma-4-26b'] = provider
        c['model_providers']['gpt-oss-20b'] = 'groq' if mode=='dev' else 'openrouter'
    for key in ('translation_model', 'validator_model', 'translation_provider', 'validator_provider'):
        c[key] = os.getenv(prefix+key.upper(), c[key])
    for role in ('translation','validator'):
        model = c[role+'_model']
        if c[role+'_provider']=='openrouter' and '/' not in model:
            if model.startswith(('gemini-','gemma-')): model='google/'+model
            else: raise ValueError('Use the exact OpenRouter model ID (including its vendor) for '+role)
        elif c[role+'_provider']=='gemini': model=model.removeprefix('google/').removesuffix(':free')
        c[role+'_model'] = model
    c['model_providers']['gemma-4-26b'] = os.getenv(prefix+'GEMMA_PROVIDER', c['model_providers']['gemma-4-26b'])
    c['model_providers']['gpt-oss-20b'] = os.getenv(prefix+'GPT_OSS_PROVIDER', c['model_providers']['gpt-oss-20b'])
    for alias, var in [('gpt-oss-20b', 'GPT_OSS_MODEL'), ('gemma-4-26b', 'GEMMA_MODEL')]:
        model = os.getenv(prefix+var, c['models'][alias])
        provider = c.get('model_providers', {}).get(alias, 'openrouter')
        if alias == 'gemma-4-26b' and provider == 'gemini' and model.startswith('google/'):
            model = model.removeprefix('google/').removesuffix(':free')
        if alias == 'gemma-4-26b' and provider == 'openrouter' and '/' not in model:
            model = 'google/' + model
        c['models'][alias] = model
    for role in ('translation','validator'):
        if os.getenv(prefix+'OPENROUTER_'+role.upper()+'_BACKEND'):
            c['openrouter_backends'][c[role+'_model']] = os.environ[prefix+'OPENROUTER_'+role.upper()+'_BACKEND']
    allowed = ('gemini','groq') if mode=='dev' else ('openrouter',)
    if any(c[role+'_provider'] not in allowed for role in ('translation','validator')) or any(
            provider not in allowed for provider in c['model_providers'].values()):
        raise ValueError('dev uses Gemini and direct Groq only; prod uses paid OpenRouter only.')
    c['request_workers']=int(os.getenv('REQUEST_WORKERS', c.get('request_workers',4)))
    if not 1<=c['request_workers']<=32: raise ValueError('request_workers must be between 1 and 32')
    for key in ('gemini_rpm', 'groq_rpm', 'openrouter_rpm'):
        c[key]=int(os.getenv(key.upper(), c[key]))
        if c[key]<0: raise ValueError(key+' cannot be negative (0 means no local pacing)')
    if c['repetitions'] != 3 or c['translation_repairs'] != 2:
        raise ValueError('The approved design uses three generations and at most two translation repairs')
    if len(c['languages'])!=9 or set(c['languages'])!=set(LANGUAGES):
        raise ValueError('Keep the approved nine-language family in config; select testing subsets with --languages')
    if len(c['models'])!=2 or len(set(c['models'].values()))!=2:
        raise ValueError('The approved design needs two distinct model IDs; duplicate aliases would repeat paid calls for the same slot')
    return c

def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--mode',choices=['dev','prod'],help='dev: Gemini and direct Groq; prod: paid OpenRouter (or set EXPERIMENT_MODE)')
    p.add_argument('--languages', default='all', help='Comma-separated language codes')
    p.add_argument('--model', default='all', help='Configured alias or exact model ID')
    p.add_argument('--dataset', choices=['all','humaneval','livecodebench','bigcodebench'], default='all')
    p.add_argument('--limit', type=int, help='Tasks per dataset; use 1 for a small run')
    p.add_argument('--workdir', type=Path, help='Saved run directory; defaults to runs/dev or runs/prod')
    p.add_argument('--dry-run', action='store_true', help='Print plan without API calls')
    p.add_argument('--max-paid-requests', type=int, help='Persistent OpenRouter POST ceiling for this workdir, including retries')
    return p

def configure_run(args):
    c=config(args.mode)
    if args.workdir is None: args.workdir=ROOT/'runs'/c['mode']
    if args.max_paid_requests is not None and args.max_paid_requests < 1:
        raise ValueError('--max-paid-requests must be positive')
    c['_paid_request_ledger']=str(args.workdir/'paid-requests.sqlite')
    c['_max_paid_requests']=args.max_paid_requests
    return c

def freeze_profile(store,c):
    value={'mode':c['mode'],'development_only':c['mode']=='dev',
           'translation':{'model':c['translation_model'],'provider':c['translation_provider']},
           'audit':{'model':c['validator_model'],'provider':c['validator_provider']},
           'models':{alias:{'model':model,'provider':c['model_providers'][alias]} for alias,model in c['models'].items()}}

    for role,configured in [('forward',value['translation']),('audit',value['audit'])]:
        previous=store.get('settings',PROTOCOL+':translation:'+role)
        if previous and any(previous.get(k)!=v for k,v in configured.items()):
            raise ValueError('Translation/audit deployment changed: use a new --workdir')
    for alias,configured in value['models'].items():
        previous=store.get('settings',PROTOCOL+':model-alias:'+alias)
        if previous and previous!=configured:
            raise ValueError('Generation deployment changed: use a new --workdir')
    store.freeze(PROTOCOL+':profile',value)

def selection(args, c, store=None):
    langs = list(c['languages']) if args.languages == 'all' else list(dict.fromkeys(args.languages.split(',')))
    if not langs or any(x not in c['languages'] for x in langs): raise ValueError('Unknown language code')
    models = [c['models'][alias] for alias in c['default_models']] if args.model == 'all' else [c['models'].get(args.model, args.model)]
    models = list(dict.fromkeys(models))
    if c['mode']=='dev' and any(provider_for_model(model,c) not in ('gemini','groq') for model in models):
        raise ValueError('Development permits Gemini and direct Groq only; paid OpenRouter requires --mode prod.')
    if args.limit is not None and args.limit < 1: raise ValueError('--limit must be positive')
    rows = [json.loads(x) for x in (ROOT / 'data' / 'tasks.jsonl').read_text(encoding='utf-8').splitlines()]

    manifest = json.loads((ROOT / 'data' / 'manifest.json').read_text())
    if digest((ROOT / 'data' / 'tasks.jsonl').read_bytes()) != manifest['tasks_sha256']:
        raise ValueError('Bundled task checksum mismatch')
    if store:
        rows = store.get('settings', PROTOCOL+':selection') or rows
    chosen = []
    for ds in ('humaneval','livecodebench','bigcodebench'):
        if args.dataset not in ('all',ds): continue
        part = [r for r in rows if r['dataset'] == ds]
        chosen.extend(part[:args.limit])
    if len({t['id'] for t in chosen})!=len(chosen):
        raise ValueError('Duplicate selected task IDs; no API requests should be made for repeated slots')
    for task in chosen:
        blocks=re.findall(r'\x60{3}python\s*\n([\s\S]*?)\x60{3}',task['prompt'],re.I)
        if not any(re.search(r'\bdef\s+'+re.escape(task['entry_point'])+r'\s*\(',b) for b in blocks):
            raise ValueError('Prepared prompt omits its required function: '+task['id']+'. Rebuild the task source before spending API requests.')
    return chosen, langs, models

def provider_for_model(model, c):
    for alias, provider in c.get('model_providers', {}).items():
        if c['models'].get(alias) == model:
            return provider
    return 'openrouter'

def models_by_provider(models, c):
    grouped = {}
    for model in dict.fromkeys(models):
        grouped.setdefault(provider_for_model(model, c), []).append(model)
    return grouped

@contextlib.contextmanager
def run_lock(path, name='.run.lock'):
    path.mkdir(parents=True, exist_ok=True)
    with (path / name).open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another command is using this workdir')
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

class Store:
    def __init__(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path / 'experiment.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA foreign_keys=ON')
        if self.db.execute('PRAGMA user_version').fetchone()[0]<3:
            self.db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, key TEXT, value TEXT, PRIMARY KEY(kind,key))')
        from .storage import migrate
        migrate(self.db,path,ROOT,digest)
        from .relational_storage import migrate_approved
        migrate_approved(self.db,path)
    def close(self):
        self.db.close()
    def backup(self, label):
        target = self.path / f'experiment.before-{label}-{time.time_ns()}.sqlite'
        with contextlib.closing(sqlite3.connect(target)) as db: self.db.backup(db)
        return target
    def get(self, kind, key):
        from .relational_storage import TABLES
        row = self.db.execute(f'SELECT payload_json FROM {TABLES[kind]} WHERE cache_key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def put(self, kind, key, value):
        from .relational_storage import write_record
        with self.db:write_record(self.db,kind,key,value)
    def all(self, kind):
        from .relational_storage import TABLES
        return [json.loads(r[0]) for r in self.db.execute(f'SELECT payload_json FROM {TABLES[kind]} ORDER BY cache_key')]
    def freeze(self, kind, value):
        previous = self.get('settings',kind)
        if previous is not None and previous != value:
            raise ValueError(f'{kind} settings changed: use a new --workdir to avoid mixing experiments')
        self.put('settings',kind,value)

    def clear_generations(self, task_ids, languages, models=None):

        if not task_ids or not languages or models == []:return 0
        where='task_id IN ('+','.join('?' for _ in task_ids)+') AND language_code IN ('+','.join('?' for _ in languages)+')'
        values=[*task_ids,*languages]
        if models is not None:
            where+=' AND model IN ('+','.join('?' for _ in models)+')'
            values.extend(models)
        self.db.execute(f'''DELETE FROM evaluations WHERE ({where}) OR generation_id IN
            (SELECT id FROM code_generation WHERE {where}) OR json_extract(payload_json,'$.generation_key') IN
            (SELECT cache_key FROM code_generation WHERE {where})''',values*3)
        return self.db.execute(f'DELETE FROM code_generation WHERE {where}',values).rowcount

    def clear_translations(self, jobs):
        from .relational_storage import TABLES
        keys=[(translation_key(t,l),) for t,l in jobs]
        with self.db:
            for kind in ('translation','translation_error','backtranslation','validation','validation_error'):
                self.db.executemany(f'DELETE FROM {TABLES[kind]} WHERE cache_key=?',keys)

            for task,language in jobs:
                self.clear_generations([task['id']],[language])

def translation_key(task, language): return PROTOCOL+':'+task['id'] + ':' + language

def translation_is_current(task, record):
    return bool(record and record.get('protocol') == PROTOCOL and record.get('accepted') is True
                and record.get('source_hash') == digest(task['prompt'])
                and isinstance(record.get('text'),str) and record['text'].strip()
                and ('ZXQKEEP' not in record['text'] or 'ZXQKEEP' in task['prompt'])
                and not record.get('integrity_error'))

def prompt(task, language, store):
    if language == 'en': return task['prompt']
    rec = store.get('translation', translation_key(task,language))
    if not translation_is_current(task,rec):
        raise ValueError(f"Missing, outdated, or invalid translation: {task['id']} {language}; rerun experiment_translations for this language and dataset")
    return rec['text']

def system_prompt(language, store):
    from .prompts import CODE_SYSTEM_PROMPT
    if language == 'en': return CODE_SYSTEM_PROMPT
    return prompt({'id':'__instructions__', 'prompt':CODE_SYSTEM_PROMPT}, language, store)

def active_generations(store):
    return [r for r in store.all('generation') if r.get('protocol') == PROTOCOL and not r.get('excluded')]

def entry(main):
    try: main()
    except KeyboardInterrupt: print('\nStopped; completed requests are saved.', file=sys.stderr); sys.exit(130)
    except (ValueError, RuntimeError) as e:
        retry=getattr(e,'retry',None)
        notice=os.getenv('CODEBENCH_RETRY_NOTICE')
        if retry and notice:
            Path(notice).write_text(json.dumps(retry|{'message':str(e)},indent=2),encoding='utf-8')
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(2)
