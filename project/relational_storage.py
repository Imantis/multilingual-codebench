import contextlib
import json
import sqlite3
import time

TABLES = {
    'generation': 'code_generation', 'translation': 'translations',
    'evaluation': 'evaluations', 'backtranslation': 'backtranslations',
    'validation': 'validations', 'translation_error': 'translation_errors',
    'validation_error': 'validation_errors', 'settings': 'run_settings',
}


def remove_language_controls(db, path):

    if db.execute('PRAGMA user_version').fetchone()[0] >= 4:return
    db.commit()
    backup = path / f'experiment.before-schema-v4-{time.time_ns()}.sqlite'
    with contextlib.closing(sqlite3.connect(backup)) as target:db.backup(target)
    print(f'Database backup: {backup}',flush=True)
    obsolete = ('prompt_version','system_prompt_language','reasoning_language_requested',
                'reasoning_language_check','reasoning_control','system_prompts_hash')
    with db:
        db.execute('BEGIN IMMEDIATE')
        for table in TABLES.values():
            columns={row[1] for row in db.execute(f'PRAGMA table_info({table})')}
            for name in obsolete:
                if name in columns:db.execute(f'ALTER TABLE {table} DROP COLUMN {name}')
            for row_id, payload in db.execute(f'SELECT id,payload_json FROM {table}').fetchall():
                record=json.loads(payload)
                if any(name in record for name in obsolete):
                    for name in obsolete:record.pop(name,None)
                    db.execute(f'UPDATE {table} SET payload_json=? WHERE id=?',
                               (json.dumps(record,ensure_ascii=False),row_id))
        db.execute('PRAGMA user_version=4')


COMMON = {
    'task_id': 'TEXT', 'dataset': 'TEXT', 'language_code': 'TEXT',
    'model': 'TEXT', 'provider': 'TEXT', 'created_at': 'REAL',
}
API = {
    'prompt': 'TEXT', 'system_prompt': 'TEXT', 'response_text': 'TEXT',
    'duration_seconds': 'REAL', 'input_tokens': 'INTEGER',
    'output_tokens': 'INTEGER', 'total_tokens': 'INTEGER',
    'raw_response_json': 'TEXT',
}
FIELDS = {
    'generation': COMMON | API | {
        'repeat': 'INTEGER', 'code': 'TEXT', 'status': 'TEXT',
        'finish_reason': 'TEXT',
        'effective_max_tokens': 'INTEGER',
    },
    'translation': COMMON | API | {'translated_prompt': 'TEXT', 'source_hash': 'TEXT', 'accepted': 'INTEGER', 'repair': 'INTEGER'},
    'evaluation': COMMON | {
        'generation_id': 'INTEGER REFERENCES code_generation(id) ON DELETE RESTRICT',
        'status': 'TEXT', 'detail': 'TEXT', 'duration_seconds': 'REAL',
        'syntax_valid': 'INTEGER', 'tests_run': 'INTEGER', 'sloc': 'INTEGER',
        'cc_mean': 'REAL', 'cc_max': 'REAL', 'halstead_volume': 'REAL',
        'halstead_difficulty': 'REAL', 'halstead_effort': 'REAL',
        'nesting_depth': 'INTEGER',
    },
    'backtranslation': COMMON | API | {'translated_prompt': 'TEXT'},
    'validation': COMMON | API | {'score': 'INTEGER', 'reason': 'TEXT'},
    'translation_error': COMMON | API | {'integrity_error': 'TEXT'},
    'validation_error': COMMON | API,
    'settings': {'stage': 'TEXT'},
}


def migrate_approved(db, path):
    if db.execute('PRAGMA user_version').fetchone()[0] >= 5: return
    db.commit()
    if db.execute('SELECT COUNT(*) FROM code_generation').fetchone()[0] or db.execute('SELECT COUNT(*) FROM translations').fetchone()[0]:
        backup = path / f'experiment.before-approved-v5-{time.time_ns()}.sqlite'
        with contextlib.closing(sqlite3.connect(backup)) as target: db.backup(target)
        print(f'Historical run preserved: {backup}', flush=True)
    with db:
        for kind, table in TABLES.items():
            existing = {r[1] for r in db.execute(f'PRAGMA table_info({table})')}
            for name, sql_type in FIELDS[kind].items():
                if name not in existing: db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {sql_type}')
        db.execute('PRAGMA user_version=5')


def write_record(db, kind, key, value):
    table = TABLES[kind]
    record = dict(value) if isinstance(value, dict) else {}
    if kind in ('translation', 'backtranslation', 'validation', 'translation_error', 'validation_error'):
        task, separator, language = key.rpartition(':')
        if separator:
            record.setdefault('task_id', task)
            record.setdefault('language', language)
    task = db.execute('SELECT dataset FROM tasks WHERE task_id=?', (record.get('task_id'),)).fetchone()
    record.setdefault('dataset', task[0] if task else None)
    usage = record.get('usage') or {}
    settings = record.get('settings') or {}
    columns = dict(record)
    if kind == 'settings':columns['stage'] = key
    columns.update(
        language_code=record.get('language'), response_text=record.get('text'),
        translated_prompt=record.get('text'), duration_seconds=record.get('seconds'),
        status=record.get('generation_status', record.get('status')),
        input_tokens=usage.get('prompt_tokens', usage.get('promptTokenCount')),
        output_tokens=usage.get('completion_tokens', usage.get('candidatesTokenCount')),
        total_tokens=usage.get('total_tokens', usage.get('totalTokenCount')),
        effective_max_tokens=settings.get('max_completion_tokens', settings.get('max_tokens', settings.get('maxOutputTokens'))),
        raw_response_json=json.dumps(record['raw'], ensure_ascii=False) if 'raw' in record else None,
    )
    if kind == 'evaluation':
        generation = db.execute('SELECT id,task_id,dataset,language_code,model,provider FROM code_generation WHERE cache_key=?',
                                (record.get('generation_key'),)).fetchone()
        columns['generation_id'] = generation[0] if generation else None
        if generation:
            for name, data in zip(('task_id', 'dataset', 'language_code', 'model', 'provider'), generation[1:]):
                columns[name] = columns.get(name) or data
    names = ['cache_key', *FIELDS[kind], 'payload_json']

    values = [key, *(columns.get(name) for name in FIELDS[kind]), json.dumps(value, ensure_ascii=False)]
    updates = ','.join(f'{name}=excluded.{name}' for name in names if name != 'cache_key')
    db.execute(f'INSERT INTO {table} ({",".join(names)}) VALUES ({",".join("?" for _ in names)}) '
               f'ON CONFLICT(cache_key) DO UPDATE SET {updates}', values)


def migrate(db, path, root, digest):
    version = db.execute('PRAGMA user_version').fetchone()[0]
    if version >= 3:
        return

    from .storage import migrate_v2
    migrate_v2(db, path, root, digest)
    db.commit()
    if db.execute('SELECT COUNT(*) FROM records').fetchone()[0]:
        backup = path / f'experiment.before-schema-v3-{time.time_ns()}.sqlite'
        with contextlib.closing(sqlite3.connect(backup)) as target:
            db.backup(target)
        print(f'Database backup: {backup}', flush=True)
    tasks_file = root / 'data' / 'tasks.jsonl'
    tasks = [json.loads(line) for line in tasks_file.read_text(encoding='utf-8').splitlines()] if tasks_file.exists() else []
    with db:
        db.execute('BEGIN IMMEDIATE')
        for view in ('generations', 'translations', 'evaluations', 'run_settings'):
            db.execute(f'DROP VIEW IF EXISTS {view}')
        db.execute('CREATE TABLE tasks (task_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, original_prompt TEXT NOT NULL)')
        db.executemany('INSERT INTO tasks VALUES (?,?,?)', [(t['id'], t['dataset'], t['prompt']) for t in tasks])
        db.execute('CREATE INDEX tasks_dataset ON tasks(dataset,task_id)')
        for kind, table in TABLES.items():
            fields = ','.join(f'{name} {sql_type}' for name, sql_type in FIELDS[kind].items())
            if fields: fields += ','
            db.execute(f'CREATE TABLE {table} (id INTEGER PRIMARY KEY, {fields}cache_key TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL)')
            if kind != 'settings':
                db.execute(f'CREATE INDEX {table}_selection ON {table}(dataset,language_code,model,task_id)')
        db.execute('CREATE INDEX evaluations_generation ON evaluations(generation_id)')

        db.execute('CREATE TABLE legacy_records (kind TEXT, cache_key TEXT, payload_json TEXT, PRIMARY KEY(kind,cache_key))')
        rows = db.execute('SELECT kind,key,value FROM records ORDER BY CASE kind WHEN \'generation\' THEN 0 WHEN \'evaluation\' THEN 2 ELSE 1 END,key').fetchall()
        for kind, key, value in rows:
            if kind in TABLES:
                write_record(db, kind, key, json.loads(value))
            else:
                db.execute('INSERT INTO legacy_records VALUES (?,?,?)', (kind, key, value))


        config_file = root / 'config.json'
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding='utf-8'))
            by_task = {t['id']:t for t in tasks}
            from .evaluate import VERSION
            for generation_id, cache_key, task_id, code in db.execute('SELECT id,cache_key,task_id,code FROM code_generation').fetchall():
                if task_id not in by_task or code is None:continue
                for evaluation_version in range(1, VERSION+1):
                    evaluation_key = digest([cache_key,digest(code),by_task[task_id],evaluation_version,
                                             config['timeout_seconds'],config['task_timeout_seconds'],config['memory_mb']])
                    db.execute('''UPDATE evaluations SET generation_id=?,
                        task_id=(SELECT task_id FROM code_generation WHERE id=?),
                        dataset=(SELECT dataset FROM code_generation WHERE id=?),
                        language_code=(SELECT language_code FROM code_generation WHERE id=?),
                        model=(SELECT model FROM code_generation WHERE id=?),
                        provider=(SELECT provider FROM code_generation WHERE id=?)
                        WHERE cache_key=? AND generation_id IS NULL''', (generation_id,)*6+(evaluation_key,))

        for kind, key, value in rows:
            if kind in TABLES:
                saved = db.execute(f'SELECT payload_json FROM {TABLES[kind]} WHERE cache_key=?', (key,)).fetchone()
            else:
                saved = db.execute('SELECT payload_json FROM legacy_records WHERE kind=? AND cache_key=?', (kind, key)).fetchone()
            if saved is None or json.loads(saved[0]) != json.loads(value):
                raise ValueError(f'Database migration could not preserve {kind}: {key}')
        if db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Database migration produced invalid generation links')
        db.execute('DROP TABLE records')
        if not db.execute('SELECT COUNT(*) FROM legacy_records').fetchone()[0]:
            db.execute('DROP TABLE legacy_records')
        db.execute('PRAGMA user_version=3')
