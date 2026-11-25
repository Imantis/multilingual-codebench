import contextlib
import json
import sqlite3
import time


LEGACY_V1_SYSTEM_PROMPT = (
    'Write a complete Python 3 solution to the supplied problem. Return only Python code in one code block. '
    'Preserve the required interface. Include necessary imports. For standard-input tasks read stdin and write stdout. '
    'For function tasks define the full function. Do not include tests or explanations.'
)


def migrate_v2(db, path, root, digest):
    if db.execute('PRAGMA user_version').fetchone()[0] >= 2:
        return
    db.commit()
    if db.execute('SELECT COUNT(*) FROM records').fetchone()[0]:
        backup = path / f'experiment.before-schema-v2-{time.time_ns()}.sqlite'
        with contextlib.closing(sqlite3.connect(backup)) as target:
            db.backup(target)
        print(f'Database backup: {backup}', flush=True)
    with db:
        db.execute('BEGIN IMMEDIATE')
        columns = {row[1] for row in db.execute('PRAGMA table_xinfo(records)')}
        expressions = {
            'task_id': "json_extract(value,'$.task_id')",
            'dataset': "json_extract(value,'$.dataset')",
            'language_code': "json_extract(value,'$.language')",
            'model': "json_extract(value,'$.model')",
            'provider': "json_extract(value,'$.provider')",
            'prompt': "json_extract(value,'$.prompt')",
            'system_prompt': "json_extract(value,'$.system_prompt')",
            'status': "COALESCE(json_extract(value,'$.generation_status'),json_extract(value,'$.status'))",
            'created_at': "json_extract(value,'$.created_at')",
        }
        for name, expression in expressions.items():
            if name not in columns:
                column_type = 'REAL' if name == 'created_at' else 'TEXT'
                db.execute(f'ALTER TABLE records ADD COLUMN {name} {column_type} GENERATED ALWAYS AS ({expression}) VIRTUAL')
        tasks_file = root / 'data' / 'tasks.jsonl'
        tasks = {t['id']: t for t in (json.loads(line) for line in tasks_file.read_text(encoding='utf-8').splitlines())} if tasks_file.exists() else {}
        translations = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM records WHERE kind='translation'")}
        settings = db.execute("SELECT value FROM records WHERE kind='settings' AND key='generation'").fetchone()
        prompt_version = json.loads(settings[0]).get('prompt_version') if settings else None
        for key, value in db.execute("SELECT key,value FROM records WHERE kind='generation'").fetchall():
            record = json.loads(value)
            task = tasks.get(record.get('task_id'), {})
            if not record.get('dataset') and task:
                record['dataset'] = task['dataset']
            if 'prompt' not in record:
                if record.get('language') == 'en':
                    candidate = task.get('prompt')
                else:
                    candidate = translations.get(f"{record.get('task_id')}:{record.get('language')}", {}).get('text')
                if isinstance(candidate, str) and record.get('prompt_hash') == digest(candidate):
                    record['prompt'] = candidate
                    record['prompt_source'] = 'recovered_hash_verified'
                else:
                    record['prompt_source'] = 'unavailable'
            if 'system_prompt' not in record and prompt_version == 1:
                record['system_prompt'] = LEGACY_V1_SYSTEM_PROMPT
                record['system_prompt_source'] = 'reconstructed_prompt_version_1'
            record.setdefault('generation_status', 'generated' if str(record.get('text') or '').strip() else 'no_text')
            db.execute("UPDATE records SET value=? WHERE kind='generation' AND key=?", (json.dumps(record, ensure_ascii=False), key))
        db.execute('CREATE INDEX IF NOT EXISTS records_selection ON records(kind,model,dataset,language_code,task_id)')
        db.execute('''CREATE VIEW IF NOT EXISTS generations AS
            SELECT key AS generation_key, task_id, dataset, language_code, model, provider,
                   json_extract(value,'$.repeat') AS repeat, status, prompt, system_prompt,
                   json_extract(value,'$.prompt_source') AS prompt_source,
                   json_extract(value,'$.system_prompt_source') AS system_prompt_source,
                   json_extract(value,'$.prompt_hash') AS prompt_hash,
                   json_extract(value,'$.text') AS response_text,
                   json_extract(value,'$.code') AS code,
                   json_extract(value,'$.finish_reason') AS finish_reason,
                   json_extract(value,'$.no_text_reason') AS no_text_reason,
                   json_extract(value,'$.seconds') AS duration_seconds, created_at,
                   datetime(created_at,'unixepoch') AS created_at_utc,
                   COALESCE(json_extract(value,'$.usage.prompt_tokens'),json_extract(value,'$.usage.promptTokenCount')) AS input_tokens,
                   COALESCE(json_extract(value,'$.usage.completion_tokens'),json_extract(value,'$.usage.candidatesTokenCount')) AS output_tokens,
                   COALESCE(json_extract(value,'$.usage.total_tokens'),json_extract(value,'$.usage.totalTokenCount')) AS total_tokens,
                   json_extract(value,'$.requested_max_tokens') AS requested_max_tokens,
                   COALESCE(json_extract(value,'$.settings.max_completion_tokens'),json_extract(value,'$.settings.max_tokens')) AS effective_max_tokens,
                   json_extract(value,'$.settings') AS settings_json,
                   json_extract(value,'$.usage') AS usage_json,
                   json_extract(value,'$.raw') AS raw_response_json
            FROM records WHERE kind='generation' ''')
        db.execute('''CREATE VIEW IF NOT EXISTS translations AS
            SELECT key AS translation_key, task_id, language_code, model, provider, prompt AS request_prompt,
                   json_extract(value,'$.text') AS translated_prompt, system_prompt,
                   json_extract(value,'$.source_hash') AS source_hash, created_at,
                   json_extract(value,'$.raw') AS raw_response_json
            FROM records WHERE kind='translation' ''')
        db.execute('''CREATE VIEW IF NOT EXISTS evaluations AS
            SELECT key AS evaluation_key, json_extract(value,'$.generation_key') AS generation_key,
                   task_id, dataset, language_code, model, status,
                   json_extract(value,'$.detail') AS detail,
                   json_extract(value,'$.seconds') AS duration_seconds, value AS metrics_json
            FROM records WHERE kind='evaluation' ''')
        db.execute("CREATE VIEW IF NOT EXISTS run_settings AS SELECT key AS stage,value AS settings_json FROM records WHERE kind='settings'")
        db.execute('PRAGMA user_version=2')


def migrate(db, path, root, digest):
    from .relational_storage import migrate as migrate_relational, remove_language_controls
    migrate_relational(db, path, root, digest)
    remove_language_controls(db,path)
