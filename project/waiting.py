import json
import math
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone


def progress_marker(workdir):
    path=workdir/'experiment.sqlite'
    if not path.exists(): return []
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return [db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] if name in tables else 0
                for name in ('translations','backtranslations','validations','translation_errors',
                             'code_generation','evaluations')]


def load_state(workdir):
    path=workdir/'dev-wait.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save_state(workdir,state):
    workdir.mkdir(parents=True,exist_ok=True)
    path=workdir/'dev-wait.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
    temporary.replace(path)


def mark_state(workdir,status):
    state=load_state(workdir)
    if state:
        state.update(status=status,updated_at=time.time())
        if status in ('running','completed'): state['consecutive_failures']=0
        save_state(workdir,state)


def wait_until_retry(workdir,state):
    retry_at=state['retry_at']
    stamp=datetime.fromtimestamp(retry_at,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    message=f"{state['provider']} {state['kind']}: progress saved; retry at {stamp}. Ctrl+C stops safely."
    print(message,flush=True)
    with (workdir/'dev-wait.log').open('a',encoding='utf-8') as log:
        log.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
    while True:
        remaining=retry_at-time.time()
        if remaining<=0: break
        time.sleep(min(remaining,30))
    state.update(status='retrying',updated_at=time.time())
    save_state(workdir,state)


def resume_pending_wait(workdir):
    state=load_state(workdir)
    if state.get('status')=='waiting': wait_until_retry(workdir,state)


def wait_for_api(workdir,stage,retry):
    if retry.get('kind') not in ('daily_quota','rate_limit','service'):
        raise ValueError('Automatic waiting requires a classified temporary API failure')
    previous=load_state(workdir)
    progress=progress_marker(workdir)
    same=(previous.get('provider')==retry['provider'] and previous.get('kind')==retry['kind']
          and previous.get('progress')==progress)
    attempt=previous.get('consecutive_failures',0)+1 if same else 1
    requested=float(retry.get('delay_seconds') or 0)
    if not math.isfinite(requested): requested=0
    if retry['kind']=='daily_quota':


        if attempt==1: fallback=60 if requested>0 else 3600
        elif attempt==2 and previous.get('delay_seconds',0)<3600: fallback=3600
        else: fallback=86400
    else:
        fallback=min(3600,60*2**min(attempt-1,6))
    delay=max(fallback,requested)
    state={'status':'waiting','stage':stage,'provider':retry['provider'],'kind':retry['kind'],
           'consecutive_failures':attempt,'progress':progress,'delay_seconds':delay,
           'retry_at':time.time()+delay,'updated_at':time.time()}
    save_state(workdir,state)
    wait_until_retry(workdir,state)
