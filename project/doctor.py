import gzip
import json
import platform
import time
from .common import *
from .api import Client
from .evaluate import evaluate, metrics
from .sandbox import ensure_image

def check_references(tasks,c,store=None):
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    path=ROOT/'data/references.json.gz'
    if digest(path.read_bytes())!=manifest['references_sha256']: raise ValueError('Reference checksum mismatch')
    with gzip.open(path,'rt',encoding='utf-8') as stream: references=json.load(stream)
    if store:
        for ds in dict.fromkeys(t['dataset'] for t in tasks):
            for t in store.get('settings',PROTOCOL+':reserves:'+ds) or []:
                if t.get('reference'): references[t['id']]=t['reference']
    checks=[]
    for task in tasks:
        ref=references.get(task['id'])
        if ref is None:
            checks.append({'task_id':task['id'],'status':'unavailable','reason':'No canonical solution supplied by LiveCodeBench'})
            continue
        count=2 if task['dataset']=='bigcodebench' else 1
        for repeat in range(count):
            result=evaluate(task,ref,c)
            checks.append({'task_id':task['id'],'repeat':repeat,**result})
            print('Reference',task['id'],repeat+1,result['status'],flush=True)
            if result['status']!='pass': raise RuntimeError('Canonical reference failed before experiment: '+task['id']+' '+str(result))
    return checks

def verify_isolation(c):
    host_path=str(ROOT/'.env')
    code=(
        'def probe():\n'
        '    import os, pathlib, socket\n'
        '    assert os.getuid() != 0\n'
        '    assert not any(k in os.environ for k in ("GEMINI_API_KEY","OPENROUTER_API_KEY"))\n'
        f'    assert not pathlib.Path({host_path!r}).exists()\n'
        '    assert not pathlib.Path("/mnt/c").exists()\n'
        '    try:\n'
        '        pathlib.Path("/opt/write-probe").write_text("x")\n'
        '    except OSError: pass\n'
        '    else: raise AssertionError("Root filesystem writable")\n'
        '    s=socket.socket(); s.settimeout(1)\n'
        '    try: s.connect(("1.1.1.1",443))\n'
        '    except OSError: pass\n'
        '    else: raise AssertionError("External network available")\n'
        '    finally: s.close()\n'
        '    return True\n'
    )
    task={'id':'isolation/probe','dataset':'humaneval','entry_point':'probe','tests':'def check(candidate):\n    assert candidate() is True'}
    result=evaluate(task,code,c)
    if result['status']!='pass': raise RuntimeError('Sandbox isolation probe failed: '+str(result))
    limited=dict(c,timeout_seconds=1,task_timeout_seconds=3)
    loop=evaluate(task,'def probe():\n    while True: pass',limited)
    if loop['status']!='timeout': raise RuntimeError('Sandbox timeout control failed: '+str(loop))
    wrong=evaluate(task,'def probe(): return False',c)
    if wrong['status']!='test_failure': raise RuntimeError('Benchmark failure control failed: '+str(wrong))
    return {'checked_at':time.time(),'unprivileged':True,'host_files_inaccessible':True,'no_credentials':True,
            'read_only_root':True,'external_network_disabled':True,'timeout_enforced':True,'wrong_output_rejected':True,
            'image_id':c['_sandbox']['image_id'], 'probes':[result,loop,wrong]}

def main():
    p=parser('Check pinned data, Linux isolation, optional references and selected API deployment access')
    p.add_argument('--offline',action='store_true',help='Do not contact model APIs')
    p.add_argument('--references',action='store_true',help='Execute available selected canonical programs inside isolation')
    a=p.parse_args(); c=configure_run(a); tasks,langs,models=selection(a,c)
    print('Host Python:',platform.python_version(),'; benchmark Python: 3.12.14 in Docker Linux')
    c['_sandbox']=ensure_image()
    print('Isolation:',json.dumps(verify_isolation(c)))
    print('Static metrics:',metrics('def f(x):\n    if x:\n        return x+1\n    return 0'))
    if a.references:
        checks=check_references(tasks,c)
        out=ROOT/'verification'; out.mkdir(exist_ok=True)
        (out/'current-reference-checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    if not a.offline:
        for provider,group in models_by_provider(models,c).items(): print(Client(provider,c).preflight(group))
    print('Ready.')

if __name__=='__main__': entry(main)
