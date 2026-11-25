import io
import json
import os
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path
from .common import ROOT, digest

IMAGE = 'multilingual-codebench-evaluator:approved-v5'

def docker_binary():
    binary = shutil.which('docker')
    if not binary and os.name=='nt':
        path=Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'Docker/Docker/resources/bin/docker.exe'
        if path.exists(): binary=str(path)
    if not binary: raise RuntimeError('Install/start Docker Desktop with its Linux engine for isolated benchmark execution')
    return binary

def source_hash():
    return digest({name:digest((ROOT/path).read_bytes()) for name,path in build_files().items()})

def build_files():

    return {'Dockerfile':'project/sandbox/Dockerfile', 'runner.py':'project/sandbox/runner.py',
            'worker.py':'project/worker.py', 'requirements.txt':'requirements.txt'}

def ensure_image():
    binary=docker_binary()
    engine=subprocess.run([binary,'info','--format','{{.OSType}}'], capture_output=True,text=True,timeout=30)
    if engine.returncode or engine.stdout.strip()!='linux':
        raise RuntimeError('Start Docker Desktop and select Linux containers. '+engine.stderr.strip()[-500:])
    wanted=source_hash()
    inspected=subprocess.run([binary,'image','inspect',IMAGE],capture_output=True,text=True,timeout=30)
    current=json.loads(inspected.stdout)[0] if inspected.returncode==0 else None
    if not current or (current['Config'].get('Labels') or {}).get('codebench.source')!=wanted:
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w') as archive:
            for name,path in build_files().items():
                content=(ROOT/path).read_bytes(); info=tarfile.TarInfo(name); info.size=len(content)
                archive.addfile(info,io.BytesIO(content))
        print('Building local isolated evaluator (first run downloads pinned Python dependencies)...',flush=True)
        process=subprocess.Popen([binary,'build','--label','codebench.source='+wanted,'-t',IMAGE,'-'],stdin=subprocess.PIPE)
        process.communicate(stream.getvalue())
        if process.returncode: raise RuntimeError('Evaluator image build failed')
        current=json.loads(subprocess.check_output([binary,'image','inspect',IMAGE],text=True))[0]
    return {'image_id':current['Id'], 'source_hash':wanted,
            'evaluation_source_hash':digest({p:digest((ROOT/p).read_bytes()) for p in ('project/sandbox.py','project/evaluate.py')}), 'python':'3.12.14',
            'engine':'Docker Linux', 'network':'none', 'host_mounts':False, 'credentials':False}

def run(request, c):
    binary=docker_binary()
    image=c.get('_sandbox') or ensure_image()
    name='codebench-'+uuid.uuid4().hex
    command=[binary,'run','--rm','-i','--name',name,'--network','none','--read-only',
             '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit',str(c['pids_limit']),
             '--memory',str(c['memory_mb'])+'m','--memory-swap',str(c['memory_mb'])+'m','--cpus','1',
             '--user','65534:65534','--tmpfs','/work:rw,nosuid,nodev,size=128m,mode=1777,uid=65534,gid=65534',image['image_id']]
    payload=json.dumps(request,ensure_ascii=False).encode('utf-8')
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        out,err=process.communicate(payload,timeout=c['task_timeout_seconds']+30)
    except subprocess.TimeoutExpired:
        subprocess.run([binary,'kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
        process.kill(); process.communicate()
        return {'status':'timeout','detail':'Container exceeded total time budget'}
    except BaseException:
        subprocess.run([binary,'kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
        process.kill(); process.communicate(); raise
    if process.returncode==137: return {'status':'resource_limit','detail':'Container was killed (memory/resource limit)'}
    if process.returncode:
        return {'status':'infrastructure_error','detail':'Docker execution failed: '+err[-2000:].decode(errors='replace')}
    try: return json.loads(out)
    except ValueError: return {'status':'infrastructure_error','detail':'Invalid evaluator response: '+err[-1000:].decode(errors='replace')}
