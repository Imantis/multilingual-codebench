import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

def restrict(memory, cpu, output):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu+1))
    resource.setrlimit(resource.RLIMIT_AS, (memory*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (output*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128,)*2)

def main():
    request = json.load(sys.stdin)
    task, code, limits = request['task'], request['code'], request['limits']
    start = time.monotonic(); completed = 0
    cases = request.get('cases') or [None]
    for index, case in enumerate(cases):
        remaining = limits['task_timeout_seconds']-(time.monotonic()-start)
        if remaining <= 0:
            return {'status':'timeout', 'detail':'Total task time budget exceeded', 'tests_run':completed}
        timeout = min(limits['timeout_seconds'], remaining)
        with tempfile.TemporaryDirectory(dir='/work') as directory:
            folder = Path(directory)
            (folder/'solution.py').write_text(code, encoding='utf-8')
            (folder/'request.json').write_text(json.dumps({'task':task, 'case':case}), encoding='utf-8')
            result_file = folder/'result.json'
            env = {'PATH':'/usr/local/bin:/usr/bin:/bin', 'HOME':directory, 'TMPDIR':directory,
                   'LANG':'C.UTF-8', 'PYTHONHASHSEED':'0', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                   'MKL_NUM_THREADS':'1', 'MPLBACKEND':'Agg', 'MPLCONFIGDIR':str(folder/'mpl')}
            with (folder/'stdout').open('wb') as out, (folder/'stderr').open('wb') as err:
                process = subprocess.Popen([sys.executable,'-I','/opt/worker.py',str(folder/'request.json'),str(result_file)],
                                           cwd=folder, env=env, stdout=out, stderr=err, start_new_session=True,
                                           preexec_fn=lambda:restrict(limits['memory_mb'],max(1,math.ceil(timeout)),limits['output_limit_mb']))
                timed_out = False
                try: process.wait(timeout=timeout)
                except subprocess.TimeoutExpired: timed_out=True
                finally:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
            if timed_out or process.returncode == -signal.SIGXCPU:
                result = {'status':'timeout','detail':'Process exceeded time budget'}
            elif not result_file.exists():
                detail = (folder/'stderr').read_bytes()[-2000:].decode(errors='replace')
                result = {'status':'runtime_error','detail':f'Worker exited {process.returncode}: {detail}'}
            else:
                try: result=json.loads(result_file.read_text(encoding='utf-8'))
                except (ValueError,OSError): result={'status':'runtime_error','detail':'Worker did not return a valid result'}
            completed += result.get('tests_run', 1)
            if result['status'] != 'pass': return result|{'case_index':index,'tests_run':completed}
    return {'status':'pass','detail':'','tests_run':completed}

if __name__ == '__main__':
    try: result=main()
    except BaseException as error: result={'status':'infrastructure_error','detail':type(error).__name__+': '+str(error)}
    print(json.dumps(result), flush=True)
