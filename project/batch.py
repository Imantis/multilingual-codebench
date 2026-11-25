from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time


def run_jobs(jobs, work, save, workers=8):
    if not jobs:return
    start=time.monotonic()
    queue=iter(jobs)
    failure=None
    completed=0
    http_seconds=0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={}
        def submit():
            job=next(queue,None)
            if job is not None:pending[pool.submit(work,job)]=job
        for _ in range(min(workers,len(jobs))):submit()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                job=pending.pop(future)
                try:
                    result=future.result()
                    save(job,result,completed)
                    completed+=1
                    http_seconds+=result.get('request_metrics',{}).get('http_seconds',0)
                except Exception as error:

                    if failure is None:failure=error
            if failure is None:
                for _ in ready:submit()
    elapsed=time.monotonic()-start
    timing=f'; {http_seconds:.1f}s total HTTP time overlapped' if http_seconds else ''
    print(f'Batch: {completed}/{len(jobs)} saved in {elapsed:.1f}s{timing}; {workers} workers.',flush=True)
    if failure is not None:raise failure
