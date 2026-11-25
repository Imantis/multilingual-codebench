import json
import os
import subprocess
import sys
from .common import ROOT, configure_run, entry, parser, selection, models_by_provider, run_lock
from .waiting import mark_state, resume_pending_wait, wait_for_api


def main():
    p=parser('Prepare prompts, generate code, evaluate and export thesis figures/tables')
    p.set_defaults(mode='dev',languages=None)
    p.add_argument('--evaluation-workers',type=int,choices=(1,2),default=1)
    p.add_argument('--wait-for-quota',action='store_true',
                   help='Dev only: wait for free API quotas/outages and automatically resume saved work')
    a=p.parse_args()
    if a.languages is None: a.languages='en,lv' if a.mode=='dev' else 'all'
    if a.mode=='dev' and a.limit is None: a.limit=2
    c=configure_run(a)
    if a.wait_for_quota and c['mode']!='dev':
        raise ValueError('--wait-for-quota is available only in dev mode; paid runs retain bounded retries.')
    if a.dry_run:
        run_experiment(a,c)
    else:
        with run_lock(a.workdir,name='.workflow.lock'):
            try: run_experiment(a,c)
            except Exception:
                if a.wait_for_quota: mark_state(a.workdir,'stopped')
                raise


def run_experiment(a,c):
    auto_wait=a.wait_for_quota and not a.dry_run
    tasks,languages,models=selection(a,c)
    outputs=len(tasks)*len(languages)*len(models)*3
    print(f'{c["mode"].upper()}: {len(tasks)} tasks, {len(languages)} languages, '
          f'{len(models)} model(s), {outputs} generation responses. Run: {a.workdir}',flush=True)
    if a.mode=='prod' and not a.dry_run:
        if a.max_paid_requests is None and not (a.workdir/'paid-requests.sqlite').exists():
            raise ValueError('Add --max-paid-requests to set a persistent ceiling for ALL paid inference requests, including preparation and retries.')
    if not a.dry_run:
        if auto_wait: resume_pending_wait(a.workdir)

        from .sandbox import ensure_image
        import pandas, matplotlib, radon, scipy.stats
        if any(language!='en' for language in languages) and any(t['dataset']=='bigcodebench' for t in tasks):
            try: import pyarrow.parquet
            except ImportError:
                raise ValueError('BigCodeBench replacements require requirements-preparation.txt; install it before model collection.') from None
        ensure_image()
        if not (a.workdir/'experiment.sqlite').exists():


            from .api import Client, APIError
            needed=models_by_provider(models,c)
            if any(language!='en' for language in languages):
                for role in ('translation','validator'):
                    needed.setdefault(c[role+'_provider'],[]).append(c[role+'_model'])
            for provider,group in needed.items():
                while True:
                    try:
                        Client(provider,c).preflight(group)
                        break
                    except APIError as error:
                        if not auto_wait or not error.retry: raise
                        wait_for_api(a.workdir,'preflight',error.retry)
    shared=['--mode',a.mode,'--languages',a.languages,'--model',a.model,
            '--dataset',a.dataset,'--workdir',str(a.workdir.resolve())]
    if a.limit is not None: shared+=['--limit',str(a.limit)]
    if a.max_paid_requests is not None: shared+=['--max-paid-requests',str(a.max_paid_requests)]
    if a.dry_run: shared+=['--dry-run']
    stages=('experiment_translations','experiment_code_generation','analyse')
    notice=a.workdir/'dev-api-retry.json'
    environment=None
    if auto_wait:
        environment=os.environ.copy()
        environment['CODEBENCH_RETRY_NOTICE']=str(notice.resolve())
    for number,stage in enumerate(stages,1):
        print(f'\nStage {number}/3: {stage.replace("experiment_", "").replace("_", " ")}',flush=True)
        extra=['--evaluation-workers',str(a.evaluation_workers)] if stage=='analyse' else []
        while True:
            if auto_wait: notice.unlink(missing_ok=True)
            options={'env':environment} if environment is not None else {}
            result=subprocess.run([sys.executable,'-m','project.'+stage,*shared,*extra],cwd=ROOT,**options)
            if not result.returncode:
                if auto_wait: mark_state(a.workdir,'running')
                break
            retry=json.loads(notice.read_text(encoding='utf-8')) if auto_wait and notice.exists() else None
            if not retry:
                raise RuntimeError('Stage stopped. Completed work is saved; fix the reported issue and rerun the same command. Later stages were not started.')
            wait_for_api(a.workdir,stage,retry)
    if not a.dry_run:
        reports=list((a.workdir/'reports').rglob('report.json'))
        if reports:
            path=max(reports,key=lambda x:x.stat().st_mtime)
            report=json.loads(path.read_text(encoding='utf-8'))
            print('\nReport:',path.with_suffix('.html').resolve(),flush=True)
            print('Thesis figures and tables:',(path.parent/'thesis').resolve(),flush=True)
            status=report.get('metadata',{}).get('collection_status',{})
            if status and not status['scope_complete']:
                raise RuntimeError('Report exported, but this scope still has missing responses or infrastructure faults. Rerun to resume; inspect collection-status.json.')
            print('Selected scope complete. '+('Development data are not main-study findings.' if a.mode=='dev' else
                  'Check the exported denominators and limitations when writing the results.'),flush=True)
        if auto_wait: mark_state(a.workdir,'completed')


if __name__=='__main__': entry(main)
