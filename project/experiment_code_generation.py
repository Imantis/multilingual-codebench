import json
import random
import re
import time
from .api import Client, APIError, PAID_MAX_ATTEMPTS, PAID_MAX_RETRY_WAIT_SECONDS
from .batch import run_jobs
from .common import *

def extract_code(text):
    fence=chr(96)*3
    blocks=re.findall(fence+r'(?:python|py)?[ \t]*\n([\s\S]*?)'+fence,text,flags=re.I)
    return '\n\n'.join(b.strip() for b in blocks).strip() if blocks else text.strip()

def generation_key(task, language, model, repeat, text, system):
    return digest([PROTOCOL,task['id'],language,model,repeat,digest(text),digest(system)])

def main():
    p=parser('Generate three independent Python candidates per task/language/model')
    p.add_argument('--force',action='store_true',help='Regenerate selected cells; clear only their code and dependent evaluations')
    a=p.parse_args(); c=configure_run(a); tasks,languages,models=selection(a,c)
    print(f'{len(tasks)} tasks × {len(languages)} languages × {len(models)} models × 3 repetitions = {len(tasks)*len(languages)*len(models)*3} responses')
    if a.dry_run: return
    if (ROOT/'data/.selection-in-progress').exists():
        raise RuntimeError('Benchmark selection is being prepared. Finish that preparation, then resume generation.')
    with run_lock(a.workdir):
        store=Store(a.workdir); tasks,languages,models=selection(a,c,store)
        freeze_profile(store,c)
        for alias,model in c['models'].items():
            if model in models:
                store.freeze(PROTOCOL+':model-alias:'+alias,{'model':model,'provider':provider_for_model(model,c)})
        store.freeze(PROTOCOL+':generation',{'repetitions':3,'seed':c['seed'],'max_tokens':c['max_tokens'],
                                            'dataset_hash':digest((ROOT/'data/tasks.jsonl').read_bytes())})
        if store.get('settings',PROTOCOL+':selection') is None:
            store.freeze(PROTOCOL+':selection',[json.loads(x) for x in (ROOT/'data/tasks.jsonl').read_text(encoding='utf-8').splitlines()])
        prompts={(t['id'],l):(prompt(t,l,store),system_prompt(l,store)) for t in tasks for l in languages}
        jobs=[]
        for repeat in range(3):
            round_jobs=[]
            for task in tasks:
                for language in languages:
                    text,system=prompts[task['id'],language]
                    for model in models:
                        key=generation_key(task,language,model,repeat,text,system)
                        round_jobs.append((task,language,model,repeat,key,text,system))
            random.Random(c['seed']+repeat).shuffle(round_jobs)
            jobs.extend(job for job in round_jobs if a.force or not store.get('generation',job[4]))
        if not jobs:
            from .analyse import export_intermediates
            export_intermediates(tasks,languages,models,store,a.workdir/'reports')
            print('All three repetitions already saved; no API calls.',flush=True); store.close(); return
        store.freeze(PROTOCOL+':analysis-environment',analysis_environment())
        clients={}
        for provider, group in models_by_provider([job[2] for job in jobs],c).items():
            client=Client(provider,c)
            for model in group:
                previous=store.get('settings',PROTOCOL+':deployment:'+model)
                if previous and provider=='openrouter': c['openrouter_backends'][model]=previous['backend']
            client.preflight(group)
            for model in group: store.freeze(PROTOCOL+':deployment:'+model,client.catalog[model])
            clients[provider]=client

        for job in jobs:
            task,language,model,repeat,key,text,system=job
            deployment=clients[provider_for_model(model,c)].catalog[model]
            if len((text+system).encode('utf-8'))+512+c['max_tokens']>deployment['context_length']:
                raise ValueError('Prompt may exceed context window: '+task['id']+' '+language)
        if a.force:
            print('Backup:',store.backup('generation-force'),flush=True)
            with store.db: removed=store.clear_generations([t['id'] for t in tasks],languages,models)
            print('Cleared selected responses:',removed,flush=True)
        limits={k:c[k] for k in ('timeout_seconds','task_timeout_seconds','memory_mb','pids_limit','output_limit_mb')}
        from .evaluate import METRICS,VERSION
        from .sandbox import ensure_image
        c['_sandbox']=ensure_image()
        store.freeze(PROTOCOL+':execution',limits|c['_sandbox']|{'metrics':list(METRICS),'evaluation_version':VERSION})

        from .doctor import check_references, verify_isolation
        reference_key=PROTOCOL+':references:'+digest([t['id'] for t in tasks])
        if store.get('settings',reference_key) is None:
            reference_checks=check_references(tasks,c,store)
            store.freeze(reference_key,reference_checks)
        if store.get('settings',PROTOCOL+':isolation') is None:
            store.freeze(PROTOCOL+':isolation',verify_isolation(c))
        from .protocol import freeze_contamination,power_record
        freeze_contamination(store,c)
        store.freeze(PROTOCOL+':analysis',{'script_hash':digest((ROOT/'project/statistics.py').read_bytes()),
                                         'bootstrap_samples':c['bootstrap_samples'],'holm_family_size':8,'power_files':power_record()})
        protocol_file=a.workdir/'protocol.json'
        previous_protocol=json.loads(protocol_file.read_text(encoding='utf-8')) if protocol_file.exists() else {}
        protocol={r[0]:json.loads(r[1]) for r in store.db.execute("SELECT cache_key,payload_json FROM run_settings WHERE cache_key LIKE ?",(PROTOCOL+':%',))}
        protocol['collection_settings']=previous_protocol.get('collection_settings') or {
                                        'harness_sha256':digest((ROOT/'project/worker.py').read_bytes()),
                                        'analysis_sha256':digest((ROOT/'project/statistics.py').read_bytes()),
                                        'frozen_at':time.time(), 'scope_languages':languages,'scope_models':models,
                                        'development_scope':a.limit is not None or a.dataset!='all' or a.languages!='all' or a.model!='all',
                                        'task_prompt_hashes':{t['id']:{l:digest(prompts[t['id'],l]) for l in languages} for t in tasks}}
        protocol['collection_batches']=previous_protocol.get('collection_batches',[])+[{
            'started_at':time.time(),'languages':languages,'models':models,
            'task_ids':[t['id'] for t in tasks],'selected_tasks_sha256':digest(tasks),
            'task_prompt_hashes':{t['id']:{l:digest(prompts[t['id'],l]) for l in languages} for t in tasks},
            'requested_responses':len(jobs),'force':a.force,
            'transport':{'workers':c['request_workers'],'provider_rpm':{p:c[p+'_rpm'] for p in clients},
                         'max_attempts':c['max_attempts'],'paid_max_attempts':min(PAID_MAX_ATTEMPTS,c['max_attempts']),
                         'paid_retry_backoff_seconds':[30,60,120,240],
                         'paid_max_retry_wait_seconds':PAID_MAX_RETRY_WAIT_SECONDS,
                         'ambiguous_paid_response_retry':False,'paid_request_ledger':c.get('_paid_request_ledger'),
                         'http_timeout_seconds':300},
            'implementation_sha256':{p:digest((ROOT/'project'/p).read_bytes()) for p in
                                     ('api.py','common.py','experiment_code_generation.py','protocol.py')}}]
        protocol_file.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
        def work(job):
            task,language,model,repeat,key,text,system=job
            seed=int(digest([c['seed'],task['id'],language,model,repeat])[:8],16)&0x7fffffff
            try: result=clients[provider_for_model(model,c)].generate(model,text,system,seed=seed)
            except APIError as error:
                return {'transport_error':str(error), 'retry':error.retry, 'created_at':time.time()}
            result.update(protocol=PROTOCOL,key=key,task_id=task['id'],dataset=task['dataset'],language=language,repeat=repeat,
                          prompt_hash=digest(text),code=extract_code(result['text']),
                          generation_status='no_text' if result['no_text'] else 'generated')
            return result
        saved_count=0
        def save(job,result,index):
            nonlocal saved_count
            task,language,model,repeat,key,text,system=job
            if result.get('transport_error'):
                store.put('settings',PROTOCOL+':generation-error:'+key+':'+str(time.time_ns()),result|{'task_id':task['id'],'language':language,'model':model,'repeat':repeat})
                raise APIError(result['transport_error'],retry=result.get('retry'))
            if result['known_cached_response']:
                store.put('generation',key+':cached:'+str(time.time_ns()),result|{'excluded':True})
                raise APIError('Known cached generation saved but excluded; rerun to request a fresh response')
            observed=result.get('observed_revision')
            expected=store.get('settings',PROTOCOL+':observed-revision:'+model)
            backend=result.get('observed_backend')
            deployment=result['deployment']
            if result['provider']=='openrouter' and backend and backend!=deployment['provider_name']:
                store.put('generation',key+':unexpected-backend:'+str(time.time_ns()),result|{'excluded':True})
                raise APIError('Response used an unexpected backend; start a separate full model run')
            if expected is not None and observed is not None and expected.get('revision')!=observed:
                store.put('generation',key+':unexpected-revision:'+str(time.time_ns()),result|{'excluded':True})
                raise APIError('Observed model revision changed; start a separate full model run')
            if observed is not None: store.freeze(PROTOCOL+':observed-revision:'+model,{'revision':observed})
            store.put('generation',key,result)
            saved_count+=1
            print(f'Saved {saved_count}/{len(jobs)} {task["id"]} {language} {model} repeat={repeat+1}: {result["finish_reason"]}',flush=True)

        try:
            for repeat in range(3):
                part=[job for job in jobs if job[3]==repeat]
                if part:
                    print('Generation round',repeat+1,flush=True)
                    run_jobs(part,work,save,c['request_workers'])
        finally:
            from .analyse import export_intermediates
            export_intermediates(tasks,languages,models,store,a.workdir/'reports')
            print('Saved code and requests:',a.workdir/'reports/generated-code.jsonl',flush=True)
        print('Generation complete.',flush=True); store.close()

if __name__=='__main__': entry(main)
