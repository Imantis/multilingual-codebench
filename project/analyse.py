import html
import json
import platform
import shutil
from importlib.metadata import version as package_version
from datetime import datetime,timezone
import pandas as pd
from .common import *
from .evaluate import evaluate,metrics,VERSION,METRICS
from .sandbox import ensure_image
from .statistics import analyses,summarize,stability
from .api import api_cost_summary
from .batch import run_jobs

def export_intermediates(tasks,languages,models,store,folder):
    from .prompts import CODE_SYSTEM_PROMPT
    folder.mkdir(parents=True,exist_ok=True)
    sources={t['id']:t['prompt'] for t in tasks}|{'__instructions__':CODE_SYSTEM_PROMPT}
    prepared=[]
    for task_id,source in sources.items():
        for language in languages:
            if language=='en':
                prepared.append({'task_id':task_id,'language':'en','source_english':source,'translation':source,
                                 'source_original':True,'accepted':True,'preparation_state':'source_original'})
                continue
            row=store.get('translation',translation_key({'id':task_id},language)) or {}
            accepted=translation_is_current({'prompt':source},row)
            back=store.get('backtranslation',row['backtranslation_key']) if row.get('backtranslation_key') else None
            audit=store.get('validation',row['audit_key']) if row.get('audit_key') else None
            state='accepted' if accepted else ('missing' if not row else ('outdated' if row.get('source_hash')!=digest(source)
                    else ('rejected' if row.get('checked_at') is not None else 'pending')))
            prepared.append({'task_id':task_id,'language':language,'source_english':source,
                             'translation':row.get('text'),'backtranslation':(back or {}).get('text'),
                             'source_original':False,'accepted':accepted,'preparation_state':state,'repair':row.get('repair'),
                             'audit_criteria':(audit or {}).get('criteria'),'audit_error':(audit or {}).get('audit_error'),
                             'translation_model':row.get('model'),'audit_model':(audit or {}).get('model')})
    (folder/'prepared-prompts.json').write_text(json.dumps(prepared,ensure_ascii=False,indent=2),encoding='utf-8')
    translations=[r for r in prepared if not r.get('source_original')]
    prep_summary=[]
    for language in languages:
        if language=='en': continue
        part=[r for r in translations if r['language']==language]
        prep_summary.append({'language':language,'planned_translations_including_instructions':len(part),
                             **{state:sum(r['preparation_state']==state for r in part) for state in ('accepted','missing','pending','rejected','outdated')},
                             **{'accepted_after_'+str(k)+'_repairs':sum(r['accepted'] and r['repair']==k for r in part) for k in (0,1,2)}})
    pd.DataFrame(prep_summary).to_csv(folder/'preparation-summary.csv',index=False)

    old_ids={r['old_id'] for r in (store.get('settings',PROTOCOL+':replacements') or []) if r['dataset'] in {t['dataset'] for t in tasks}}
    with (folder/'preparation-attempts.jsonl').open('w',encoding='utf-8') as output:
        for kind in ('translation_error','backtranslation','validation','validation_error'):
            for row in store.all(kind):
                if row.get('protocol')==PROTOCOL and row.get('task_id') in sources.keys()|old_ids and row.get('language') in languages:
                    output.write(json.dumps({'record_kind':kind,'in_final_selection':row['task_id'] in sources,**row},ensure_ascii=False)+'\n')
    fields=('key','task_id','dataset','language','model','provider','repeat','code','text','request',
            'raw','settings','request_seed','finish_reason','observed_revision','observed_backend','usage','seconds',
            'request_metrics','generation_status','created_at')
    with (folder/'generated-code.jsonl').open('w',encoding='utf-8') as output:
        for row in active_generations(store):
            if row['task_id'] in sources and row['language'] in languages and row['model'] in models:
                output.write(json.dumps({k:row.get(k) for k in fields},ensure_ascii=False)+'\n')
    (folder/'intermediate-scope.json').write_text(json.dumps({'task_ids':[t['id'] for t in tasks],
        'languages':languages,'models':models,'protocol':PROTOCOL,'updated_at':time.time()},indent=2),encoding='utf-8')
    return {'planned':len(translations),'accepted':sum(r['accepted'] for r in translations),'by_language':prep_summary}


def collection_status(records,tasks,languages,models,c,preparation):
    expected={(t['id'],language,model,repeat) for t in tasks for language in languages for model in models for repeat in range(3)}
    actual=[(r['task_id'],r['language'],r['model'],r['repeat']) for r in records]
    if len(actual)!=len(set(actual)) or not set(actual)<=expected:
        raise ValueError('Duplicate or unexpected task/language/model/repetition slots; analysis stopped to avoid incorrect denominators.')
    missing=len(expected-set(actual)); infrastructure=sum(r['status']=='infrastructure_error' for r in records)
    counts={ds:sum(t['dataset']==ds for t in tasks) for ds in DATASETS}
    full_scope=counts==dict.fromkeys(DATASETS,50) and set(languages)==set(c['languages']) and set(models)==set(c['models'].values())
    complete=missing==0 and infrastructure==0 and preparation['planned']==preparation['accepted']
    return {'scope_complete':complete,'full_study_scope':full_scope,'development_only':c['mode']=='dev',
            'main_study_collection_complete':c['mode']=='prod' and full_scope and complete,
            'planned_outputs':len(expected),'saved_outputs':len(actual),'missing_api_outputs':missing,
            'infrastructure_outputs':infrastructure,'evaluated_outputs':len(actual)-infrastructure,
            'returned_functional_failures':sum(r['status'] not in ('pass','infrastructure_error') for r in records),
            'tasks_by_benchmark':counts,'preparation':preparation,
            'note':'Completeness means all planned slots are available and evaluated; it does not mean every program passes. Development and partial-scope data are not full-study findings.'}

def report(records,tasks,languages,models,store,folder,c):
    if not records: raise ValueError('No current-protocol generations match these filters')
    folder.mkdir(parents=True,exist_ok=True)
    preparation=export_intermediates(tasks,languages,models,store,folder)
    status=collection_status(records,tasks,languages,models,c,preparation)
    df=pd.DataFrame(records); df.to_csv(folder/'results.csv',index=False)
    diagnostics=[]
    for (model,dataset,language),part in df.groupby(['model','dataset','language']):
        row={'model':model,'dataset':dataset,'language':language,'saved_responses':len(part)}
        for field in ('input_tokens','output_tokens','combined_prompt_utf8_bytes','generation_seconds'):
            values=pd.to_numeric(part[field],errors='coerce').dropna()
            row[field+'_available']=len(values)
            row[field+'_median']=float(values.median()) if len(values) else None
        diagnostics.append(row)
    pd.DataFrame(diagnostics).to_csv(folder/'deployment-diagnostics.csv',index=False)
    summary,difficulty=summarize(records,tasks,languages,models)
    stats=analyses(records,tasks,languages,models,c)
    repeats=stability(records)
    static_summary=[]
    for (dataset,language,model),part in df.groupby(['dataset','language','model']):
        for metric in METRICS:
            for passing_only in (False,True):
                eligible=part if not passing_only else part[part.status=='pass']
                defined=eligible[metric].notna(); values=eligible.loc[defined,metric].astype(float)
                static_summary.append({'dataset':dataset,'language':language,'model':model,'metric':metric,
                                       'subset':'test_passing' if passing_only else 'analyzable',
                                       'generated_outputs':len(part),'eligible_outputs':len(eligible),
                                       'defined_outputs':len(values),'tasks_with_metric':int(eligible.loc[defined,'task_id'].nunique()),
                                       **{name:float(values.quantile(q)) if len(values) else None for name,q in
                                          [('min',0),('q25',.25),('median',.5),('q75',.75),('max',1)]}})
    pd.DataFrame(static_summary).to_csv(folder/'static-summary.csv',index=False)
    table=pd.DataFrame(summary); table.to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(difficulty).to_csv(folder/'difficulty.csv',index=False)
    pd.DataFrame(stats['static_comparisons']).to_csv(folder/'static-comparisons.csv',index=False)
    pd.DataFrame(stats['model_benchmark_comparisons']).to_csv(folder/'language-comparisons.csv',index=False)
    for filename,rows in [('primary-comparisons.csv',stats['primary_english_comparisons']),
                          ('difficulty-comparisons.csv',stats['difficulty_comparisons']),('interactions.csv',stats['interactions'])]:
        pd.DataFrame(rows).to_csv(folder/filename,index=False)
    missing_reasons=[]
    for row in records:
        for metric,reason in row.get('metric_missing_reasons',{}).items():
            missing_reasons.append({k:row[k] for k in ('task_id','dataset','language','model','repeat')}|{'metric':metric,'reason':reason})
    pd.DataFrame(missing_reasons,columns=['task_id','dataset','language','model','repeat','metric','reason']).to_csv(folder/'metric-missing-reasons.csv',index=False)
    pd.DataFrame([{k:t.get(k) for k in ('id','dataset','entry_point','difficulty','contest_date','source_hash','cases_sha256')}
                  for t in tasks]).to_csv(folder/'selected-tasks.csv',index=False)
    audits=[r for r in store.all('validation') if r.get('protocol')==PROTOCOL and r.get('language') in languages]
    for filename,value in [('statistics.json',stats),('stability.json',repeats),('translation-validation.json',audits)]:
        (folder/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    metadata={'generated_at':datetime.now(timezone.utc).isoformat(),'host_python':platform.python_version(),
              'package_versions':{name:package_version(name) for name in ('numpy','scipy','pandas','matplotlib','radon')},
              'mode':c['mode'],'development_only':c['mode']=='dev',
              'protocol':PROTOCOL,'expected_outputs':len(tasks)*len(languages)*len(models)*3,'observed_outputs':len(records),
              'report_script_sha256':digest(Path(__file__).read_bytes()),'collection_status':status,
              'thesis_export_sha256':digest((ROOT/'project/thesis_export.py').read_bytes()),
              'languages':languages,'models':models,'datasets':list(dict.fromkeys(t['dataset'] for t in tasks)),
              'task_ids':[t['id'] for t in tasks],'repetitions':3,'evaluation_version':VERSION,
              'execution':c['_sandbox'],'dataset_sha256':digest((ROOT/'data/tasks.jsonl').read_bytes()),
              'selected_tasks_sha256':digest(tasks),
              'frozen_selection_sha256':digest(store.get('settings',PROTOCOL+':selection') or tasks),
              'replacements':store.get('settings',PROTOCOL+':replacements') or [],
              'livecodebench_reserve_extension':store.get('settings',PROTOCOL+':livecodebench-reserve-extension'),
              'source_corrections':store.get('settings',PROTOCOL+':source-corrections') or [],
              'contamination':store.get('settings',PROTOCOL+':contamination'),
              'frozen_analysis_environment':store.get('settings',PROTOCOL+':analysis-environment'),
              'deployments':{m:store.get('settings',PROTOCOL+':deployment:'+m) for m in models}}
    (folder/'collection-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'metadata':metadata,'summary':summary,'difficulty':difficulty,'statistics':stats,'stability':repeats,
            'static_summary':static_summary,'deployment_diagnostics':diagnostics,
            'api_costs':api_cost_summary(store),
            'failure_counts':df.groupby(['dataset','model','language','status']).size().rename('count').reset_index().to_dict('records')}
    (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    from .thesis_export import export
    thesis=export(result,folder/'thesis')
    shutil.copyfile(folder/'thesis/figures/functional-correctness.png',folder/'pass-rates.png')
    def to_table(value):
        if not value: return '<p>No eligible comparisons.</p>'
        frame=pd.DataFrame(value)
        for column in ('pass_at_1','pass_at_3','language_analyzability_rate','english_analyzability_rate',
                       'language_functional_failure_rate','english_functional_failure_rate'):
            if column in frame: frame[column]=frame[column]*100
        if 'planned_pass_at_1_bounds' in frame:
            frame['planned_pass_at_1_bounds']=frame['planned_pass_at_1_bounds'].map(lambda b:[100*x for x in b] if isinstance(b,list) else None)
        names={'pass_at_1':'Pass@1 (%)','pass_at_3':'Pass@3 (%)',
               'planned_pass_at_1_bounds':'Planned Pass@1 bounds (%)',
               'language_analyzability_rate':'Language analyzability (%)','english_analyzability_rate':'English analyzability (%)',
               'language_functional_failure_rate':'Language functional failures (%)','english_functional_failure_rate':'English functional failures (%)'}
        frame=frame.rename(columns={col:names.get(col,col.replace('_',' ').capitalize()) for col in frame.columns})
        return frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')
    def percentage_point_rows(rows):
        return [{k:v for k,v in r.items() if k not in ('mean_difference','ci95')}|{
            'difference_pp':100*r['mean_difference'] if r.get('mean_difference') is not None else None,
            'pointwise_95_interval_pp':[100*x for x in r['ci95']] if r.get('ci95') is not None else None} for r in rows]
    primary=[{k:r.get(k) for k in ('language','effect_percentage_points','p_value','p_holm','testable','reason')}|{
        'pointwise_95_interval_pp':[100*x for x in r['ci95']] if r.get('ci95') else None,
        'complete_pairs_by_benchmark':{s['dataset']:s['pairs'] for s in r['strata']},
        'whole_planned_sample_bounds_pp':[100*x for x in r['whole_planned_sample_bounds']] if r.get('whole_planned_sample_bounds') else None}
        for r in stats['primary_english_comparisons']]
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Multilingual code experiment</title>
<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 24px;color:#192b40}
table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}
img{max-width:100%}.notice{padding:18px;background:#eef4fa}</style><h1>Multilingual Python generation</h1>'''
    body+=f'<p>{html.escape(metadata["generated_at"])}</p><div class="notice"><strong>{len(records)} / {metadata["expected_outputs"]} planned responses saved.</strong> '
    body+='Results cover the selected benchmark tasks. Pass@1 and Pass@3 use task conditions with all three evaluated outcomes. Missing API responses and infrastructure faults are reported separately; returned invalid code is a functional failure.</div>'
    if metadata['development_only']: body+='<p><strong>DEVELOPMENT RUN — these are not main-study findings.</strong></p>'
    elif not status['full_study_scope']: body+='<p><strong>Partial study scope: this report does not cover the full approved design.</strong></p>'
    body+='<p>Collection '+('complete for this scope.' if status['scope_complete'] else '<strong>incomplete for this scope.</strong>')+' See <a href="collection-status.json">counts and preparation status</a>.</p>'
    body+='<h2>Functional correctness</h2><img src="pass-rates.png" alt="Pass@1 and descriptive Pass@3">'
    body+=to_table(summary)
    body+='<h2>Recorded API costs</h2>'+to_table(result['api_costs']['groups'])+'<p>'+html.escape(result['api_costs']['note'])+'</p>'
    body+='<h2>Token and deployment diagnostics</h2><p>Per-saved-response medians, with available counts. Token counts use provider reports; output tokens may include internal reasoning. '
    body+='Combined prompt bytes cover task and generation-instruction text, excluding API framing. Generation seconds include pacing and retries within the retained request, '
    body+='and exclude earlier unsuccessful collection attempts. Available model cutoff metadata is in report.json; pretraining-data volumes are not inferred.</p>'+to_table(diagnostics)
    body+='<h2>Primary language–English differences in Pass@1</h2><p>A difference measures change relative to English; it is not the correctness score itself. '
    body+='The primary test requires both models and all three benchmarks. Its eight-comparison Holm family stays fixed when report filters change. Unavailable tests are shown explicitly.</p>'+to_table(primary)
    body+='<h2>Exploratory paired comparisons</h2><p>Correctness differences and intervals use percentage points (pp).</p>'+to_table(percentage_point_rows(stats['model_benchmark_comparisons']))
    body+='<h3>Model and benchmark interactions</h3><p>Differences between language-English contrasts, in percentage points.</p>'+to_table(percentage_point_rows(stats['interactions']))
    if difficulty: body+='<h2>LiveCodeBench difficulty</h2>'+to_table(difficulty)+to_table(percentage_point_rows(stats['difficulty_comparisons']))
    body+='<h2>Static code characteristics</h2><p>Maximum function/method cyclomatic complexity, module Halstead volume, module source lines of code and maximum function/method control nesting. '
    body+='Undefined values remain missing. Comparisons average defined repetitions within common tasks and are repeated for test-passing outputs. These metrics do not measure overall code quality.</p>'+to_table(stats['static_comparisons'])
    body+='<h3>Descriptive distributions</h3><p>Defined metrics in all analyzable outputs and, separately, test-passing outputs. These distributions are descriptive and unpaired.</p>'+to_table(static_summary)
    body+='<h2>Repeat stability</h2>'+to_table(repeats)
    body+='<h2>Methods and limitations</h2><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in stats['notes'])+'</ul>'
    body+='<p>Generated code is evaluated as received in isolated Linux containers with no network, host mounts or credentials. '
    body+='Historical benchmark contamination remains possible. LiveCodeBench uses the compatible function subset and all supplied Lite cases, rather than official leaderboard scoring. '
    body+='The automatic translation checks do not prove perfect prompt equivalence.</p>'
    body+='<h2>Thesis figures and tables</h2><p><a href="thesis/README.md">Overleaf instructions</a> · <a href="thesis/figures.tex">Figure environments and captions</a> · <a href="thesis/tables.tex">Detailed LaTeX tables</a></p>'
    for figure in thesis['figures']:
        if figure['name']=='functional-correctness': continue
        body+='<h3>'+html.escape(figure['name'].replace('-',' ').replace('_',' '))+'</h3><a href="thesis/'+figure['pdf']+'">Vector PDF</a><br><img src="thesis/'+figure['png']+'" alt="'+html.escape(figure['caption'])+'"><p>'+html.escape(figure['caption'])+'</p>'
    body+='<h2>Files</h2><p><a href="results.csv">Per-output results</a> · <a href="summary.csv">Correctness summary</a> · <a href="statistics.json">All comparisons</a> · '
    body+='<a href="prepared-prompts.json">Prepared translations and checks</a> · <a href="generated-code.jsonl">Generated code and requests</a> · '
    body+='<a href="preparation-attempts.jsonl">Preparation attempts, including rejected drafts</a> · '
    body+='<a href="static-summary.csv">Static metric distributions</a> · '
    body+='<a href="stability.json">Repeat stability</a> · <a href="translation-validation.json">Translation checks</a> · <a href="report.json">Complete report</a></p></html>'
    (folder/'report.html').write_text(body,encoding='utf-8')
    return folder/'report.html'

def evaluation_key(record,task,c):
    return digest([PROTOCOL,record['key'],digest(record['code']),task,VERSION,c['_sandbox'],
                   {k:c[k] for k in ('timeout_seconds','task_timeout_seconds','memory_mb','pids_limit','output_limit_mb')}])

def main():
    p=parser('Evaluate saved candidates in Linux isolation and report all three repetitions')
    p.add_argument('--force',action='store_true',help='Reevaluate selected saved code; no model calls')
    p.add_argument('--evaluation-workers',type=int,choices=(1,2),default=1,help='Local isolated evaluations at once; two require capacity for two 2 GiB containers')
    a=p.parse_args(); c=configure_run(a); tasks,languages,models=selection(a,c)
    if a.dry_run: print('Analyse',len(tasks),'tasks;',languages,models,'with three repetitions'); return
    with run_lock(a.workdir):
        store=Store(a.workdir); tasks,languages,models=selection(a,c,store); by_id={t['id']:t for t in tasks}
        freeze_profile(store,c)
        environment=store.get('settings',PROTOCOL+':analysis-environment')
        if environment and environment!=analysis_environment():
            raise ValueError('Frozen Python/statistical/static-analysis tool versions changed; restore them before combining evaluations.')
        generations=[r for r in active_generations(store) if not r.get('excluded') and r['task_id'] in by_id
                     and r['language'] in languages and r['model'] in models]
        if not generations: raise ValueError('No saved current-protocol code matches these filters')
        c['_sandbox']=ensure_image()
        analysis=store.get('settings',PROTOCOL+':analysis')
        if analysis and (analysis['script_hash']!=digest((ROOT/'project/statistics.py').read_bytes()) or analysis['bootstrap_samples']!=c['bootstrap_samples']):
            raise ValueError('The frozen analysis changed; use a separately identified analysis rather than silently overwriting it.')
        frozen=store.get('settings',PROTOCOL+':execution')
        if frozen:
            current={k:c[k] for k in ('timeout_seconds','task_timeout_seconds','memory_mb','pids_limit','output_limit_mb')}|c['_sandbox']|{'metrics':list(METRICS),'evaluation_version':VERSION}
            if frozen!=current: raise ValueError('Execution image, definitions or limits changed. Use a separate run instead of mixing outcomes.')
        if a.force:
            print('Backup:',store.backup('analysis-force'),flush=True)
            keys=[(r['key'],) for r in generations]
            with store.db: store.db.executemany("DELETE FROM evaluations WHERE json_extract(payload_json,'$.generation_key')=?",keys)
        records=[]; new=0; evaluations={}; jobs=[]
        for r in generations:
            task=by_id[r['task_id']]; key=evaluation_key(r,task,c); ev=store.get('evaluation',key)
            if not ev or ev['status']=='infrastructure_error':
                jobs.append((r,task,key))
            else: evaluations[key]=ev
        def work(job):
            r,task,key=job
            ev=evaluate(task,r['code'],c)|metrics(r['code'])
            ev.update(generation_key=r['key'],task_id=r['task_id'],dataset=r['dataset'],language=r['language'],
                      model=r['model'],provider=r['provider'],protocol=PROTOCOL)
            return ev
        def save(job,ev,index):
            nonlocal new
            r,task,key=job
            store.put('evaluation',key,ev); evaluations[key]=ev; new+=1
            print(f'Evaluated {index+1}/{len(jobs)} {task["id"]} repeat={r["repeat"]+1}: {ev["status"]}',flush=True)
        run_jobs(jobs,work,save,a.evaluation_workers)
        for r in generations:
            ev=evaluations[evaluation_key(r,by_id[r['task_id']],c)]
            usage=r.get('usage') or {}
            records.append({k:r[k] for k in ('task_id','dataset','language','model','provider','repeat')}|ev|{
                'code_hash':digest(r['code']),'response_hash':digest(r['text']),'generation_seconds':r['seconds'],
                'finish_reason':r['finish_reason'],'observed_revision':r.get('observed_revision'),
                'observed_backend':r.get('observed_backend'),'request_seed':r.get('request_seed'),
                'input_utf8_bytes':r.get('input_utf8_bytes'),
                'combined_prompt_utf8_bytes':len((r['prompt']+r['system_prompt']).encode('utf-8')) if isinstance(r.get('prompt'),str) and isinstance(r.get('system_prompt'),str) else None,
                'input_tokens':usage.get('promptTokenCount',usage.get('prompt_tokens')),
                'output_tokens':usage.get('candidatesTokenCount',usage.get('completion_tokens')),
                'total_tokens':usage.get('totalTokenCount',usage.get('total_tokens')),
                'transport_attempts':r.get('request_metrics',{}).get('attempts'),
                'sampling_temperature':r['settings'].get('temperature'),'difficulty':task.get('difficulty')})
        folder=a.workdir/'reports'
        try: path=report(records,tasks,languages,models,store,folder,c)
        except PermissionError:
            folder=folder/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            path=report(records,tasks,languages,models,store,folder,c)
        print(f'{new} new evaluations; {len(records)-new} reused. Report: {path}',flush=True); store.close()

if __name__=='__main__': entry(main)
