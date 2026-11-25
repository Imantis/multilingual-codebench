import json
from .common import ROOT,PROTOCOL,digest

def freeze_contamination(store,c):
    rows=store.get('settings',PROTOCOL+':selection') or []
    cutoffs={}
    for model in c['models'].values():
        if model.removeprefix('google/').startswith('gemma-4-'):
            cutoffs[model]={'date':'2025-01-31','precision':'month (end used conservatively)',
                           'source':'https://ai.google.dev/gemma/docs/core/model_card_4'}
        elif model=='openai/gpt-oss-20b':
            cutoffs[model]={'date':'2024-06-30','precision':'month (end used conservatively)',
                           'source':'https://deploymentsafety.openai.com/gpt-oss'}
        else: cutoffs[model]={'date':None,'source':None}
    dates=[x['date'] for x in cutoffs.values() if x['date']]
    threshold=max(dates) if len(dates)==len(cutoffs) else None
    selected=[t for t in rows if t['dataset']=='livecodebench']
    pool=selected+(store.get('settings',PROTOCOL+':reserves:livecodebench') or [])
    newer=[t for t in pool if threshold and t['contest_date'][:10]>threshold]
    value={'documented_cutoffs':cutoffs,'after_both_cutoffs':threshold,
           'qualifying_compatible_tasks_in_pinned_pool':len(newer),
           'selected_task_dates':{t['id']:t['contest_date'] for t in selected},
           'selected_date_groups':{t['id']:'qualifying_newer' if threshold and t['contest_date'][:10]>threshold else 'older_or_unknown' for t in selected},
           'contamination_uncertain':True,
           'reason':'The pinned compatible LiveCodeBench snapshot cannot supply the planned difficulty quotas after both model cutoffs; retain the attainable dated subset.' if threshold else 'Both model cutoffs are not verifiable.',
           'other_benchmarks':'HumanEval and BigCodeBench may be contaminated; no contamination-free claim.'}
    previous=store.get('settings',PROTOCOL+':contamination')
    if previous is not None and previous!=value:


        selection_fields={'selected_task_dates','selected_date_groups'}
        unchanged={k:v for k,v in previous.items() if k not in selection_fields}=={
            k:v for k,v in value.items() if k not in selection_fields}
        collected=any(r.get('protocol')==PROTOCOL and r.get('dataset')=='livecodebench' for r in store.all('generation'))
        if unchanged and not collected:
            store.put('settings',PROTOCOL+':contamination-before-selection:'+digest(previous),previous)
            store.put('settings',PROTOCOL+':contamination',value)
        else: store.freeze(PROTOCOL+':contamination',value)
    else: store.freeze(PROTOCOL+':contamination',value)
    return value

def power_record():
    folder=ROOT/'power-analysis'
    return {p.name:digest(p.read_bytes()) for p in sorted(folder.glob('*')) if p.is_file() and p.suffix in ('.py','.json','.csv','.md')}
