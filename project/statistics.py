import itertools
import numpy as np
from scipy.stats import t as student_t
from .common import DATASETS, digest
from .evaluate import METRICS

def holm(pvalues):

    ordered=sorted(range(len(pvalues)),key=lambda i:1. if pvalues[i] is None else pvalues[i])
    adjusted=[None]*len(pvalues); previous=0.
    for rank,index in enumerate(ordered):
        value=1. if pvalues[index] is None else pvalues[index]
        previous=max(previous,min(1.,(len(pvalues)-rank)*value))
        if pvalues[index] is not None: adjusted[index]=previous
    return adjusted

def stratified_test(parts):
    result={'testable':False,'strata':[], 'effect':None,'se':None,'df':None,'p_value':None,'ci95':None}
    for dataset in DATASETS:
        values=np.asarray(parts.get(dataset,[]),dtype=float)
        result['strata'].append({'dataset':dataset,'pairs':len(values),'mean':float(values.mean()) if len(values) else None,
                                 'variance':float(values.var(ddof=1)) if len(values)>1 else None})
    if any(s['pairs']<2 for s in result['strata']):
        result['reason']='Requires complete language-English pairs in both models and at least two tasks in each of the three benchmarks'
        return result
    effect=sum(s['mean'] for s in result['strata'])/3
    variances=np.array([s['variance']/(9*s['pairs']) for s in result['strata']])
    se=float(np.sqrt(variances.sum()))
    result.update(effect=effect,se=se,effect_percentage_points=100*effect)
    denominator=sum(v*v/(s['pairs']-1) for v,s in zip(variances,result['strata']))
    if se==0 or denominator==0:
        result['reason']='Zero contrast variance; not testable'; return result
    df=float(variances.sum()**2/denominator)
    quantile=float(student_t.ppf(.975,df))
    result.update(testable=True,df=df,p_value=float(2*student_t.sf(abs(effect/se),df)),
                  ci95=[effect-quantile*se,effect+quantile*se])
    return result

def bootstrap(parts, seed, samples=10000):

    arrays=[np.asarray(p,dtype=float) for p in parts if len(p)]
    if not arrays or any(len(p)<2 for p in arrays): return None
    rng=np.random.default_rng(seed); draws=np.zeros(samples)
    for values in arrays:

        for begin in range(0,samples,500):
            size=min(500,samples-begin)
            indexes=rng.integers(0,len(values),size=(size,len(values)))
            draws[begin:begin+size]+=values[indexes].mean(axis=1)/len(arrays)
    return [float(x) for x in np.quantile(draws,[.025,.975])]

def stable_seed(seed, scope):
    return int(digest([seed,scope])[:16],16)

def summarize(records,tasks,languages,models):
    by_task={t['id']:t for t in tasks}
    groups={}
    for r in records:
        groups.setdefault((r['model'],r['dataset'],r['language'],r['task_id']),[]).append(r)
    rows=[]; difficulty=[]
    datasets=list(dict.fromkeys(t['dataset'] for t in tasks))
    for model in models:
        for dataset in datasets:
            for language in languages:
                selected=[t for t in tasks if t['dataset']==dataset]
                def summary(part,level=None):
                    outputs=[r for t in part for r in groups.get((model,dataset,language,t['id']),[])]
                    complete=[]
                    for task in part:
                        cell=groups.get((model,dataset,language,task['id']),[])
                        usable=[r for r in cell if r['status']!='infrastructure_error']
                        if len(usable)==3 and {r['repeat'] for r in usable}=={0,1,2}: complete.append(usable)
                    known=[r for r in outputs if r['status']!='infrastructure_error']
                    successes=sum(r['status']=='pass' for r in known)
                    slots=len(part)*3
                    result={'model':model,'dataset':dataset,'language':language,'planned_tasks':len(part),
                            'generated_outputs':len(outputs),'missing_api_outputs':slots-len(outputs),
                            'infrastructure_outputs':len(outputs)-len(known),'evaluated_outputs':len(known),
                            'complete_tasks':len(complete),'passed_outputs':successes,
                            'pass_at_1':float(np.mean([sum(r['status']=='pass' for r in cell)/3 for cell in complete])) if complete else None,
                            'pass_at_3':float(np.mean([any(r['status']=='pass' for r in cell) for cell in complete])) if complete else None,
                            'planned_pass_at_1_bounds':[successes/slots,(successes+slots-len(known))/slots] if slots else None}
                    if level: result['difficulty']=level
                    return result
                rows.append(summary(selected))
                if dataset=='livecodebench':
                    for level in ('easy','medium','hard'):
                        difficulty.append(summary([t for t in selected if t['difficulty'].lower()==level],level))
    return rows,difficulty

def analyses(records,tasks,languages,models,c):
    lookup={(r['task_id'],r['language'],r['model'],r['repeat']):r for r in records}
    def rows(task,language,model):
        return [lookup.get((task['id'],language,model,k)) for k in range(3)]
    def outcomes(task,language,model):
        cell=rows(task,language,model)
        if any(r is None or r['status']=='infrastructure_error' for r in cell): return None
        return np.array([float(r['status']=='pass') for r in cell])
    def difference(task,language,model,kind='pass_at_1'):
        a,b=outcomes(task,language,model),outcomes(task,'en',model)
        if a is None or b is None: return None
        return float(a.mean()-b.mean()) if kind=='pass_at_1' else float(a.max()-b.max())
    datasets=list(dict.fromkeys(t['dataset'] for t in tasks))
    family=[l for l in c['languages'] if l!='en']
    primary_models=list(c['models'].values())
    primary=[]
    for language in family:
        parts={ds:[] for ds in DATASETS}; bounds=[]; planned=[]
        for dataset in DATASETS:
            subset=[t for t in tasks if t['dataset']==dataset]; planned.append(len(subset))
            lower=[]; upper=[]
            for task in subset:
                values=[difference(task,language,model) for model in primary_models]
                if all(v is not None for v in values): parts[dataset].append(float(np.mean(values)))
                lo=hi=0.
                for model in primary_models:
                    for repeat in range(3):
                        a=lookup.get((task['id'],language,model,repeat)); b=lookup.get((task['id'],'en',model,repeat))
                        aa=None if a is None or a['status']=='infrastructure_error' else float(a['status']=='pass')
                        bb=None if b is None or b['status']=='infrastructure_error' else float(b['status']=='pass')
                        lo+=((0. if aa is None else aa)-(1. if bb is None else bb))/6
                        hi+=((1. if aa is None else aa)-(0. if bb is None else bb))/6
                lower.append(lo); upper.append(hi)
            bounds.append([float(np.mean(lower)),float(np.mean(upper))] if subset else None)
        result=stratified_test(parts)
        result.update(language=language,baseline='en',models=primary_models,
                      estimand='Mean task-level Pass@1 difference among complete pairs, equally weighted across two models and three benchmarks',
                      planned_task_counts=dict(zip(DATASETS,planned)),
                      whole_planned_sample_bounds=[float(np.mean([b[0] for b in bounds])),float(np.mean([b[1] for b in bounds]))] if all(b is not None for b in bounds) else None)
        primary.append(result)
    for result,adjusted in zip(primary,holm([r['p_value'] for r in primary])):
        result['p_holm']=adjusted; result['family_size']=8
        result['reject_holm_05']=adjusted<.05 if adjusted is not None else None
    comparisons=[]; difficulty=[]; interactions=[]; rq2=[]
    def paired_report(values,scope):
        return {'paired_tasks':len(values),'mean_difference':float(np.mean(values)) if values else None,
                'ci95':bootstrap([values],stable_seed(c['seed'],scope),c['bootstrap_samples'])}
    for language in [l for l in languages if l!='en']:
        for model in models:
            for dataset in datasets:
                subset=[t for t in tasks if t['dataset']==dataset]
                for kind in ('pass_at_1','pass_at_3'):
                    values=[v for t in subset if (v:=difference(t,language,model,kind)) is not None]
                    scope=[language,model,dataset,kind]
                    comparisons.append({'language':language,'baseline':'en','model':model,'dataset':dataset,'measure':kind,
                                        **paired_report(values,scope),'exploratory':True})
                    if dataset=='livecodebench':
                        for level in ('easy','medium','hard'):
                            v=[x for t in subset if t['difficulty'].lower()==level and (x:=difference(t,language,model,kind)) is not None]
                            difficulty.append({'language':language,'baseline':'en','model':model,'dataset':dataset,'difficulty':level,
                                               'measure':kind,**paired_report(v,scope+[level]),'exploratory':True})
                for metric in METRICS:
                    for passing_only in (False,True):
                        differences=[]; left_values=[]; right_values=[]; a_count=b_count=0
                        analyzable_a=analyzable_b=fail_a=fail_b=generated_a=generated_b=known_a=known_b=0
                        for task in subset:
                            a=[r for r in rows(task,language,model) if r is not None]
                            b=[r for r in rows(task,'en',model) if r is not None]
                            generated_a+=len(a); generated_b+=len(b)
                            known_a+=sum(r['status']!='infrastructure_error' for r in a)
                            known_b+=sum(r['status']!='infrastructure_error' for r in b)
                            analyzable_a+=sum(r.get(metric) is not None for r in a)
                            analyzable_b+=sum(r.get(metric) is not None for r in b)
                            fail_a+=sum(r['status'] not in ('pass','infrastructure_error') for r in a)
                            fail_b+=sum(r['status'] not in ('pass','infrastructure_error') for r in b)
                            av=[r[metric] for r in a if r.get(metric) is not None and (not passing_only or r['status']=='pass')]
                            bv=[r[metric] for r in b if r.get(metric) is not None and (not passing_only or r['status']=='pass')]
                            if not av or not bv: continue
                            a_count+=len(av); b_count+=len(bv); left_values.extend(av); right_values.extend(bv)
                            differences.append(float(np.mean(av)-np.mean(bv)))
                        def distribution(values):
                            return {'n':len(values),'quantiles':list(map(float,np.quantile(values,[0,.25,.5,.75,1])))} if values else {'n':0,'quantiles':None}
                        rq2.append({'language':language,'baseline':'en','model':model,'dataset':dataset,'metric':metric,
                                    'subset':'test_passing' if passing_only else 'analyzable',
                                    **paired_report(differences,[language,model,dataset,metric,passing_only]),
                                    'contributing_language_outputs':a_count,'contributing_english_outputs':b_count,
                                    'defined_language_outputs':analyzable_a,'defined_english_outputs':analyzable_b,
                                    'failed_language_outputs':fail_a,'failed_english_outputs':fail_b,
                                    'generated_language_outputs':generated_a,'generated_english_outputs':generated_b,
                                    'language_analyzability_rate':analyzable_a/generated_a if generated_a else None,
                                    'english_analyzability_rate':analyzable_b/generated_b if generated_b else None,
                                    'language_functional_failure_rate':fail_a/known_a if known_a else None,
                                    'english_functional_failure_rate':fail_b/known_b if known_b else None,
                                    'language_distribution':distribution(left_values),'english_distribution':distribution(right_values),
                                    'exploratory':True})

        for first,second in itertools.combinations(models,2):
            parts=[]; per_dataset={}
            for dataset in datasets:
                values=[]
                for task in [t for t in tasks if t['dataset']==dataset]:
                    a,b=difference(task,language,first),difference(task,language,second)
                    if a is not None and b is not None: values.append(a-b)
                per_dataset[dataset]=values
                if values: parts.append(values)
                interactions.append({'type':'language_by_model_within_benchmark','language':language,'dataset':dataset,
                                     'model_a':first,'model_b':second,**paired_report(values,[language,first,second,dataset]),'exploratory':True})
            if len(parts)==len(datasets):
                interactions.append({'type':'language_by_model','language':language,'model_a':first,'model_b':second,
                                     'paired_tasks_by_benchmark':{ds:len(v) for ds,v in per_dataset.items()},
                                     'mean_difference':float(np.mean([np.mean(p) for p in parts])),
                                     'ci95':bootstrap(parts,stable_seed(c['seed'],[language,first,second]),c['bootstrap_samples']),'exploratory':True})

        for model in models:
            for first,second in itertools.combinations(datasets,2):
                a=[v for task in tasks if task['dataset']==first and (v:=difference(task,language,model)) is not None]
                b=[v for task in tasks if task['dataset']==second and (v:=difference(task,language,model)) is not None]
                interactions.append({'type':'language_by_benchmark','language':language,'model':model,'benchmark_a':first,'benchmark_b':second,
                                     'paired_tasks':[len(a),len(b)],'mean_difference':float(np.mean(a)-np.mean(b)) if a and b else None,
                                     'ci95':[2*x for x in bootstrap([a,[-v for v in b]],stable_seed(c['seed'],[language,model,first,second]),c['bootstrap_samples'])] if len(a)>1 and len(b)>1 else None,'exploratory':True})
    return {'primary_english_comparisons':primary,'model_benchmark_comparisons':comparisons,'difficulty_comparisons':difficulty,
            'interactions':interactions,'static_comparisons':rq2,
            'notes':['Pass@1 is the mean of three single-attempt test outcomes, averaged over complete task conditions.',
                     'Pass@3 is descriptive: at least one of the three candidates passes every supplied test.',
                     'Primary language-English tests require both models and all three benchmark strata; Holm always covers eight contrasts.',
                     'Complete-pair estimates and whole-planned-sample missing-slot bounds describe different populations.',
                     'Exploratory intervals resample tasks, retaining all linked language/model/repetition values.',
                     'RQ2 uses only defined values on common tasks; functional failures are reported separately. Lower metrics do not establish better overall code quality.']}

def stability(records):
    groups={}
    for r in records: groups.setdefault((r['model'],r['dataset'],r['language'],r['task_id']),[]).append(r)
    result=[]
    for (model,dataset,language,task_id),cell in sorted(groups.items()):
        valid=[r for r in cell if r['status']!='infrastructure_error']
        complete=len(valid)==3 and {r['repeat'] for r in valid}=={0,1,2}
        result.append({'model':model,'dataset':dataset,'language':language,'task_id':task_id,'responses':len(cell),
                       'complete_evaluated_repetitions':complete,'unique_code_hashes':len({r['code_hash'] for r in cell}),
                       'unique_response_hashes':len({r['response_hash'] for r in cell}),
                       'code_hash_agreement':len({r['code_hash'] for r in cell})==1 if len(cell)==3 else None,
                       'pass_fail_agreement':len({r['status']=='pass' for r in valid})==1 if complete else None,
                       'metric_variation':{m:{'defined_repetitions':len(values),'min':min(values) if values else None,
                                               'max':max(values) if values else None,'sd':float(np.std(values,ddof=1)) if len(values)>1 else None}
                                           for m in METRICS for values in [[r[m] for r in cell if r.get(m) is not None]]}})
    return result
