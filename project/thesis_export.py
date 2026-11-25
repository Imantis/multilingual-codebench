import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from .common import digest

BENCHMARKS={'humaneval':'HumanEval','livecodebench':'LiveCodeBench','bigcodebench':'BigCodeBench'}
METRICS={'cc_max':'Cyclomatic complexity','halstead_volume':'Halstead volume',
         'sloc':'Source lines of code','nesting_depth':'Nesting depth'}
BLUE='#0072B2'
ORANGE='#D55E00'
WIDTH=15.0/2.54


def model_name(model):
    if 'gemma-4-' in model: return 'Gemma 4'
    if 'gpt-oss-20b' in model: return 'GPT-OSS 20B'
    return model


def tex(text):
    substitutions={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                   '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(substitutions.get(c,c) for c in str(text))


def number(value,decimals=2,scale=1):
    return '--' if value is None else f'{value*scale:.{decimals}f}'


def interval(value,scale=1):
    return '--' if value is None else '['+', '.join(number(x,2,scale) for x in value)+']'


def pvalue(value):
    return '--' if value is None else ('<0.001' if value<.001 else f'{value:.3f}')


def latex_table(path,caption,headers,rows,alignment,label):
    lines=[r'\begingroup',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',
           r'\begin{longtable}{@{}'+alignment+r'@{}}',
           r'\caption{'+tex(caption)+r'}\label{tab:exp-'+label+r'}\\',r'\toprule',
           ' & '.join(tex(x) for x in headers)+r' \\',r'\midrule',r'\endfirsthead',
           r'\toprule',' & '.join(tex(x) for x in headers)+r' \\',r'\midrule',r'\endhead',
           r'\bottomrule',r'\endfoot']
    lines+=[' & '.join(tex(x) for x in row)+r' \\' for row in rows]
    lines+=[r'\end{longtable}',r'\endgroup','']
    path.write_text('\n'.join(lines),encoding='utf-8')


def stability_summary(rows):
    grouped=defaultdict(list)
    for row in rows: grouped[(row['model'],row['dataset'],row['language'])].append(row)
    output=[]
    for (model,dataset,language),part in grouped.items():
        code=[r['code_hash_agreement'] for r in part if r['code_hash_agreement'] is not None]
        tests=[r['pass_fail_agreement'] for r in part if r['pass_fail_agreement'] is not None]
        output.append({'model':model,'dataset':dataset,'language':language,
                       'tasks_with_three_codes':len(code),'identical_code_rate':float(np.mean(code)) if code else None,
                       'tasks_with_three_test_outcomes':len(tests),'test_agreement_rate':float(np.mean(tests)) if tests else None})
    return output


def export(report,folder):

    folder=Path(folder); figures=folder/'figures'; tables=folder/'tables'
    figures.mkdir(parents=True,exist_ok=True); tables.mkdir(exist_ok=True)

    for name in ('functional-correctness','primary-comparisons','difficulty',*METRICS,'repeat-stability'):
        (tables/(name+'.tex')).unlink(missing_ok=True)
    meta=report['metadata']; stats=report['statistics']; languages=meta['languages']; models=meta['models']
    datasets=meta['datasets']; status=meta.get('collection_status',{})
    dev=meta.get('development_only',meta.get('mode')=='dev')
    banner='SYNTHETIC LAYOUT TEST - NOT EXPERIMENT DATA' if meta.get('mode')=='test' else ('DEVELOPMENT DATA - NOT MAIN STUDY' if dev else (
        'PARTIAL STUDY SCOPE' if not status.get('full_study_scope',False) else (
        'INCOMPLETE COLLECTION' if not status.get('scope_complete',False) else '')))
    prefix='Synthetic layout test. ' if meta.get('mode')=='test' else ('Development run. ' if dev else ('Partial or incomplete run. ' if banner else ''))
    manifest={'source_report':'../report.json','mode':meta.get('mode'),'development_only':dev,
              'analysis_data_sha256':digest(report),'exporter_sha256':digest(Path(__file__).read_bytes()),
              'collection_status':status,'figure_width_cm':15.0,
              'figures':[],'unavailable_figures':[],
              'note':'All numbers come from report.json. No API calls, new estimators or additional hypothesis tests.'}
    primary=stats['primary_english_comparisons']
    primary_flat=[]
    for row in primary:
        ci=row.get('ci95'); bounds=row.get('whole_planned_sample_bounds')
        primary_flat.append({'language':row['language'],'effect_percentage_points':row.get('effect_percentage_points'),
            'ci_low_percentage_points':ci[0]*100 if ci else None,'ci_high_percentage_points':ci[1]*100 if ci else None,
            'p_value':row['p_value'],'p_holm':row.get('p_holm'),'testable':row['testable'],'reason':row.get('reason'),
            'whole_sample_bound_low_pp':bounds[0]*100 if bounds else None,'whole_sample_bound_high_pp':bounds[1]*100 if bounds else None,
            **{s['dataset']+'_pairs':s['pairs'] for s in row['strata']}})
    repeat_rows=stability_summary(report['stability'])
    sources={'functional-correctness':report['summary'],'primary-comparisons':primary_flat,
             'difficulty':report['difficulty'],'language-contrasts':stats['model_benchmark_comparisons'],
             'difficulty-contrasts':stats['difficulty_comparisons'],'interactions':stats['interactions'],
             'static-comparisons':stats['static_comparisons'],'static-distributions':report.get('static_summary',[]),
             'repeat-stability':repeat_rows,'failures':report['failure_counts']}
    sources['deployment-diagnostics']=report.get('deployment_diagnostics',[])
    for name,rows in sources.items(): pd.DataFrame(rows).to_csv(tables/(name+'.csv'),index=False)
    variations=[{k:r[k] for k in ('task_id','model','dataset','language')}|{'metric':metric,**values}
                for r in report['stability'] for metric,values in r['metric_variation'].items()]
    pd.DataFrame(variations).to_csv(tables/'repeat-metric-variation.csv',index=False)

    latex_table(tables/'functional-correctness.tex',prefix+
        'Functional correctness on the selected benchmark tasks, by condition. Rates (%) use task conditions with all three evaluated outcomes. '
        'Gaps count missing API responses and infrastructure faults; returned invalid programs are evaluated failures.',
        ['Model','Benchmark','Lang.','Planned','Complete','Pass@1 (%)','Pass@3 (%)','Gaps'],
        [[model_name(r['model']),BENCHMARKS[r['dataset']],r['language'].upper(),r['planned_tasks'],r['complete_tasks'],
          number(r['pass_at_1'],1,100),number(r['pass_at_3'],1,100),r['missing_api_outputs']+r['infrastructure_outputs']]
         for r in report['summary']],'lllrrrrr','functional-correctness')
    latex_table(tables/'primary-comparisons.tex',prefix+
        'Primary target-language minus English differences in Pass@1 (percentage points), equally weighted over two models '
        'and three benchmarks. Intervals are pointwise 95%; p-values use the fixed eight-comparison Holm family. '
        'H/L/B counts are complete task pairs in HumanEval/LiveCodeBench/BigCodeBench. Whole-sample bounds concern all '
        'planned tasks rather than the complete-pair population. -- denotes an unavailable value.',
        ['Lang.','Difference','95% interval','Holm p','Pairs H/L/B','Whole-sample bounds'],
        [[r['language'].upper(),number(r.get('effect_percentage_points')),interval(r.get('ci95'),100),pvalue(r.get('p_holm')),
          '/'.join(str(s['pairs']) for s in r['strata']),interval(r.get('whole_planned_sample_bounds'),100)] for r in primary],
        'lrlrll','primary-comparisons')
    if report['difficulty']:
        latex_table(tables/'difficulty.tex',prefix+
            'LiveCodeBench source difficulty levels. Rates (%) use complete three-response task conditions. '
            'The full selection has 10 easy, 20 medium and 20 hard tasks; a development subset may omit levels.',
            ['Model','Lang.','Difficulty','Planned','Complete','Pass@1 (%)','Pass@3 (%)','Gaps'],
            [[model_name(r['model']),r['language'].upper(),r['difficulty'],r['planned_tasks'],r['complete_tasks'],
              number(r['pass_at_1'],1,100),number(r['pass_at_3'],1,100),r['missing_api_outputs']+r['infrastructure_outputs']]
             for r in report['difficulty']],'lllrrrrr','difficulty')
    for metric,title in METRICS.items():
        rows=[]
        for r in stats['static_comparisons']:
            if r['metric']!=metric: continue
            rows.append([model_name(r['model']),BENCHMARKS[r['dataset']],r['language'].upper(),
                         'All' if r['subset']=='analyzable' else 'Passing',r['paired_tasks'],number(r['mean_difference']),
                         interval(r['ci95']),str(r['contributing_language_outputs'])+'/'+str(r['contributing_english_outputs'])])
        if rows:
            latex_table(tables/(metric+'.tex'),prefix+title+
                ': paired differences of task means (target minus English). All uses defined metrics; Passing uses '
                'test-passing outputs with defined metrics. Both retain common tasks. N L/EN counts contributing outputs. '
                'Intervals are pointwise 95% task-bootstrap intervals. Lower values do not establish better overall code quality.',
                ['Model','Benchmark','Lang.','Subset','Pairs','Difference','95% interval','N L/EN'],rows,
                'llllrrp{2.4cm}r',metric)
    latex_table(tables/'repeat-stability.tex',prefix+
        'Agreement across the three repetitions. Identical code requires equal extracted-code hashes. '
        'Test agreement means three equal pass/fail outcomes; it can represent three failures. '
        'Each rate uses its own eligible task count.',
        ['Model','Benchmark','Lang.','N code','Identical (%)','N test','Test agreement (%)'],
        [[model_name(r['model']),BENCHMARKS[r['dataset']],r['language'].upper(),r['tasks_with_three_codes'],
          number(r['identical_code_rate'],1,100),r['tasks_with_three_test_outcomes'],number(r['test_agreement_rate'],1,100)]
         for r in repeat_rows],'lllrrrr','repeat-stability')

    style={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,'axes.labelsize':9,
           'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
           'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777777','axes.linewidth':.6,
           'grid.color':'#dddddd','grid.linewidth':.5,'savefig.facecolor':'white'}
    figure_tex=[r'% Generated from saved report.json. Requires graphicx; tables also require booktabs and longtable.',
                r'\providecommand{\experimentResultsDir}{experiment-results}']

    def save(fig,name,caption,source):
        if banner: fig.text(.5,.988,banner,ha='center',va='top',fontsize=8,color='#555555',weight='bold')
        fig.savefig(figures/(name+'.pdf'))
        fig.savefig(figures/(name+'.png'),dpi=300)
        plt.close(fig)
        manifest['figures'].append({'name':name,'pdf':'figures/'+name+'.pdf','png':'figures/'+name+'.png',
                                    'caption':prefix+caption,'source':source})
        figure_tex.extend([r'\begin{figure}[tbp]',r'  \centering',
            r'  \includegraphics[width=\linewidth]{\experimentResultsDir/figures/'+name+r'.pdf}',
            r'  \caption{'+tex(prefix+caption)+r'}',r'  \label{fig:exp-'+name+r'}',r'\end{figure}',''])

    def panels(height=2.05):
        fig,axes=plt.subplots(len(models),len(datasets),squeeze=False,figsize=(WIDTH,height*len(models)+.55))
        fig.subplots_adjust(left=.105,right=.98,bottom=.28 if len(models)==1 else .17,top=.78 if len(models)==1 else .87,
                            hspace=.7,wspace=.3)
        for i,model in enumerate(models):
            for j,dataset in enumerate(datasets):
                axes[i,j].set_title(BENCHMARKS[dataset]+'\n'+model_name(model))
        return fig,axes

    def rates(rows,fields,name,caption,source):
        fig,axes=panels(max(2.05,.21*len(languages)+.7))
        lookup={(r['model'],r['dataset'],r['language']):r for r in rows}
        for i,model in enumerate(models):
            for j,dataset in enumerate(datasets):
                ax=axes[i,j]; xs=np.arange(len(languages))
                for offset,(field,title,color,marker) in zip((-.20,.20),fields):
                    values=[lookup.get((model,dataset,l),{}).get(field) for l in languages]
                    ax.plot([np.nan if v is None else 100*v for v in values],xs+offset,linestyle='none',
                            marker=marker,markersize=4.5,color=color,label=title)
                ax.set_yticks(xs,[l.upper() for l in languages]); ax.set_ylim(len(languages)-.5,-.5)
                ax.set_xlim(-3,103); ax.set_xticks([0,50,100]); ax.grid(axis='x'); ax.set_axisbelow(True)
        fig.supxlabel('Functional correctness (%)' if name=='functional-correctness' else 'Agreement (%)',
                      y=.12 if len(models)==1 else .07,fontsize=9)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,0))
        save(fig,name,caption,source)

    def contrasts(rows,series,name,caption,source,scale=1):

        target_languages=[l for l in languages if l!='en']
        usable=[r for r in rows if r.get('mean_difference') is not None]
        if not usable or not target_languages:
            manifest['unavailable_figures'].append({'name':name,'reason':'No eligible language-English pairs in this scope.'})
            return
        values=[0.]+[scale*v for r in usable for v in ([r['mean_difference']]+(r['ci95'] or []))]
        lo,hi=min(values),max(values); pad=max((hi-lo)*.08,.1*scale)
        fig,axes=panels(max(2.0,.24*len(target_languages)+.7))
        for i,model in enumerate(models):
            for j,dataset in enumerate(datasets):
                ax=axes[i,j]; ax.axvline(0,color='#666666',linewidth=.8); ax.grid(axis='x'); ax.set_axisbelow(True)
                for k,language in enumerate(target_languages):
                    for offset,(key,kind,label,color,marker) in zip((-.20,.20),series):
                        row=next((r for r in rows if r['language']==language and r['model']==model and
                                  r['dataset']==dataset and r[key]==kind),None)
                        if row is None or row['mean_difference'] is None: continue
                        y=k+offset; ci=row.get('ci95')
                        if ci: ax.plot([ci[0]*scale,ci[1]*scale],[y,y],color=color,linewidth=1.1)
                        ax.plot(row['mean_difference']*scale,y,marker=marker,color=color,markersize=4,linestyle='none')
                ax.set_yticks(range(len(target_languages)),[l.upper() for l in target_languages])
                ax.set_ylim(len(target_languages)-.5,-.5); ax.set_xlim(lo-pad,hi+pad)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        xlabel='Difference (percentage points)' if scale==100 else METRICS.get(name,'Metric')+' difference'
        fig.supxlabel(xlabel,y=.12 if len(models)==1 else .07,fontsize=9)
        handles=[Line2D([],[],marker=s[4],color=s[3],label=s[2],linewidth=1) for s in series]
        fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,0))
        save(fig,name,caption,source)

    with plt.rc_context(style):
        rates(report['summary'],[('pass_at_1','Pass@1',BLUE,'o'),('pass_at_3','Pass@3',ORANGE,'^')],
              'functional-correctness','Functional correctness on the selected benchmark tasks, by language and model. Pass@1 is mean single-attempt '
              'success; Pass@3 is success in at least one of the same three candidates. Rates use complete task conditions. '
              'Contributing task counts and gaps appear in the functional-correctness table. Missing rates are not plotted.',
              'report.json:summary')
        if any(r.get('effect') is not None for r in primary):
            fig,ax=plt.subplots(figsize=(WIDTH,3.3)); fig.subplots_adjust(left=.13,right=.95,bottom=.16,top=.9)
            for k,r in enumerate(primary):
                if r.get('effect') is None:
                    ax.text(.99,k,'Unavailable',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8)
                    continue
                if r.get('ci95'): ax.plot([100*x for x in r['ci95']],[k,k],color=BLUE,linewidth=1.2)
                ax.plot(100*r['effect'],k,'o',color=BLUE,markersize=4.5)
            ax.axvline(0,color='#666666',linewidth=.8); ax.grid(axis='x'); ax.set_axisbelow(True)
            ax.set_yticks(range(len(primary)),[r['language'].upper() for r in primary]); ax.set_ylim(len(primary)-.5,-.5)
            ax.set_xlabel('Target-language minus English Pass@1 (percentage points)')
            save(fig,'primary-comparisons','Primary differences in Pass@1, equally weighted over both models and all three '
                 'benchmarks, on complete task pairs. Lines are pointwise 95% intervals; the table supplies the eight-family '
                 'Holm-adjusted p-values, paired-task counts and whole-planned-sample bounds. An unavailable interval is omitted.',
                 'report.json:statistics.primary_english_comparisons')
        else:
            manifest['unavailable_figures'].append({'name':'primary-comparisons',
                'reason':'Primary estimates require both models and at least two complete task pairs in every benchmark.'})
        contrasts(stats['model_benchmark_comparisons'],
                  [('measure','pass_at_1','Pass@1',BLUE,'o'),('measure','pass_at_3','Pass@3',ORANGE,'^')],
                  'language-contrasts','Exploratory paired language-English differences by model and benchmark. Lines '
                  'are pointwise 95% task-bootstrap intervals. Each comparison uses common complete task conditions; '
                  'paired counts are in language-contrasts.csv. An interval requires at least two pairs.',
                  'report.json:statistics.model_benchmark_comparisons',100)
        if report['difficulty']:
            fig,axes=plt.subplots(len(models),1,squeeze=False,figsize=(WIDTH,1.8*len(models)+.65))
            fig.subplots_adjust(left=.13,right=.79,bottom=.13,top=.86,hspace=.7)
            for i,model in enumerate(models):
                ax=axes[i,0]; values=np.full((3,len(languages)),np.nan)
                for y,level in enumerate(('easy','medium','hard')):
                    for x,language in enumerate(languages):
                        row=next((r for r in report['difficulty'] if r['model']==model and r['language']==language and r['difficulty']==level),None)
                        if row and row['pass_at_1'] is not None: values[y,x]=100*row['pass_at_1']
                        label=(('100%' if row['pass_at_1']==1 else number(row['pass_at_1'],1,100)+'%')+'\nn='+str(row['complete_tasks'])) if row and row['pass_at_1'] is not None else 'N/A'
                        ax.text(x,y,label,ha='center',va='center',fontsize=8,color='white' if np.isfinite(values[y,x]) and values[y,x]<50 else '#202020')
                cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#eeeeee')
                image=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=100,aspect='auto')
                ax.set_xticks(range(len(languages)),[l.upper() for l in languages]); ax.set_yticks(range(3),['Easy','Medium','Hard'])
                ax.set_title('LiveCodeBench / '+model_name(model)); ax.tick_params(length=0)
            color_ax=fig.add_axes([.85,.2,.018,.6]); fig.colorbar(image,cax=color_ax,label='Pass@1 (%)')
            save(fig,'difficulty','LiveCodeBench Pass@1 by the source difficulty labels, with complete task counts n. '
                 'Unavailable levels are N/A. Difficulty tables also report descriptive Pass@3 and paired language-English differences.',
                 'report.json:difficulty')
        for metric,title in METRICS.items():
            contrasts([r for r in stats['static_comparisons'] if r['metric']==metric],
                      [('subset','analyzable','All analyzable',BLUE,'o'),('subset','test_passing','Test-passing',ORANGE,'s')],
                      metric,title+': target-language minus English differences of task means on common analyzable '
                      'support and, separately, common test-passing support. Lines are pointwise 95% task-bootstrap intervals. '
                      'Pair and output counts are in the corresponding table. These are conditional code characteristics; '
                      'a lower value does not establish better overall code quality.',
                      'report.json:statistics.static_comparisons[metric='+metric+']')
        rates(repeat_rows,[('identical_code_rate','Identical code',BLUE,'o'),('test_agreement_rate','Same test outcome',ORANGE,'s')],
              'repeat-stability','Agreement across all three repetitions, by language, model and benchmark. Code agreement '
              'uses tasks with three extracted-code hashes; test agreement uses three evaluated pass/fail outcomes. '
              'The repeat-stability table gives both denominators. Three failures also count as test agreement.',
              'report.json:stability')


    current={r['name'] for r in manifest['figures']}
    for name in ['functional-correctness','primary-comparisons','language-contrasts','difficulty','repeat-stability',*METRICS]:
        if name not in current:
            for suffix in ('.pdf','.png'): (figures/(name+suffix)).unlink(missing_ok=True)
    (folder/'figures.tex').write_text('\n'.join(figure_tex)+'\n',encoding='utf-8')
    table_tex=[r'% Include this in an appendix if all detailed tables are needed.',r'\providecommand{\experimentResultsDir}{experiment-results}']
    for name in ['functional-correctness','primary-comparisons','difficulty',*METRICS,'repeat-stability']:
        if (tables/(name+'.tex')).exists(): table_tex.append(r'\input{\experimentResultsDir/tables/'+name+r'.tex}')
    (folder/'tables.tex').write_text('\n'.join(table_tex)+'\n',encoding='utf-8')
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description='Re-export thesis figures/tables from a saved report; no APIs or evaluation')
    p.add_argument('--workdir',type=Path,required=True)
    a=p.parse_args(); paths=list((a.workdir/'reports').rglob('report.json'))
    if not paths: raise ValueError('No saved report.json here; run project.analyse first.')
    path=max(paths,key=lambda x:x.stat().st_mtime)
    manifest=export(json.loads(path.read_text(encoding='utf-8')),path.parent/'thesis')
    print('Exported',len(manifest['figures']),'figures and tables:',(path.parent/'thesis').resolve())


if __name__=='__main__':
    from .common import entry
    entry(main)
