import ast
import gzip
import json
import lzma
import time
from .common import ROOT, digest
from .sandbox import run
VERSION=5
METRICS=('cc_max','halstead_volume','sloc','nesting_depth')

def evaluate(task, code, c):
    started=time.monotonic()
    if not code.strip(): return {'status':'no_text','detail':'No extracted code','seconds':0.,'tests_run':0}
    try: ast.parse(code)
    except (SyntaxError,ValueError,RecursionError) as error:
        return {'status':'syntax_error','detail':str(error),'seconds':0.,'tests_run':0}
    cases=None
    if task.get('cases_file'):
        path=ROOT/'data'/task['cases_file']
        if not path.exists() or digest(path.read_bytes())!=task['cases_sha256']:
            return {'status':'infrastructure_error','detail':'Benchmark test checksum mismatch','seconds':0.,'tests_run':0}
        with (lzma.open if path.suffix=='.xz' else gzip.open)(path,'rt',encoding='utf-8') as stream: cases=json.load(stream)
    limits={k:c[k] for k in ('timeout_seconds','task_timeout_seconds','memory_mb','output_limit_mb')}
    result=run({'task':{k:v for k,v in task.items() if k!='reference'},'code':code,'cases':cases,'limits':limits},c)
    result['seconds']=time.monotonic()-started
    return result

CONTROLS=(ast.If,ast.For,ast.AsyncFor,ast.While,ast.Try,ast.TryStar,ast.With,ast.AsyncWith,ast.Match)
FUNCTIONS=(ast.FunctionDef,ast.AsyncFunctionDef)

def function_nesting(function):
    maximum=0
    def visit(node, depth, parent=None):
        nonlocal maximum
        if isinstance(node, FUNCTIONS+(ast.ClassDef,ast.Lambda)): return
        level=depth
        if isinstance(node,CONTROLS):

            same_branch=isinstance(node,ast.If) and isinstance(parent,ast.If) and node in parent.orelse and node.col_offset==parent.col_offset
            level=depth if same_branch else depth+1
            maximum=max(maximum,level)
        for child in ast.iter_child_nodes(node): visit(child,level,node)
    for node in function.body: visit(node,0)
    return maximum

def metrics(code):
    result={'syntax_valid':False, **{name:None for name in METRICS}, 'metric_missing_reasons':{}}
    if not code.strip():
        result['metric_missing_reasons']={m:'empty_code' for m in METRICS}; return result
    try: tree=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError):
        result['metric_missing_reasons']={m:'syntax_error' for m in METRICS}; return result
    result['syntax_valid']=True
    functions=[n for n in ast.walk(tree) if isinstance(n,FUNCTIONS)]
    if functions: result['nesting_depth']=max(function_nesting(f) for f in functions)
    else: result['metric_missing_reasons'].update(cc_max='absent_function',nesting_depth='absent_function')
    from radon.complexity import cc_visit
    from radon.metrics import h_visit
    from radon.raw import analyze
    try:

        from radon.visitors import Function,Class
        measured=[]
        def collect(block):
            if isinstance(block,Function):
                measured.append(block.complexity)
                for child in block.closures: collect(child)
            elif isinstance(block,Class):
                for method in block.methods: collect(method)
                for child in block.inner_classes: collect(child)
        for block in cc_visit(code): collect(block)
        if measured: result['cc_max']=max(measured)
        elif functions: result['metric_missing_reasons']['cc_max']='analyzer_error'
    except Exception as error: result['metric_missing_reasons']['cc_max']='analyzer_error: '+type(error).__name__
    for metric, extract in [('halstead_volume',lambda:h_visit(code).total.volume),('sloc',lambda:analyze(code).sloc)]:
        try: result[metric]=extract()
        except Exception as error: result['metric_missing_reasons'][metric]='analyzer_error: '+type(error).__name__
    return result
