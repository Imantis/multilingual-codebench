import contextlib
import json
import sys
import unittest
from pathlib import Path

def same_json(left, right):
    if type(left) is not type(right): return False
    if isinstance(left, list): return len(left)==len(right) and all(same_json(a,b) for a,b in zip(left,right))
    if isinstance(left, dict): return left.keys()==right.keys() and all(same_json(left[k],right[k]) for k in left)
    return left==right

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); result=Path(sys.argv[2])
    task=request['task']; code=Path('solution.py').read_text(encoding='utf-8')
    namespace={'__name__':'solution', '__file__':str(Path('solution.py').resolve())}
    status='runtime_error'; detail=''; tests=0
    try:
        compiled=compile(code,'solution.py','exec')
        with open('candidate-output.txt','w') as captured, contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compiled,namespace)
            if task['dataset']=='humaneval':
                exec(task['tests'],namespace)
                namespace['check'](namespace[task['entry_point']]); status='pass'; tests=1
            elif task['dataset']=='bigcodebench':
                exec(task['tests'],namespace)
                suite=unittest.defaultTestLoader.loadTestsFromTestCase(namespace['TestCases'])
                if suite.countTestCases()==0: raise RuntimeError('No benchmark tests discovered')
                run=unittest.TextTestRunner(stream=captured,verbosity=0).run(suite); tests=run.testsRun
                status='pass' if run.wasSuccessful() else ('runtime_error' if run.errors else 'test_failure')
                detail='; '.join(str(e[1])[-1500:] for e in run.errors+run.failures)
                if any(any("No module named '"+m+"'" in e[1] for m in task.get('imports',[])) for e in run.errors):
                    status='infrastructure_error'
            elif task['dataset']=='livecodebench':
                case=request['case']; args=[json.loads(line) for line in case['input'].splitlines() if line.strip()]
                func=getattr(namespace['Solution'](),task['entry_point'])
                actual=json.loads(json.dumps(func(*args))); expected=json.loads(case['output'])
                status='pass' if same_json(actual,expected) else 'test_failure'; tests=1
                detail='' if status=='pass' else 'Function output differs'
            else: raise RuntimeError('Unknown benchmark adapter')
    except SyntaxError as error: status='syntax_error'; detail=str(error)
    except AssertionError as error: status='test_failure'; detail=str(error); tests=1
    except (ModuleNotFoundError,ImportError) as error:
        missing=(getattr(error,'name','') or '').split('.')[0]
        status='infrastructure_error' if missing in task.get('imports',[]) else 'runtime_error'; detail=str(error)
    except BaseException as error: status='runtime_error'; detail=type(error).__name__+': '+str(error)
    result.write_text(json.dumps({'status':status,'detail':detail[-3000:],'tests_run':tests}),encoding='utf-8')

if __name__=='__main__': main()
