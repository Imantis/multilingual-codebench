import ast
import textwrap
from project.common import ROOT, digest


def imports(code):
    tree=ast.parse(code);mods=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):mods.update(a.name.split('.')[0] for a in n.names)
        if isinstance(n,ast.ImportFrom) and n.module:mods.add(n.module.split('.')[0])
    return mods

def humaneval_task(row):

    source=row['prompt']; tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==row['entry_point'])
    description=next(n for n in function.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant)
                     and isinstance(n.value.value,str))
    lines=source.splitlines(keepends=True)
    prefix=''.join(lines[:description.lineno-1])
    prefix+=lines[description.lineno-1].encode('utf-8')[:description.col_offset].decode('utf-8')
    prompt=textwrap.dedent(description.value.value).strip()+'\n\nRequired Python interface:\n```python\n'+prefix.rstrip()+'\n```'
    return {'id':row['task_id'],'dataset':'humaneval','prompt':prompt,'entry_point':row['entry_point'],
            'tests':row['test'],'source_hash':digest(row)}
