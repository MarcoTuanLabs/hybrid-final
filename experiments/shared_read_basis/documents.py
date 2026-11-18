"""Build a fact prefix without any knowledge of future questions."""
import hashlib
import json
from pathlib import Path
import sys
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'update_service'))
from cache_service import advance

MARK='<|im_end|>\n<|im_start|>user\n'
def setup(repo,path):
    code=repo.resolve()/'experiments/read_projection';sys.path.insert(0,str(code))
    from run import Reader
    cfg=json.loads((code/'release.json').read_text());data=code/'data/dev_inputs.json'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==cfg['inputs']['dev']
    for n,sha in cfg['model_files'].items():
        with (path/n).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==sha
    inputs=json.loads(data.read_text());parts={'fit':[],'eval':[]}
    for src in ['babi','mquake']:
        cases=sorted([x for x in inputs if x['source']==src],key=lambda x:hashlib.sha256(('shared-doc-v1/'+src+'/'+str(x['id'])).encode()).digest())
        for part,group in [('fit',cases[:6]),('eval',cases[6:12])]:
            parts[part].extend(group)
    assert not ({(v['source'],v['id']) for v in parts['fit']}&{(v['source'],v['id']) for v in parts['eval']})
    trial=Reader(path);mx.set_cache_limit(128*1024**2)
    return trial,parts
def message(src,facts):
    rule=('Read this story in order. Answer questions with only the location.' if src=='babi' else
          'Use these updated facts when answering. Answer the question with just the shortest correct answer.')
    return rule+'\n\n'+'\n'.join(facts)
def template(trial,text,question):
    return trial.tok.apply_chat_template([dict(role='system',content=text),dict(role='user',content=question)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def encoding(trial,s):
    return trial.tok.encode(s,add_special_tokens=False)
def document(trial,text):
    fmt=template(trial,text,'Future request.');front,rest=fmt.split(MARK,1);assert rest.startswith('Future request.')
    return encoding(trial,front)
def future(trial,text,question,ids):
    fmt=template(trial,text,question);full=encoding(trial,fmt);assert full[:len(ids)]==ids
    return full[len(ids):]
def build_cache(trial,ids):
    return advance(trial.model,make_prompt_cache(trial.model),ids)
