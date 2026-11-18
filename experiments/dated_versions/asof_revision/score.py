"""Reproduce DynaQuest answer normalization, with paired old/new edition scores."""
import argparse
import ast
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import string
import unicodedata

P=Path(__file__).resolve().parent
REQUIRED={'normalize_time_string','normalize_answer','get_tokens','compute_exact','compute_f1'}


def load_helpers(utils,evaluator):
    code=ast.parse(utils.read_text())
    fns=[x for x in code.body if isinstance(x,ast.FunctionDef) and x.name in REQUIRED]
    assert {f.name for f in fns}==REQUIRED
    regex=next(x for x in code.body if isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='ARTICLES_REGEX')
    norm=next(x for x in ast.parse(evaluator.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='normalize_to_english')
    env=dict(re=re,datetime=datetime,string=string,collections=collections,unicodedata=unicodedata)
    exec(compile(ast.Module(body=[regex,norm,*fns],type_ignores=[]),'<published DynaQuest evaluation helpers>','exec'),env)
    return env


def mark(text,gold,item,helpers,parties):
    q=item['question']
    for marker in ['Provide the shortest answer','||\n\n## Step','||.','|||',' |']:
        text=text.split(marker)[0].strip()
    if 'party of' in q:
        gold=parties.get(item['page']+'||'+gold,gold)
    if any(t in q for t in [' year ',' years ',' date ',' dates ','term start',' period ']):
        text=helpers['normalize_time_string'](text);gold=helpers['normalize_time_string'](gold)
    a,b=map(helpers['normalize_to_english'],(gold,text))
    return helpers['compute_exact'](a,b),helpers['compute_f1'](a,b)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('official',type=Path);ap.add_argument('tag');args=ap.parse_args()
    h=load_helpers(args.official/'official_utils.py',args.official/'evaluate.py')
    parties=json.loads((args.official/'party_mapping.json').read_text())
    gold=json.loads((P/'labels.json').read_text());rank=json.loads((P/'ranked.json').read_text())
    objects={x['id']:x for x in rank}
    spans=json.loads((P/'span_answers.json').read_text());rows=list(map(json.loads,(P/(args.tag+'_rows.jsonl')).read_text().splitlines()))
    assert len(rows)==len(gold)==len(set(r['id'] for r in rows)) and objects.keys()==spans.keys()==gold.keys()
    stats=collections.defaultdict(collections.Counter);per_question={}
    for row in rows:
        uid=row['id'];item=objects[uid];methods=collections.defaultdict(list)
        for i in range(2):
            answers=dict(row['answers'][str(i)])
            answers['extractive_span']=spans[uid][i]
            top=item['evidence'][i][0]
            answers['field']=top['cells'][-1] if len(top['cells'])>=2 else top['text']
            for name,answer in answers.items():
                em,f1=mark(answer,str(gold[uid][i]),item,h,parties)
                c=stats[name];c['old' if i==0 else 'new']+=em;c['correct']+=em;c['f1_sum']+=f1
                c['plain_exact']+=h['compute_exact'](str(gold[uid][i]),answer)
                methods[name].append(em)
        for name,pair in methods.items():stats[name]['both']+=int(all(pair))
        per_question[uid]=methods
    for c in stats.values():c['mean_f1']=c.pop('f1_sum')/(2*len(rows))
    candidate=stats['attention_skip']
    passed=all(candidate['both']>=s['both']+3 and candidate['old']>=s['old'] and candidate['new']>=s['new'] for k,s in stats.items() if k!='attention_skip')
    out=dict(pages=len(rows),queries=2*len(rows),stats=stats,gate_passed=passed,per_page=per_question)
    (P/(args.tag+'_summary.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(stats=stats,passed=passed)))
if __name__=='__main__':main()
