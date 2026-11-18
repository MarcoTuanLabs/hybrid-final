"""Measure prospective answer-pool coverage separately from attainable top-one accuracy."""
import json
from pathlib import Path
import sys
P=Path(__file__).resolve().parent;S=P.parent/'asof_revision'
sys.path.insert(0,str(S))
from score import mark,load_helpers
def main():
    src=Path(sys.argv[1]);h=load_helpers(src/'official_utils.py',src/'evaluate.py')
    read=lambda f:json.loads(f.read_text())
    items={x['id']:x for x in read(S/'ranked.json')};keys=read(S/'labels.json');m=read(src/'party_mapping.json')
    rows=read(P/'proposals.json');assert len(rows)==len({x['id'] for x in rows})==len(keys)
    cases={}
    for row in rows:
        uid=row['id'];pool=[];top=[]
        for i,rec in enumerate(row['editions']):
            ok=lambda s:mark(s,str(keys[uid][i]),items[uid],h,m)[0]
            pool.append(any(ok(x) for x in rec['pool']));top.append(ok(rec['span']))
        cases[uid]=dict(covered=pool,top=top)
    output=dict(covered=sum(sum(c['covered']) for c in cases.values()),both_covered=sum(all(c['covered']) for c in cases.values()),
                top_correct=sum(sum(c['top']) for c in cases.values()),top_both=sum(all(c['top']) for c in cases.values()),pages=len(cases),per_page=cases)
    output['gate']=output['both_covered']>=17
    (P/'summary.json').write_text(json.dumps(output,indent=2)+'\n');print({k:v for k,v in output.items() if k!='per_page'})
if __name__=='__main__':main()
