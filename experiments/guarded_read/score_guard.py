"""Check all gold, stable and unwritten decisions without treating paired pages as independent endpoints."""
import argparse
import json
from pathlib import Path
from statistics import median
def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
    rows=[json.loads(z) for z in (a.out/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['page'] for r in rows})==24 and all(len(r['entries'])==4 for r in rows)
    examples=[e for r in rows for e in r['entries']]
    out={}
    for k in ('original','guard','matched'):
        changed=stable=lost=agree=0;mov=[]
        for e in examples:
            sh,w=e['outputs'][k];truth=e['gold'][e['edition']];ch=e['gold'][0]!=e['gold'][1]
            if ch:changed+=w['label']==truth
            else:stable+=w['label']==truth;lost+=e['previous']==truth and w['label']!=truth
            agree+=sh['label']==e['previous'];mov.extend(x[k] for x in e['drift'])
        out[k]=dict(changed=changed,stable=stable,stable_old_correct_lost=lost,sham_agrees_with_old=agree,median_unwritten_read_displacement=median(mov))
    o,g,h=out['original'],out['guard'],out['matched']
    assert o['changed']==50 and o['stable']==3 and o['stable_old_correct_lost']==3
    s=dict(pages=24,endpoints=96,changed=86,stable=10,methods=out,
           promote=g['changed']>=o['changed'] and g['changed']>h['changed'] and g['sham_agrees_with_old']>=o['sham_agrees_with_old']+2 and g['stable_old_correct_lost']==0)
    (a.out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s))
if __name__=='__main__':main()
