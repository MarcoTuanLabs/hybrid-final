"""Score document pairs and check whether selecting proposed answers could meet the gate."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
S=P.parent/'asof_revision'
sys.path.insert(0,str(S))
from score import load_helpers,mark


def read(path):
    return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('official',type=Path);ap.add_argument('tag');a=ap.parse_args()
    rows=list(map(json.loads,(P/(a.tag+'_rows.jsonl')).read_text().splitlines()))
    keys=read(S/'labels.json');items={r['id']:r for r in read(S/'ranked.json')}
    assert len(rows)==len({r['id'] for r in rows})==len(keys) and {r['id'] for r in rows}==keys.keys()
    helpers=load_helpers(a.official/'official_utils.py',a.official/'evaluate.py');party=read(a.official/'party_mapping.json')
    stats=defaultdict(Counter);both_reachable=0;any_reachable=0
    for row in rows:
        uid=row['id'];item=items[uid]
        def grade(i,answer):
            return mark(answer,str(keys[uid][i]),item,helpers,party)
        opts=[[grade(i,answer)[0] for i,answer in enumerate(pair)] for pair in row['options']]
        both_reachable+=any(all(pair) for pair in opts)
        any_reachable+=sum(any(pair[i] for pair in opts) for i in (0,1))
        methods={**row['proposals'],**{'score_'+k:v for k,v in row['winners'].items()}}
        for name,pair in methods.items():
            marks=[grade(i,answer) for i,answer in enumerate(pair)]
            c=stats[name];c['old']+=marks[0][0];c['new']+=marks[1][0];c['both']+=int(all(v[0] for v in marks))
            c['plain_exact']+=sum(helpers['compute_exact'](str(keys[uid][i]),answer) for i,answer in enumerate(pair))
            c['f1_sum']+=sum(v[1] for v in marks)
    for c in stats.values():
        c['correct']=c['old']+c['new'];c['mean_f1']=c.pop('f1_sum')/(2*len(rows))
    prev=read(S/(a.tag+'_summary.json'))['stats'];target=stats['score_tangent']
    controls=[v for k,v in stats.items() if k!='score_tangent']+list(prev.values())
    gate=all(target['both']>=c['both']+3 and target['old']>=c['old'] and target['new']>=c['new'] for c in controls)
    out=dict(pages=len(rows),queries=2*len(rows),stats=stats,eligible_pool_ceiling_pairs=both_reachable,
             eligible_pool_ceiling_individual=any_reachable,gate_passed=gate)
    (P/(a.tag+'_summary.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
