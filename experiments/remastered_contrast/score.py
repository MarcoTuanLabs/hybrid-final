import json,re,string,sys
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent
def read(s):return json.loads((p/s).read_text())
def norm(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
source=Path(sys.argv[1]) if len(sys.argv)>1 else p/'rows.jsonl'
reg=read('contract.json');keys=read('labels.json');rows=list(map(json.loads,source.read_text().splitlines()))
assert len(keys)==96 and len(rows)==len(keys)*len(reg['modes'])==len({(x['mode'],x['id']) for x in rows})
stats={}
for mode in reg['modes']:
    group=[x for x in rows if x['mode']==mode];score=Counter()
    for x in group:
        k=keys[x['id']];score[k['role']+'_'+k['structure']]+=int(bool(x['answer']) and norm(x['answer']) in set(map(norm,k['aliases'])))
    stats[mode]=dict(cells=score,correct=sum(score.values()),positions=sum(x['forward_positions'] for x in group),reader_s=sum(x['elapsed_s'] for x in group),routes=Counter(x['route'] for x in group))
c=stats['native']['cells'];controls=[v for k,v in stats.items() if k!='native']
passed=all(c['edited_multi']>=s['cells']['edited_multi']+3 and c['edited_single']>=s['cells']['edited_single'] and sum(c[k] for k in ('unaffected_multi','unaffected_single'))>=sum(s['cells'][k] for k in ('unaffected_multi','unaffected_single')) for s in controls)
out=dict(n=96,per_cell=24,stats=stats,promote=passed,decision='test_larger' if passed else 'discard_native_contrast',retrieval_cost=read('ranking_cost.json'))
(p/('summary.json' if source==p/'rows.jsonl' else 'fresh_summary.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
