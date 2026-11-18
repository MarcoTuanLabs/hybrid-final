from collections import Counter
from pathlib import Path
import json, re, sys
import numpy as np
p=Path(__file__).resolve().parent
def words(v):return re.findall(r'\w+',str(v).casefold())
def f1(a,b):
    x,y=words(a),words(b)
    return 2*sum((Counter(x)&Counter(y)).values())/(len(x)+len(y)) if x or y else 1.
model=sys.argv[1]; labels=json.loads((p/'labels.json').read_text());cfg=json.loads((p/'contract.json').read_text())
rows=[json.loads(line) for line in (p/(model+'_rows.jsonl')).read_text().splitlines()]
assert len(rows)==len(labels)*len(cfg['methods'])==len({(x['id'],x['mode']) for x in rows})
ref={x['id']:x for x in rows if x['mode']=='full'};stats={}
for mode in cfg['methods']:
    group=[x for x in rows if x['mode']==mode]
    stats[mode]=dict(exact=sum(words(x['text'])==words(labels[x['id']]) for x in group),
                     f1=float(np.mean([f1(x['text'],labels[x['id']]) for x in group])),
                     same_sequence=sum(x['seq']==ref[x['id']]['seq'] for x in group),
                     rmse=float(np.mean([x['centered_logit_rmse'] for x in group])),
                     online_forward=sum(x['update_answer_forward'] for x in group),
                     lifecycle_forward=sum(x['update_answer_forward']+x['old_forward'] for x in group),
                     online_s=sum(x['update_answer_s'] for x in group),
                     lifecycle_s=sum(x['update_answer_s']+x['old_prefix_s']+x['old_suffix_s'] for x in group),
                     extra_tape_bytes=sum(x['tape_bytes'] for x in group))
s=stats['transport']
eligible=s['same_sequence']>=max(stats[k]['same_sequence'] for k in ('stale','gates'))+2 and s['rmse']<=.8*min(stats[k]['rmse'] for k in ('stale','gates')) and s['f1']>=stats['full']['f1'] and s['online_s']<=.8*stats['full']['online_s']
out=dict(model=model,summary=stats,promote=bool(eligible),decision='test_larger' if eligible else 'discard_as_accuracy_efficiency_method',metric='Normalized exact and token F1 in bounded timestamped workload; not official GPT grader or unmodified full history.')
(p/(model+'_summary.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
