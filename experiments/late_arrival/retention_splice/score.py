import json,math,re
from pathlib import Path
from collections import Counter
import numpy as np
p=Path(__file__).resolve().parent
read=lambda f:json.loads((p/f).read_text())
reg=read('contract.json');labels=read('labels.json');rows=[json.loads(l) for l in (p/'rows.jsonl').read_text().splitlines()]
assert set(labels)==set(read('../suffix_transport/split.json')['reserve'])
assert {(x['id'],x['mode']) for x in rows}=={(uid,m) for uid in labels for m in reg['modes']} and len(rows)==len(labels)*len(reg['modes'])
words=lambda s:re.findall(r'\w+',str(s).casefold())
def f1(a,b):
    x,y=words(a),words(b)
    return 2*sum((Counter(x)&Counter(y)).values())/max(1,len(x)+len(y))
groups={m:sorted([x for x in rows if x['mode']==m],key=lambda x:x['id']) for m in reg['modes']}
stats={}
for mode,group in groups.items():
    stats[mode]=dict(exact=sum(words(x['text'])==words(labels[x['id']]) for x in group),f1=np.mean([f1(x['text'],labels[x['id']]) for x in group]),
        exact_sequences=sum(x['seq']==ref['seq'] for x,ref in zip(group,groups['full'],strict=True)),mean_rmse=np.mean([x['rmse'] for x in group]),
        online_s=sum(x['online_s'] for x in group),lifecycle_s=sum(x['online_s']+x['ingest_s'] for x in group),
        lifecycle_positions=sum(x['ingest_positions']+x['online_positions'] for x in group),
        lifecycle_dense_equivalent_ops=sum(2*(x['ingest_macs']+x['online_macs'])+x['estimated_repair_scalar_ops'] for x in group),
        max_saved_retention_bytes=max(x['saved_retention_bytes'] for x in group))
g,f,s=map(stats.get,['gates','full','stale'])
ratio=g['lifecycle_dense_equivalent_ops']/f['lifecycle_dense_equivalent_ops']
paired={};rng=np.random.default_rng(491790);samples=rng.integers(len(labels),size=(5000,len(labels)))
for key,fn in [('lifecycle_time',lambda x:x['ingest_s']+x['online_s']),('lifecycle_operations',lambda x:2*(x['ingest_macs']+x['online_macs'])+x['estimated_repair_scalar_ops'])]:
    vals=np.array([fn(x) for x in groups['gates']]);ref=np.array([fn(x) for x in groups['full']])
    boot=(vals[samples].sum(axis=1)/ref[samples].sum(axis=1))
    paired[key]=dict(ratio=float(vals.sum()/ref.sum()),interval=np.quantile(boot,[.025,.975]).tolist())
passed=(ratio<=.8 and g['exact']>=f['exact'] and g['f1']>=f['f1'] and (g['exact_sequences']>=s['exact_sequences']+2 or g['mean_rmse']<=.8*s['mean_rmse']))
out=dict(stats=stats,paired=paired,passed=bool(passed),n=len(labels),grade='fixed lowercase alphanumeric exact and token F1; bounded knowledge-update histories, not official judge.')
(p/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
