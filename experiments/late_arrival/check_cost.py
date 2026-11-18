"""Recheck lifecycle cost, answer changes, and intervals grouped by shared evidence sessions."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent/'retention_splice'
rows=list(map(json.loads,(p/'rows.jsonl').read_text().splitlines()))
groups={}
for row in rows:
    groups.setdefault(row['mode'],{})[row['id']]=row
ids=set(groups['full'])
clusters=json.loads((p/'split_audit.json').read_text())['groups_by_shared_evidence_session']
clusters=[[i for i in group if i in ids] for group in clusters]
clusters=[group for group in clusters if group]
assert len(ids)==34 and len(clusters)==33 and all(set(method)==ids for method in groups.values())
rng=np.random.default_rng(8330)
draws=rng.integers(len(clusters),size=(10000,len(clusters)))
out={'evidence_groups':len(clusters),'questions':len(ids),'comparisons':{}}
for other in ('full','stale','bm25'):
    entry={}
    for name,fn in [('lifecycle_wall',lambda x:x['ingest_s']+x['online_s']),
                    ('lifecycle_ops',lambda x:2*(x['ingest_macs']+x['online_macs'])+x['estimated_repair_scalar_ops']),
                    ('update_wall',lambda x:x['online_s'])]:
        x,y=[np.array([sum(fn(groups[m][i]) for i in cluster) for cluster in clusters]) for m in ('gates',other)]
        ratios=x[draws].sum(axis=1)/y[draws].sum(axis=1)
        entry[name]=dict(ratio=x.sum()/y.sum(),cluster_95=np.quantile(ratios,[.025,.975]).tolist())
    entry['different_sequences']=sum(groups['gates'][i]['seq']!=groups[other][i]['seq'] for i in ids)
    out['comparisons'][other]=entry
print(json.dumps(out,indent=2))
