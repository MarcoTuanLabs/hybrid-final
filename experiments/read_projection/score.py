import json
import sys
from pathlib import Path
import numpy as np
import re
import gzip
import string
import argparse

p=Path(__file__).resolve().parent
def read(f):return json.loads(f.read_text())
def normalize(text):
    text=text.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())
parser=argparse.ArgumentParser()
parser.add_argument('phase',choices=['dev','test'])
parser.add_argument('--run-dir',type=Path,default=p.parents[1]/'evidence/read_projection')
parser.add_argument('--save',type=Path)
args=parser.parse_args()
phase=args.phase;cfg=read(p/'release.json')
items=read(p/'data'/(phase+'_labels.json'))
labels={(x['source'],x['id'],k):x[key] for x in items for k,key in [('changed','updated'),('stable','stable')]}
f=args.run_dir/(phase+'_rows.jsonl')
raw=f.read_text() if f.exists() else gzip.decompress(f.with_suffix('.jsonl.gz').read_bytes()).decode()
rows=list(map(json.loads,raw.splitlines()))
by={(r['source'],r['id'],r['role'],r['mode']):r for r in rows}
assert len(by)==len(rows)==len(labels)*len(cfg['modes'])
def good(r):
    pred=r['text'].strip().splitlines()
    return bool(pred) and normalize(pred[0]) in {normalize(x) for x in labels[r['source'],r['id'],r['role']]}
stats={};gate={}
for src in ('babi','mquake'):
    stats[src]={}
    for m in cfg['modes']:
        rr=[r for r in rows if r['source']==src and r['mode']==m]
        c={k:sum(good(r) for r in rr if r['role']==k) for k in ('changed','stable')}
        c.update(n=len(rr),first_match=sum(r['first_match'] for r in rr),sequence_matches=sum(r['same_tokens'] for r in rr),
                 charged_matrix_mib=rr[0]['bytes']/2**20,mean_logit_rmse=float(np.mean([r['rmse'] for r in rr])),
                 max_read_error=max((r['max_read_error'] for r in rr if r['max_read_error'] is not None),default=None))
        c['per_role_losses_to_full']={k:sum(not good(r) and good(by[src,r['id'],k,'dense']) for r in rr if r['role']==k) for k in ('changed','stable')}
        stats[src][m]=c
    s=stats[src];c=s['transported']
    rel=[c['mean_logit_rmse']/max(s[m]['mean_logit_rmse'],1e-15) for m in ('raw_queries','svd','row_2bit')]
    gate[src]=dict(max_error_ratio_to_controls=max(rel),logit_win=max(rel)<=.5,first_match=c['first_match']>=.95*c['n'],
      sequence_match=c['sequence_matches']>=.9*c['n'],loss_ok=max(c['per_role_losses_to_full'].values())<=1,
      payload=c['charged_matrix_mib']<=s['dense']['charged_matrix_mib']/16)
keep=all(all(v for k,v in g.items() if k!='max_error_ratio_to_controls') for g in gate.values())
gain=sum(stats[s]['transported']['changed']-stats[s]['no_matrix']['changed'] for s in stats)
keep=bool(keep and gain>=4)
result=dict(phase=phase,cases=len(items),real_arms=len(rows),stats=stats,gate=gate,pooled_gain_over_no_matrix=gain,
            decision=('confirm_public_projection' if phase=='dev' else 'retain_independent_public_projection') if keep else 'discard_public_projection_claim')
if args.save:args.save.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
