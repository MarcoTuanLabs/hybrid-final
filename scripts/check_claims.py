"""Independently recompute paper claims from complete archived score rows.

No model inference. Requires NumPy. Does not drop failed target cases.
"""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def read_rows(name):
    with gzip.open(ROOT / 'evidence' / name, 'rt') as f:
        return [json.loads(line) for line in f]

def interval(v, draws, quantiles=(.025, .975)):
    return np.quantile(v[draws].mean(1), quantiles).tolist()

r9 = read_rows('9b_rows.jsonl.gz')
r27 = read_rows('27b_rows.jsonl.gz')
d9 = json.loads((ROOT / 'evidence/9b_results.json').read_text())
d27 = json.loads((ROOT / 'evidence/27b_results.json').read_text())
by9 = {tuple(r[k] for k in ('seed','surface','control','query','schedule','written')):r for r in r9}
by27 = {tuple(r[k] for k in ('seed','surface','control','reader','written')):r for r in r27}
seeds9, seeds27 = list(range(991000,991256)), list(range(995000,995128))
arms9 = [(q,'all') for q in ('native','raw_svd','transported_key','shared','whitened_key',
         'whitened_svd','group_normalized','group_output','complete_null')]
arms9 += [('group_normalized','final'),('whitened_key','final')]
methods27 = ('single_start16','batched16','raw_svd','shared_minimal','shared_instrumented','native')
assert len(by9) == len(r9) == 22528
assert set(by9) == {(s,f,c,q,a,w) for s in seeds9 for f in ('table','prose')
                    for c in (False,True) for q,a in arms9 for w in (False,True)}
assert len(by27) == len(r27) == 6144
assert set(by27) == {(s,f,c,m,w) for s in seeds27 for f in ('table','prose')
                     for c in (False,True) for m in methods27 for w in (False,True)}

def v9(q, schedule='all', control=False, written=True, field='donor_top'):
    return np.array([[by9[s,f,control,q,schedule,written][field] for f in ('table','prose')]
                     for s in seeds9], dtype=float)

def v27(m, control=False, written=True, field='donor_top'):
    return np.array([[by27[s,f,control,m,written][field] for f in ('table','prose')]
                     for s in seeds27], dtype=float)

draw9 = np.random.default_rng(20261107201).integers(0,256,(20000,256))
draw27 = np.random.default_rng(20261108021).integers(0,128,(20000,128))
effect9 = (v9('group_normalized')-v9('whitened_key')).mean(1)
assert effect9.mean() == d9['primary']['mean']
assert interval(effect9,draw9) == d9['primary']['ci95']
assert [int(v9(q).sum()) for q in ('native','whitened_key','whitened_svd','group_output','group_normalized')] == [0,235,232,330,361]
assert v9('group_normalized','final').sum() == 82
timing9 = (v9('group_normalized')-v9('group_normalized','final')).mean(1)
assert np.allclose(np.array(interval(timing9,draw9))*100,[49.21875,59.9609375])
assert [int(x) for x in v9('group_normalized').sum(0)] == [167,194]
assert [int(x) for x in v9('whitened_key').sum(0)] == [124,111]
delta9 = v9('group_normalized')-v9('whitened_key')
assert ((delta9>0).sum(),(delta9<0).sum()) == (128,2)

for a in d9['aggregate']:
    for written,label in [(True,'written'),(False,'unwritten')]:
        for field,key in [('donor_top','donor_count'),('recipient_top','recipient_count')]:
            assert int(v9(a['query'],a['schedule'],a['control'],written,field).sum()) == a[label][key]

for a in d27['aggregate']:
    assert int(v27(a['reader']).sum()) == a['target_count']
    assert np.isclose(v27(a['reader'],field='total_component_seconds').mean(),a['mean_target_total_seconds'])
for c in d27['primary_contrasts']:
    effect = (v27(c['method'])-v27(c['baseline'])).mean(1)
    assert effect.mean() == c['gain']
    assert interval(effect,draw27,(.0125,.9875)) == c['ci97_5_seed_bootstrap']
    assert (effect>0).sum() == c['seed_wins'] and (effect<0).sum() == c['seed_losses']

for control,written in [(True,True),(False,False),(True,False)]:
    assert v9('group_normalized',control=control,written=written,field='recipient_top').sum() == 512
    assert v27('single_start16',control,written,'recipient_top').sum() == 256
    for m in methods27:
        assert v27(m,control,written,'recipient_top').sum() == 256

cost = {m:v27(m,field='total_component_seconds').mean(1) for m in methods27}
ratio = cost['single_start16'].mean()/cost['batched16'].mean()
ratio_ci = np.quantile(cost['single_start16'][draw27].mean(1)/cost['batched16'][draw27].mean(1),[.025,.975]).tolist()
assert np.isclose(ratio,d27['primary_cost']['candidate_batched16_ratio'])
assert np.allclose(ratio_ci,d27['primary_cost']['ci95_seed_bootstrap_ratio'])
assert int(((v27('batched16')==1)&(v27('single_start16')==0)).sum()) == 0
assert np.isclose(1-.05**(1/128),d27['control_preservation']['one_sided_95_exact_loss_upper'])
for s in seeds27:
    for f in ('table','prose'):
        for c in (False,True):
            for w in (False,True):
                a,b=(by27[s,f,c,m,w] for m in ('shared_minimal','shared_instrumented'))
                for key in ('logits_sha256','cache_sha256','queries_sha256','top_id'):
                    assert a[key] == b[key]

report = dict(status='PASS', complete_rows_checked=len(r9)+len(r27),
    independent_seeds={'9b':256,'27b':128},
    target_counts={'9b_baseline':235,'9b_normalized':361,'9b_n':512,
                   '27b_baseline':217,'27b_normalized':256,'27b_n':256},
    gain9_pp=float(effect9.mean()*100),ci9_pp=[x*100 for x in interval(effect9,draw9)],
    primary27=d27['primary_contrasts'],
    preservation={'9b_each_arm':'512/512','27b_each_arm':'256/256','arms':3},
    total_seconds={m:float(v.mean()) for m,v in cost.items()},
    reduction_percent=float((1-ratio)*100),ratio=float(ratio),ratio_ci95=ratio_ci,
    reference_lost_answers=0,shared_parity_branches=1024,
    note='CPU-only analysis of archived score records; no fresh model execution.')
(ROOT/'qa/claims_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
