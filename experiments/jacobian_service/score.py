"""Recalculate all same-write quality, paired cost and preservation results."""
import json
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
def main():
    rows=list(map(json.loads,(HERE/'rows.jsonl').read_text().splitlines()))
    assert len(rows)==len({tuple(r['key']) for r in rows})==64
    assert all(len({a['donor_digest'] for a in r['arms'].values()})==1 for r in rows)
    ref='cut16';orig='full16';mode_names=sorted(rows[0]['arms'])
    groups={'target':[r for r in rows if r['key'][3] and not r['key'][2]],'nuisance':[r for r in rows if r['key'][2] and r['key'][3]],'unwritten':[r for r in rows if not r['key'][3]]}
    assert [len(v) for v in groups.values()]==[16,16,32]
    rng=random.Random(24177);seeds=sorted({r['key'][0] for r in rows});samples=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)];stats={};gates={}
    def success(r,m,target):return r['arms'][m]['top']==r['donor' if target else 'recipient']
    def sums(mode):return {s:sum(r['arms'][mode]['seconds'] for r in rows if r['key'][0]==s) for s in seeds}
    for m in mode_names:
        t=groups['target'];base=sums(ref);v=sums(m)
        ratios=sorted(sum(v[s] for s in samp)/sum(base[s] for s in samp) for samp in samples)
        stats[m]=s0=dict(target=sum(success(r,m,True) for r in t),nuisance=sum(success(r,m,False) for r in groups['nuisance']),unwritten=sum(success(r,m,False) for r in groups['unwritten']),mean_s=statistics.mean(r['arms'][m]['seconds'] for r in rows),mean_prefill_write_s=statistics.mean(r['arms'][m]['prefill_write_s'] for r in rows),mean_suffix_s=statistics.mean(r['arms'][m]['query_and_output_s'] for r in rows),ratio_vs_matched=sum(v.values())/sum(base.values()),ratio_ci95=[ratios[124],ratios[4874]],retained=sum(success(r,m,True) and success(r,ref,True) for r in t),new_gains=sum(success(r,m,True) and not success(r,ref,True) for r in t))
        if m not in [ref,orig,'cut_native']:
            gates[m]=s0['retained']>=.95*sum(success(r,ref,True) for r in t) and s0['nuisance']==16 and s0['unwritten']==32 and s0['ratio_vs_matched']<=.8
    parity=all(r['arms'][ref]['top']==r['arms'][orig]['top'] and r['arms'][ref]['query_digest']==r['arms'][orig]['query_digest'] for r in rows)
    base=sums(orig);v=sums(ref);ci=sorted(sum(v[s] for s in samp)/sum(base[s] for s in samp) for samp in samples)
    cut_ratio=sum(v.values())/sum(base.values());gates['truncation']=parity and cut_ratio<=.8
    out=dict(services=len(rows)*len(mode_names),seeds=len(seeds),per_condition=16,scores=stats,identical_write_count=len(rows),cut_and_full_parity=parity,cut_vs_full_ratio=cut_ratio,cut_vs_full_ci95=[ci[124],ci[4874]],gates=gates)
    (HERE/'summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
