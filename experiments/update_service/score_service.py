"""Reconstruct the per-request service tradeoff from observed serialized handoffs."""
import argparse
import json
from pathlib import Path
import re
import statistics
import string

HERE=Path(__file__).resolve().parent
def normalize(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('public',type=Path);a=ap.parse_args()
    target=json.loads((a.public/'experiments/read_projection/data/dev_labels.json').read_text());labels={(v['source'],v['id'],role):{normalize(q) for q in v[k]} for v in target for k,role in [('updated','changed'),('stable','stable')]}
    rows=list(map(json.loads,(HERE/'rows.jsonl').read_text().splitlines()));modes=['patch_full','patch_prefix','dense','projection','quantized','answer_only']
    assert len(rows)==len({(r['source'],r['id'],r['role']) for r in rows})==16
    by={m:[] for m in modes}
    for r in rows:
        ref=next(v for v in r['arms'] if v['mode']=='dense')
        assert sorted(v['mode'] for v in r['arms'])==sorted(modes)
        for v in r['arms']:
            line=v['text'].strip().splitlines()
            v=v|dict(hit=bool(line) and normalize(line[0]) in labels[r['source'],r['id'],r['role']],exact=(v['tokens']==ref['tokens']),task=r['source'],role=r['role'])
            by[v['mode']].append(v)
    profiles={}
    for m,r in by.items():
        sums={k:sum(v[k] for v in r) for k in ['cold_s','warm_s','donor_s','sender_s','receiver_s','network_s','decode_s','wire_bytes']}
        mean={k:v/len(r) for k,v in sums.items()}
        links={str(mbps):{phase:sums[phase]-sums['network_s']+8*sums['wire_bytes']/(mbps*1e6) for phase in ['cold_s','warm_s']} for mbps in [100,1000,10000,100000]}
        profiles[m]=dict(mean=mean,total=sums,sequence_matches=sum(v['exact'] for v in r),by_task={t:{role:sum(v['hit'] for v in r if v['task']==t and v['role']==role) for role in ['changed','stable']} for t in ['babi','mquake']},link_scenarios_seconds=links)
    contender=profiles['projection'];fast=min(['patch_full','patch_prefix','dense'],key=lambda m:profiles[m]['total']['cold_s'])
    quality_gate=contender['sequence_matches']==16 and contender['by_task']==profiles['dense']['by_task']
    cost_ratio=contender['total']['cold_s']/profiles[fast]['total']['cold_s']
    def crossover(phase,baseline):
        # If the projection has more bytes but less compute, this is its minimum bandwidth.
        b=profiles[baseline]['total'];v=contender['total'];extra_bytes=v['wire_bytes']-b['wire_bytes']
        saved=(b[phase]-b['network_s'])-(v[phase]-v['network_s'])
        if extra_bytes<=0:return dict(extra_bytes=extra_bytes,compute_seconds_saved=saved)
        return dict(extra_bytes=extra_bytes,compute_seconds_saved=saved,minimum_mbps=(8*extra_bytes/saved/1e6 if saved>0 else None))
    result=dict(requests=len(rows),distinct_cases=len(rows)//2,profiles=profiles,fastest_required_baseline=fast,projection_cold_time_ratio=cost_ratio,quality_gate=quality_gate,
                promotion=(quality_gate and cost_ratio<=.80),crossover_vs_text={p:{m:crossover(p,m) for m in ['patch_prefix','patch_full']} for p in ['cold_s','warm_s']},
                old_cache_setup_s=sum(r['old_setup_s'] for r in rows),mean_old_tokens=statistics.mean(r['arms'][0]['old_tokens'] for r in rows),
                mean_new_tokens=statistics.mean(r['arms'][0]['new_tokens'] for r in rows),median_logical_snapshot_mib=statistics.median(r['logical_snapshot_bytes']/2**20 for r in rows))
    (HERE/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['requests','fastest_required_baseline','quality_gate','projection_cold_time_ratio','promotion']}))
if __name__=='__main__':main()
