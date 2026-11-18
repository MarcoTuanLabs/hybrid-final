"""Score complete cold and warm shared-document updates and their downstream answers."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import statistics
import string
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def norm(t):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',t.lower().translate(str.maketrans('','',string.punctuation))).split())
def ratio_ci(vals,base,draws):
    ratios=sorted(sum(vals[j] for j in draw)/sum(base[j] for j in draw) for draw in draws)
    return [ratios[124],ratios[4874]]
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('public',type=Path);a=ap.parse_args()
    labels={(v['source'],v['id']):[set(map(norm,v[k])) for k in ['updated','stable']] for v in read(a.public/'experiments/read_projection/data/dev_labels.json')}
    rows=list(map(json.loads,(HERE/'rows.jsonl').read_text().splitlines()));assert len(rows)==len({(r['source'],r['id']) for r in rows})==12
    ws=['patch_full','patch_prefix','dense','zero','bit1','bit2','random','svd','read']
    by=defaultdict(list)
    for r in rows:
        ref=next(v for v in r['arms'] if v['mode']=='dense');key=r['source'],r['id']
        assert sorted(v['mode'] for v in r['arms'])==sorted(ws)
        for v in r['arms']:
            wins=[bool(t.strip()) and norm(t.strip().splitlines()[0]) in labs for t,labs in zip(v['texts'],labels[key],strict=True)]
            by[v['mode']].append(v|dict(src=r['source'],hits=wins,matches=[x==y for x,y in zip(v['tokens'],ref['tokens'],strict=True)]))
    stats={}
    for mode,rr in by.items():
        results={k:statistics.mean(v[k] for v in rr) for k in ['wire_bytes','ready_cold_s','ready_warm_s','cold_s','warm_s','donor_s','sender_s','net_s','receiver_s','answers_s']}
        stats[mode]=dict(mean=results,sequence_matches=sum(sum(v['matches']) for v in rr),facts_correct=sum(sum(v['hits']) for v in rr),by_source={src:[sum(v['hits'][k] for v in rr if v['src']==src) for k in range(2)] for src in ['babi','mquake']})
    valid=[m for m in ws if m!='read' and stats[m]['sequence_matches']==24]
    best=min(valid,key=lambda m:stats[m]['mean']['warm_s']);r=stats['read'];time_ratio=r['mean']['warm_s']/stats[best]['mean']['warm_s']
    quality=(r['sequence_matches']==24 and r['by_source']==stats['dense']['by_source'])
    gain=r['sequence_matches']-stats['zero']['sequence_matches']
    rng=random.Random(91873);draws=[rng.choices(range(12),k=12) for _ in range(5000)]
    times=lambda m,k:[v[k] for v in by[m]]
    install=read(HERE/'install.json');fit=read(HERE/'fit.json')
    benefit=stats[best]['mean']['warm_s']-r['mean']['warm_s']
    out=dict(documents=12,answers=24,stats=stats,complete_quality=quality,gain_vs_zero=gain,fastest_exact_control=best,warm_time_ratio=time_ratio,
             paired_document_bootstrap95=ratio_ci(times('read','warm_s'),times(best,'warm_s'),draws),promotion=(quality and gain>=2 and time_ratio<=.80),
             basis_install=install['read'],combined_calibration_seconds=fit['seconds'],amortization=dict(mean_s_saved_vs_best=benefit,
             documents_to_recover_one_install=(install['read']['seconds']/benefit if benefit>0 else None),documents_to_recover_full_calibration_and_install=((install['read']['seconds']+fit['seconds'])/benefit if benefit>0 else None)))
    (HERE/'summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:out[k] for k in ['answers','complete_quality','gain_vs_zero','fastest_exact_control','warm_time_ratio','promotion']}))
if __name__=='__main__':main()
