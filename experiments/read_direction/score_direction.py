"""Evaluate readout-component swaps with the seed as the paired sampling unit."""
from pathlib import Path
import argparse
import json
import random
import statistics
P=Path(__file__).resolve().parent
def mean(values):return statistics.mean(values)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['dev','test']);ph=ap.parse_args().phase
    seeds=list(range(27112000,27112016)) if ph=='dev' else list(range(96374120,96374184))
    rows=[json.loads(v) for v in (P/(ph+'_rows.jsonl')).read_text().splitlines()]
    lookup={(v['seed'],v['surface'],v['nuisance']):v for v in rows}
    assert len(rows)==len(lookup)==4*len(seeds) and all((s,t,n) in lookup for s in seeds for t in ['table','prose'] for n in (False,True))
    targets=[v for v in rows if not v['nuisance']];nu=[v for v in rows if v['nuisance']]
    ms=list(rows[0]['outputs']);scores={}
    def correct(v,m):return v['outputs'][m]['top']==v['donor_id']
    def preserved(v,m):return v['outputs'][m]['top']==v['recipient_id']
    for m in ms:
        scores[m]=dict(target=sum(correct(v,m) for v in targets),nuisance=sum(preserved(v,m) for v in nu),
                       target_recipient=sum(preserved(v,m) for v in targets),median_target_margin=statistics.median(v['outputs'][m]['margin'] for v in targets))
    rng=random.Random(60396);draws=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)]
    paired={}
    for a,b in [('direction','magnitude'),('direction','full'),('full','white_written'),('magnitude','reset')]:
        per={s:mean([int(correct(lookup[s,t,False],a))-int(correct(lookup[s,t,False],b)) for t in ['table','prose']]) for s in seeds}
        bs=sorted(mean(per[s] for s in d) for d in draws)
        paired[a+' minus '+b]=dict(difference=mean(per.values()),ci95=[bs[124],bs[4874]],gains=sum(correct(v,a) and not correct(v,b) for v in targets),losses=sum(correct(v,b) and not correct(v,a) for v in targets))
    retained=sum(correct(v,'full') and correct(v,'direction') for v in targets);full=scores['full']['target'];den=len(targets)
    credible=all(scores[m]['nuisance']==den for m in ['unwritten','full','direction','magnitude','reset'])
    unwritten=sum(preserved(v,'unwritten') for v in rows)
    reset=max(v['reset_max_logit_error'] for v in rows)
    effect=paired['direction minus magnitude']
    if ph=='dev':
        gate=full>=12 and retained>=full-1 and scores['direction']['target']>=scores['magnitude']['target']+8 and credible and reset<=1e-5
    else:
        gate=full>0 and retained/full>=.95 and effect['difference']>=.20 and effect['ci95'][0]>0 and credible and unwritten==len(rows) and reset<=1e-5
    share=[v['read_geometry']['angular_energy']/v['read_geometry']['total_energy'] for v in targets if v['read_geometry']['total_energy']>0]
    out=dict(phase=ph,seeds=len(seeds),per_condition=den,scored_paths=len(ms)*len(rows),scores=scores,paired=paired,
             full_successes_retained_by_direction=retained,nuisance_checks=credible,unwritten_recipient=unwritten,
             unwritten_n=len(rows),max_reset_error_all_vocab=reset,
             native_mean_angle_median_degrees=statistics.median(v['native_geometry']['mean_raw_angle_degrees'] for v in targets),
             optimized_mean_angle_median_degrees=statistics.median(v['read_geometry']['mean_raw_angle_degrees'] for v in targets),
             target_angular_energy_fraction_median=statistics.median(share),target_angular_energy_fraction_min=min(share),seconds=sum(v['seconds'] for v in rows),gate=gate)
    (P/(ph+'_summary.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
