"""Evaluate all frozen rankings and open responses on the same 32 development pages."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'hoh_updates'
sys.path.insert(0,str(SOURCE))
from score import evaluate,tokens

def read(p):return json.loads(p.read_text())
def indexed(path,field):
    rs=list(map(json.loads,path.read_text().splitlines()))
    assert len(rs)==len({r[field] for r in rs}),path
    return {r[field]:r for r in rs}

def main():
    qs=read(SOURCE/'questions.json');labels=read(SOURCE/'labels.json');ids=[q['id'] for q in qs]
    pool=read(HERE/('pool.json' if (HERE/'pool.json').exists() else 'pool_ids.json'));cards=indexed(HERE/'cards.jsonl','key')
    ll=indexed(HERE/'likelihoods.jsonl','id');new=indexed(HERE/'answers.jsonl','id')
    spans=read(HERE/'spans.json');prior=read(SOURCE/'summary.json')
    assert set(ids)==set(pool)==set(ll)==set(new)==set(spans)==set(labels) and len(ids)==32
    assert all(len(set(r['key'] for r in v))==6 for v in pool.values())
    assert set(cards)=={r['key'] for v in pool.values() for r in v}
    methods={k:{i:new[i]['answers'][k] for i in ids} for k in new[ids[0]]['answers']}
    date=indexed(HERE/'date_answers.jsonl','id')
    assert set(date)==set(ids)
    for name in ('all_six','all_six_dated'):
        methods[name]={i:date[i]['answers'][name] for i in ids}
    methods['expanded_extractive']={i:spans[i]['answers'][spans[i]['selected']]['answer'] for i in ids}
    def stats(pred):
        em,f1=zip(*(evaluate(pred[i],labels[i]['current']) for i in ids),strict=True)
        return dict(em=sum(em),f1=statistics.mean(f1),stale=sum(evaluate(pred[i],labels[i]['previous'])[0] and not e for i,e in zip(ids,em,strict=True)))
    methods['upr_top1_extractive']={i:spans[i]['answers'][new[i]['ranked']['upr'][0]]['answer'] for i in ids}
    scores={k:stats(v) for k,v in methods.items()};counts={}
    for mode in new[ids[0]]['ranked']:
        counts[mode]=[sum(labels[i]['source_key'] in new[i]['ranked'][mode][:n] for i in ids) for n in (1,3,6)]
    conventional=prior['practical_controls']
    base_em=max([prior['measures'][m]['em'] for m in conventional]+[scores[m]['em'] for m in ('expanded_open','upr_open','hypothetical_open','hypothetical_copy','expanded_extractive','upr_top1_extractive','all_six','all_six_dated')])
    base_f=max([prior['measures'][m]['f1'] for m in conventional]+[scores[m]['f1'] for m in ('expanded_open','upr_open','hypothetical_open','hypothetical_copy','expanded_extractive','upr_top1_extractive','all_six','all_six_dated')])
    gates={};paired={}
    for prefix,rest in [('paired',[]),('whole',['paired']),('recurrent',['paired','whole'])]:
        others=[scores[f'{arm}_{readout}'] for arm in rest for readout in ['open','copy']]
        e=max([base_em]+[o['em'] for o in others]);f=max([base_f]+[o['f1'] for o in others])
        for readout in ('open','copy'):
            m=prefix+'_'+readout;gates[m]=scores[m]['em']>=e+3 and scores[m]['f1']>=f
            b={'paired':'hypothetical','whole':'paired','recurrent':'paired'}[prefix]+'_'+readout
            pairs=Counter((evaluate(methods[b][i],labels[i]['current'])[0],evaluate(methods[m][i],labels[i]['current'])[0]) for i in ids)
            paired[m]=dict(base=b,gain=pairs[0,1],loss=pairs[1,0],changed_text=sum(tokens(methods[b][i])!=tokens(methods[m][i]) for i in ids))
    card_cover={}
    for mode in ['current','paired']:
        card_cover[mode]=dict(valid=sum(v[mode]['valid'] for v in cards.values()),source_cards=len(cards),
            any_exact=sum(any(evaluate(cards[r['key']][mode]['current'],labels[i]['current'])[0] for r in pool[i]) for i in ids),
            labeled_source_exact=sum(any(r['key']==labels[i]['source_key'] and evaluate(cards[r['key']][mode]['current'],labels[i]['current'])[0] for r in pool[i]) for i in ids))
    previous=read(SOURCE/('retrieved.json' if (SOURCE/'retrieved.json').exists() else 'ranking_ids.json'))
    novel=sum(r['key'] not in {x['key'] for a in ['direct','expanded'] for x in previous[i][a]} for i in ids for r in pool[i])
    timing={}
    for k,items in [('query',read(HERE/'rewrites.json').values()),('cards',cards.values()),('likelihood',ll.values()),('answers',new.values()),('date_control',date.values())]:
        timing[k]={field:sum(v[field] for v in items) for field in ('seconds','positions')}
    report=dict(n=len(ids),measures=scores,previous_open_best=prior['best_control_em'],best_control_em=base_em,best_control_f1=base_f,
                retrieval_top1_top3_top6=counts,additional_source_slots=novel,card_coverage=card_cover,gate=gates,paired=paired,
                total_matched_answer_calls=sum(r['calls'] for r in new.values()),model_usage=timing)
    (HERE/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
