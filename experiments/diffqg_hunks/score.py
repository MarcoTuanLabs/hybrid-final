"""Score open QA and the separate answer-proposal upper bound with one question per topic."""
from pathlib import Path
from collections import Counter
import json
import statistics
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_updates'))
import importlib.util
_spec=importlib.util.spec_from_file_location('hoh_metric',HERE.parent/'hoh_updates'/'score.py')
_metric=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_metric)
evaluate=_metric.evaluate

def main():
    read=lambda n:json.loads((HERE/n).read_text())
    labels=read('labels.json');qs=read('questions.json');ranks=read('ranks.json')
    ids=[q['id'] for q in qs]
    assert len(set(ids))==len({d['title'] for d in labels.values()})==len(ids)==32 and set(labels)==set(ids)
    span={r['id']:r for r in map(json.loads,(HERE/'span_rows.jsonl').read_text().splitlines())}
    gen={r['id']:r for r in map(json.loads,(HERE/'open_rows.jsonl').read_text().splitlines())}
    assert set(span)==set(gen)==set(ids)
    methods={kind+'_'+name:{i:table[i]['answers'][name] for i in ids} for kind,table in [('span',span),('open',gen)] for name in table[ids[0]]['answers']}
    methods['span_top1']={i:span[i]['docs'][ranks[i]['main'][0]]['policy']['current'][1] for i in ids}
    metrics={}
    for name,pred in methods.items():
        em,f1=zip(*(evaluate(pred[i],labels[i]['current']) for i in ids),strict=True)
        metrics[name]=dict(em=sum(em),f1=statistics.mean(f1))
    controls=['span_top1','open_current_1','span_current','span_changed','span_whole','open_current_3','open_current_6','open_paired_3','open_current_hint','open_changed_hint','open_whole_hint']
    best=max(metrics[k]['em'] for k in controls);best_f1=max(metrics[k]['f1'] for k in controls)
    gates={m:metrics[m]['em']>=best+3 and metrics[m]['f1']>=best_f1 for m in ['span_rollback','span_symmetric','open_rollback_hint','open_symmetric_hint']}
    recall={mode:[sum(labels[i]['source_key'] in ranks[i][mode][:k] for i in ids) for k in [1,3,6]] for mode in ['main','all_years','dense','lexical']}
    oracle=sum(any(evaluate(s,labels[i]['current'])[0] for d in span[i]['docs'].values() for s in d['proposals']) for i in ids)
    paired={}
    for m,b in [('span_symmetric','span_current'),('open_symmetric_hint','open_current_hint'),('span_rollback','span_whole'),('open_rollback_hint','open_whole_hint')]:
        counts=Counter((evaluate(methods[b][i],labels[i]['current'])[0],evaluate(methods[m][i],labels[i]['current'])[0]) for i in ids)
        paired[m]=dict(base=b,gains=counts[0,1],losses=counts[1,0])
    s=dict(n=len(ids),metrics=metrics,controls=controls,best_control_em=best,best_control_f1=best_f1,gates=gates,source_recall_top_1_3_6=recall,span_oracle=oracle,paired=paired,
           encoder_passages=sum(r['passages'] for r in span.values()),encoder_seconds=sum(r['seconds'] for r in span.values()),open_calls=sum(r['actual_calls'] for r in gen.values()),
           open_positions=sum(r['positions'] for r in gen.values()),open_seconds=sum(r['seconds'] for r in gen.values()))
    (HERE/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
