"""Compare selected evidence and masking signals without looking at an answer label."""
import json
from pathlib import Path
import statistics
from answer import selections

HERE=Path(__file__).resolve().parent
def main():
    reads=[json.loads(z) for z in (HERE/'retrieval.jsonl').read_text().splitlines()]
    sup={v['id']:v['scores'] for v in map(json.loads,(HERE/'supports.jsonl').read_text().splitlines())}
    comparison={};signal={};ordered={}
    for r in reads:
        keys={k:set(v) for k,v in selections(r,sup[r['id']]).items()}
        anchors={str(v['session'])+'/'+str(v['offset']) for v in r['current']}
        ordered[r['id']]={k:s|anchors for k,s in keys.items()}
    for a,b in [('matrix_gain','whole_gain'),('matrix_gain','lm'),('whole_gain','lm'),('matrix_gain','span_quarters')]:
        sets=[(v[a],v[b]) for v in ordered.values()]
        comparison[a+' versus '+b]=dict(equal_prompt=sum(l==r for l,r in sets),median_overlap=statistics.median(len(l&r)/len(l|r) for l,r in sets),different_prompts=sum(l!=r for l,r in sets))
    for k in ['matrix_gain','whole_gain']:
        v=[z[k] for q in sup.values() for z in q.values()]
        signal[k]=dict(median=statistics.median(v),median_absolute=statistics.median(map(abs,v)),positive=sum(x>0 for x in v),zero=sum(x==0 for x in v),n=len(v))
    (HERE/'ranking_diagnostic.json').write_text(json.dumps(dict(selected=comparison,log_likelihood_changes=signal),indent=2)+'\n')
if __name__=='__main__':main()
