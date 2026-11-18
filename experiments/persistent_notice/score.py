"""Keep factual accuracy, retention across turns and exact execution fidelity separate."""
import argparse
import json
from pathlib import Path
import random
import re
import statistics
import string

HERE=Path(__file__).resolve().parent
def norm(text):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text.lower().translate(str.maketrans('','',string.punctuation))).split())
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('public',type=Path);a=ap.parse_args()
    labels={(v['source'],v['id']):[set(map(norm,v[k])) for k in ['updated','stable','updated']] for v in json.loads((a.public/'experiments/read_projection/data/dev_labels.json').read_text())}
    rows=list(map(json.loads,(HERE/'rows.jsonl').read_text().splitlines()));assert len(rows)==len({(r['source'],r['id']) for r in rows})==12
    methods=sorted(w['mode'] for w in rows[0]['arms']);by={m:[] for m in methods};stats={}
    for r in rows:
        assert sorted(w['mode'] for w in r['arms'])==methods
        full=next(w for w in r['arms'] if w['mode']=='full')
        for w in r['arms']:
            hit=[bool(t.strip()) and norm(t.strip().splitlines()[0]) in names for t,names in zip(w['texts'],labels[r['source'],r['id']],strict=True)]
            by[w['mode']].append(w|dict(hits=hit,src=r['source'],doc_id=r['id'],exact=[v==f for v,f in zip(w['tokens'],full['tokens'],strict=True)],joint=hit[0] and hit[2]))
    for m,rr in by.items():
        stats[m]=dict(turn_accuracy=[sum(v['hits'][j] for v in rr) for j in range(3)],joint_changed=sum(v['joint'] for v in rr),turn_sequence_matches=[sum(v['exact'][j] for v in rr) for j in range(3)],
                      all_three_exact=sum(all(v['exact']) for v in rr),all_three_correct=sum(all(v['hits']) for v in rr),by_task={s:[sum(v['hits'][j] for v in rr if v['src']==s) for j in range(3)] for s in ['babi','mquake']},
                      means={k:statistics.mean(v[k] for v in rr) for k in ['wire_bytes','old_cache_logical_bytes','old_setup_s','ready_cold_s','ready_warm_s','complete_cold_s','complete_warm_s','downstream_s']})
    pair={}
    for m in methods:
        other='old_notice_rag';win=[[],[],[]];loss=[[],[],[]];jointwin=[];jointloss=[]
        for v,b in zip(by[m],by[other],strict=True):
            key=[v['src'],v['doc_id']]
            for j in range(3):
                if v['hits'][j] and not b['hits'][j]:win[j].append(key)
                if b['hits'][j] and not v['hits'][j]:loss[j].append(key)
            if v['joint'] and not b['joint']:jointwin.append(key)
            if b['joint'] and not v['joint']:jointloss.append(key)
        pair[m]=dict(wins=win,losses=loss,joint_wins=jointwin,joint_losses=jointloss)
    m='native_notice_rag';cheap=['old','old_notice','old_notice_rag','fresh_rag','bounded16','checkpoint']
    needed=max(stats[x]['joint_changed'] for x in cheap)+2;no_loss=all((v['hits'][1] or not b['hits'][1]) for x in cheap for v,b in zip(by[m],by[x],strict=True))
    rel=stats[m]['means']['complete_cold_s']/stats['full']['means']['complete_cold_s']
    rng=random.Random(8317);ratios=[]
    for _ in range(5000):
        it=rng.choices(range(len(rows)),k=len(rows))
        ratios.append(sum(by[m][i]['complete_cold_s'] for i in it)/sum(by['full'][i]['complete_cold_s'] for i in it))
    ratios.sort()
    s=dict(documents=len(rows),turns=len(rows)*3,policies=stats,paired_vs_notice_retrieval=pair,required_joint_changed=needed,no_extra_unaffected_loss=no_loss,cold_ratio_vs_full=rel,paired_ratio_ci95=[ratios[124],ratios[4874]],
           promotion=stats[m]['joint_changed']>=needed and no_loss and rel<=.8,bounded_skipped_documents=sum(v['skipped_suffix']>0 for v in by['bounded16']))
    (HERE/'summary.json').write_text(json.dumps(s,sort_keys=True,indent=2)+'\n');print(json.dumps({k:s[k] for k in ['documents','required_joint_changed','no_extra_unaffected_loss','cold_ratio_vs_full','promotion','bounded_skipped_documents']}))
if __name__=='__main__':main()
