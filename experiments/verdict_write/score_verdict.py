"""Recompute every native-write judgment using both the full vocabulary and legal answer set."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    split=json.loads((a.repo/'experiments/vitaminc_read/split.json').read_text())['pilot']
    rows=[json.loads(l) for l in (a.out/'rows.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[s['id'] for s in split]
    modes=['old','full','write_native','white','normalized','unwritten','recurrent','attention']
    examples=[]
    for r in rows:
        for claim in r['claims']:
            g=claim['gold']
            for j,v in enumerate(claim['arms']):
                assert set(v)==set(modes)
                clean=v['old']['class_id']==g[1-j] and v['full']['class_id']==g[j]
                examples.append((g[j],r['labels'][g[j]],g[0]!=g[1],clean,v))
    def count(m,key):
        return dict(total=sum(v[m][key]==(g if key=='class_id' else tok) for g,tok,ch,n,v in examples),
                  changed=sum(ch and v[m][key]==(g if key=='class_id' else tok) for g,tok,ch,n,v in examples),
                  stable=sum(not ch and v[m][key]==(g if key=='class_id' else tok) for g,tok,ch,n,v in examples),
                  both_native_changed=sum(ch and n and v[m][key]==(g if key=='class_id' else tok) for g,tok,ch,n,v in examples))
    s={m:dict(legal=count(m,'class_id'),vocab=count(m,'top')) for m in modes}
    others=['write_native','white','unwritten'];primary=s['normalized']['legal']
    loss=sum(not ch and v['old']['class_id']==g and v['normalized']['class_id']!=g for g,_,ch,_,v in examples)
    gate=primary['changed']>=max(s[m]['legal']['changed'] for m in others)+3 and primary['both_native_changed']>max(s[m]['legal']['both_native_changed'] for m in others) and not loss
    out=dict(pages=len(rows),endpoints=len(examples),changed=sum(z[2] for z in examples),stable=sum(not z[2] for z in examples),both_native_changed=sum(z[2] and z[3] for z in examples),stable_old_correct_lost=loss,methods=s,promote=bool(gate))
    assert out['endpoints']==96 and out['stable']==10
    (a.out/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
