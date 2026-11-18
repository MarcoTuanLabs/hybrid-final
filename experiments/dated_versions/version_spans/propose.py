"""Search every available source row or paragraph for independently proposed short spans."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import torch
from transformers import AutoModelForQuestionAnswering,AutoTokenizer

P=Path(__file__).resolve().parent;R=P.parent


def snippet_chunks(edition):
    rows=[' | '.join(x for x in [r['section'],*r['cells']] if x) for r in edition['rows']]
    chunks=[]
    for start in range(0,len(rows),4):
        chunks.append('\n'.join(rows[max(0,start-1):start+5]))
    return list(dict.fromkeys([x for x in chunks+edition['paragraphs'] if x.strip()]))


def canonical(s):
    return ' '.join(re.findall(r'\w+',s.casefold()))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('weights',type=Path);a=ap.parse_args()
    cfg=json.loads((P/'contract.json').read_text())
    for f,h in cfg['pins'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h,f
    tok=AutoTokenizer.from_pretrained(a.weights,local_files_only=True)
    net=AutoModelForQuestionAnswering.from_pretrained(a.weights,local_files_only=True).eval();torch.set_num_threads(4)
    data=json.loads((R/'asof_revision/inputs.json').read_text())
    old={x['id']:x for x in map(json.loads,(R/'revision_pairs/4b_rows.jsonl').read_text().splitlines())}
    report=[];started=time.monotonic()
    for it in data:
        candidates=[]
        for ed,source in enumerate(it['editions']):
            chunks=snippet_chunks(source);opts={};position=0
            for start in range(0,len(chunks),8):
                batch=chunks[start:start+8]
                pack=tok([it['question']]*len(batch),batch,padding=True,truncation='only_second',max_length=384,return_tensors='pt',return_offsets_mapping=True)
                maps=pack.pop('offset_mapping').tolist()
                with torch.inference_mode():o=net(**pack)
                for j,document in enumerate(batch):
                    ids=pack.sequence_ids(j);offsets=maps[j]
                    a1=o.start_logits[j];a2=o.end_logits[j]
                    base=float(a1[0]+a2[0])
                    indices=[k for k,z in enumerate(ids) if z==1]
                    top_s=sorted(indices,key=lambda k:-float(a1[k]))[:6]
                    top_e=sorted(indices,key=lambda k:-float(a2[k]))[:6]
                    local=[]
                    for l in top_s:
                        for h in top_e:
                            if not l<=h<l+24:continue
                            text=document[offsets[l][0]:offsets[h][1]].strip()
                            if not text or not canonical(text) or '\n' in text or '|' in text:continue
                            score=float(a1[l]+a2[h])-base
                            local.append((score,text))
                    for val,text in sorted(local,reverse=True)[:5]:
                        key=canonical(text)
                        if key not in opts or opts[key]['score']<val:
                            opts[key]=dict(text=text,score=val,chunk=start+j)
                    position+=len(indices)
            ranked=sorted(opts.values(),key=lambda d:-d['score'])
            proposals=old[it['id']]['proposals'];inc=proposals['all'][ed]
            pool=[];seen=set()
            def add(s):
                key=canonical(s)
                if key and key not in seen:pool.append(s);seen.add(key)
            for name in ['all','snapshot','span','aligned_snapshot']:add(proposals[name][ed])
            for row in ranked:
                if len(pool)>=12:break
                add(row['text'])
            candidates.append(dict(pool=pool,span=ranked[0]['text'] if ranked else inc,top_spans=ranked[:20],encoded_context_tokens=position))
        report.append(dict(id=it['id'],editions=candidates))
    (P/'proposals.json').write_text(json.dumps(report,ensure_ascii=False)+'\n')
    (P/'receipt.json').write_text(json.dumps(dict(pages=len(data),wall_s=time.monotonic()-started))+'\n')
if __name__=='__main__':main()
