"""Make a page-disjoint pilot and build an answer-free bank of paired evidence."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import pyarrow.parquet as pq

P=Path(__file__).resolve().parent
def digest(s):return sha256(s.encode()).hexdigest()
def clean(s):return ' '.join(s.split())
def dump(name,v):(P/name).write_text(json.dumps(v,ensure_ascii=False)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path);a=ap.parse_args()
    cfg=json.loads((P/'selection.json').read_text())
    with a.source.open('rb') as f:assert __import__('hashlib').file_digest(f,'sha256').hexdigest()==cfg['sha256']
    excluded=set(cfg['excluded_title_hashes'])
    groups=defaultdict(list);bank={}
    for batch in pq.ParquetFile(a.source).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            doc=row['document'];pid=doc['id'];title=doc['title'];date=str(row['last_modified_time'])[:10]
            old=[x for x in row['outdated_infos'] if x['evidence'] and str(x['last_modified_time'])[:10]<date]
            if not old or not row['evidence'] or not row['question']:continue
            prior=max(old,key=lambda x:x['last_modified_time']);before=clean(prior['evidence']);after=clean(row['evidence'])
            rd=str(prior['last_modified_time'])[:10];key=digest('\n'.join([pid,rd,before,date,after]))
            bank[key]=[pid,title,rd,before,date,after]
            if digest(title) in excluded:continue
            q=clean(row['question']);groups[pid].append((digest('hoh-source-seed-1/'+pid+'/'+q+'/'+key),q,key,row['answer'],prior['answer']))
    pages=sorted(groups,key=lambda key:digest('hoh-page-20261118/'+key))
    assert len(pages)>352
    dev=pages[:32];reserve=pages[32:128];cal=pages[128:352]
    inp=[];gold={};selection={}
    for split,ids in [('dev',dev),('reserve',reserve),('calibration',cal)]:
        selection[split]=ids
        if split!='dev':continue
        for pid in ids:
            uid,q,key,new,old=min(groups[pid])
            inp.append(dict(id=uid,question=q))
            gold[uid]=dict(current=new,previous=old,source_key=key,page=pid)
    dump('questions.json',inp);dump('labels.json',gold);dump('split.json',selection)
    with (P/'bank.jsonl').open('w') as f:
        for key,v in sorted(bank.items()):f.write(json.dumps([key,*v],ensure_ascii=False)+'\n')
    dump('source_stats.json',dict(documents=len(groups),evidence_pairs=len(bank),queries=len(inp),reserved=len(reserve),calibration=len(cal)))
if __name__=='__main__':main()
