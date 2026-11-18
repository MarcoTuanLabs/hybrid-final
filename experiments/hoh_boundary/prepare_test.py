"""Draw the registered page-disjoint cohort for the repeated-query efficiency study."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_refresh'))
from inputs import words
def hash_text(s):
    return hashlib.sha256(s.encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);a=p.parse_args()
    source=HERE.parent/'hoh_updates'
    cfg=json.loads((source/'selection.json').read_text())
    with a.source.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==cfg['sha256']
    reserved={page for ids in json.loads((source/'split.json').read_text()).values() for page in ids}
    excluded=set(cfg['excluded_title_hashes'])
    catalog=defaultdict(list)
    for batch in pq.ParquetFile(a.source).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            doc=row['document'];pid=doc['id'];title=doc['title'];date=str(row['last_modified_time'])[:10]
            if pid in reserved or hash_text(title) in excluded:continue
            priors=[x for x in row['outdated_infos'] if x['evidence'] and str(x['last_modified_time'])[:10]<date]
            if not priors or not row['evidence'] or not row['question']:continue
            prior=max(priors,key=lambda x:x['last_modified_time'])
            old=' '.join(prior['evidence'].split());new=' '.join(row['evidence'].split());rd=str(prior['last_modified_time'])[:10]
            key=hash_text('\n'.join([pid,rd,old,date,new]));q=' '.join(row['question'].split())
            uid=hash_text('hoh-source-seed-1/'+pid+'/'+q+'/'+key)
            catalog[pid].append((uid,q,key,row['answer'],prior['answer']))
    selected=sorted(catalog,key=lambda p:hash_text('hoh-refresh-test-20261118/'+p))[:96]
    assert not set(selected)&reserved and len(set(selected))==96
    bank=defaultdict(list)
    with (source/'bank.jsonl').open() as f:
        for key,page,title,rd,old,nd,new in map(json.loads,f):
            if page in selected:
                bank[page].append(dict(key=key,title=title,old=old,current=new,old_date=rd,date=nd))
    rows=[];labels={}
    for page in selected:
        uid,q,correct,new,old=min(catalog[page])
        def rank(row):
            w=words(row['old'])
            return (-len(w&words(q))/max(1,len(w))**.5,row['old_date'],row['key'])
        rows.append(dict(id=uid,question=q,page=page,**min(bank[page],key=rank)))
        labels[uid]=dict(page=page,source_key=correct,current=new,previous=old)
    for name,obj in [('test_inputs',rows),('test_labels',labels)]:
        (HERE/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
