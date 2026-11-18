"""Build an answer-free corpus of complete revision contexts for expert-written questions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def clean(s):return ' '.join(s.replace('[SEP]',' ').split())
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False)+'\n')
def load(path):return json.loads(path.read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',type=Path);a=parser.parse_args()
    cfg=load(HERE/'selection.json')
    assert hashlib.sha256(a.input.read_bytes()).hexdigest()==cfg['sha256']
    prior=set(load(HERE.parent/'hoh_updates/selection.json')['excluded_title_hashes'])
    prior.add(sha('Rose Bowl (stadium)'))
    prior.update(load(HERE/'prior_subscription_title_hashes.json'))
    bank={};candidates=defaultdict(list)
    for line in a.input.read_text().splitlines():
        row=json.loads(line);topic=clean(row['topic']);old=clean(row['base_context']);new=clean(row['target_context'])
        if int(row['target_year'])<=int(row['base_year']):continue
        key=sha('\n'.join([topic,old,new,str(row['base_year']),str(row['target_year'])]))
        bank[key]=dict(key=key,title=topic,old=old,current=new,old_date=str(row['base_year']),date=str(row['target_year']))
        question=clean(row['q'])
        if not question or sha(topic) in prior:continue
        uid=sha('diffqg-question-v1|'+topic+'|'+question+'|'+key)
        candidates[topic].append((uid,question,key,row['a']))
    topics=sorted(candidates,key=lambda t:sha('diffqg-hunks-v1|'+t))
    assert len(topics)>=128
    groups=dict(dev=topics[:32],reserve=topics[32:128],other=topics[128:])
    questions=[];labels={}
    for topic in groups['dev']:
        uid,q,key,answer=min(candidates[topic]);questions.append(dict(id=uid,question=q,as_of=bank[key]['date']))
        labels[uid]=dict(current=answer,source_key=key,title=topic)
    dump(HERE/'bank.json',[bank[k] for k in sorted(bank)]);dump(HERE/'questions.json',questions);dump(HERE/'labels.json',labels)
    dump(HERE/'split.json',groups);dump(HERE/'prep_receipt.json',dict(corpus=len(bank),eligible_topics=len(topics),
          dev=len(questions),reserved=len(groups['reserve']),prior_titles=len(prior)))
if __name__=='__main__':main()
