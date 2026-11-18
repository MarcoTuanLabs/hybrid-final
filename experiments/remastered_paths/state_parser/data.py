"""Separate roots and delexicalized questions, then fit an entity span tagger on training only."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

P=Path(__file__).resolve().parent
R=P.parent
def h(x):return hashlib.sha256(str(x).encode()).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def mask(q,entity):
    return norm(re.sub(re.escape(entity),' entity ',q,flags=re.I)) if entity else norm(q)
def tokenize(q):return list(re.finditer(r"[\w]+|[^\w\s]",q))
def features(tokens,pos):
    feat={'bias':1.,'position':pos/max(len(tokens)-1,1),'end':(len(tokens)-pos)/len(tokens)}
    for off in range(-3,4):
        i=pos+off
        label=tokens[i][0] if 0<=i<len(tokens) else '<bound>'
        prefix=str(off)
        feat[prefix+'word']=label.casefold()
        feat[prefix+'shape']=('title' if label.istitle() else 'upper' if label.isupper() else 'lower' if label.islower() else 'digit' if label.isdigit() else 'other')
    return feat

class SpanTagger:
    def __init__(self,rows):
        feats=[];labels=[]
        for row in rows:
            q=row['q'];name=row['gold_root'];m=re.search(re.escape(name),q,re.I)
            if not m:continue
            toks=tokenize(q)
            for i,t in enumerate(toks):
                feats.append(features(toks,i));labels.append(int(m.start()<=t.start()<m.end()))
        self.vectorizer=DictVectorizer()
        matrix=self.vectorizer.fit_transform(feats)
        matrix.indices=matrix.indices.astype(np.int32)
        matrix.indptr=matrix.indptr.astype(np.int32)
        self.model=LogisticRegression(C=2,max_iter=300,solver='liblinear',random_state=197).fit(matrix,labels)

    def predict(self,q):
        tokens=tokenize(q)
        raw=self.model.decision_function(self.vectorizer.transform([features(tokens,i) for i in range(len(tokens))]))
        best=(-float('inf'),0,0)
        for a in range(len(tokens)):
            for b in range(a+1,min(len(tokens),a+12)+1):
                val=float(sum(raw[a:b]))
                if val>best[0]:best=val,a,b
        return q[tokens[best[1]].start():tokens[best[2]-1].end()]

def main():
    cfg=json.loads((P/'data_contract.json').read_text());source=R/'remastered/CF3k.parquet'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['source_sha256']
    data=pq.read_table(source).to_pylist()
    valid={r['case_id']:r for r in data if r['case_id'] not in set(json.loads((R/'remastered/source_check.json').read_text())['excluded'])}
    official={x['id']:x['question'] for b in json.loads((R/'remastered/banks.json').read_text()) for x in b['queries']}
    keys=json.loads((R/'remastered/labels.json').read_text())
    test=[];blocked=set();forms=set()
    def row(q,s,rels,code,cid):
        return dict(q=q,gold_root=s,rels=rels,id=code,case=cid)
    for uid,key in keys.items():
        d=valid[key['case']];blocked.add(d['orig_triples'][0][0])
        if key['structure']=='multi':
            subject=d['orig_triples_labeled'][0][0];rels=[t[1] for t in d['orig_triples']]
        elif key['role']=='edited':
            f=d['requested_rewrite'][0];subject=f['subject'];rels=[f['relation_id']]
        else:
            subject=d['orig_triples_labeled'][0][0];rels=[d['orig_triples'][0][1]]
        q=official[uid];forms.add(mask(q,subject));test.append(row(q,subject,rels,uid,key['case']))
    # Reserve a separate corpus section by underlying root before constructing training and validation.
    pool=sorted((x for x in valid.values() if x['orig_triples'][0][0] not in blocked),key=lambda x:h('state-plan-split-2026/'+str(x['case_id'])))
    reserve=[];val=[]
    for x in pool:
        root=x['orig_triples'][0][0];s=x['orig_triples_labeled'][0][0];q=x['questions'][0]
        if root in blocked or mask(q,s) in forms or s.casefold() not in q.casefold():continue
        group=reserve if len(reserve)<192 else val
        group.append(row(q,s,[t[1] for t in x['orig_triples']],h(q),x['case_id']))
        blocked.add(root);forms.add(mask(q,s))
        if len(val)==128:break
    training=[];seen=set(forms);train_roots=set()
    for x in pool:
        root=x['orig_triples'][0][0];s=x['orig_triples_labeled'][0][0]
        if root in blocked:continue
        train_roots.add(root)
        for q in x['questions']:
            if s.casefold() not in q.casefold():continue
            typed=mask(q,s)
            if typed in seen:continue
            seen.add(typed)
            training.append(row(q,s,[t[1] for t in x['orig_triples']],h(q),x['case_id']))
        if len(training)>=1800:break
    assert len(reserve)==192 and len(val)==128 and training
    tag=SpanTagger(training)
    out=[];gt={}
    for segment,seq in [('train',training),('validation',val),('exposed',test)]:
        for d in seq:
            entity=tag.predict(d['q'])
            # No gold root or relation enters the neural feature file.
            out.append(dict(id=d['id'],split=segment,q=d['q'],root=entity,text=mask(d['q'],entity)))
            gt[d['id']]=(d['gold_root'],d['rels'])
    assert len(gt)==len(out)
    (P/'items.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
    (P/'targets.json').write_text(json.dumps(gt,ensure_ascii=False)+'\n')
    (P/'reserve.json').write_text(json.dumps([dict(id=d['id'],case=d['case']) for d in reserve])+'\n')
    stats={g:sum(x['split']==g for x in out) for g in ('train','validation','exposed')}
    (P/'split.json').write_text(json.dumps(dict(counts=stats,training_roots=len(train_roots),reserved_roots=len(reserve),overlap_train_gold_masked_forms=0,span_exact={g:sum(norm(x['root'])==norm(gt[x['id']][0]) for x in out if x['split']==g) for g in stats}),indent=2)+'\n')
    print((P/'split.json').read_text())
if __name__=='__main__':main()
