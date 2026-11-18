"""Select persona-disjoint questions and expose only dialogue available at request time."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def hash_text(s):return hashlib.sha256(s.encode()).hexdigest()
def save(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False)+'\n')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('data',type=Path);args=ap.parse_args()
    cfg=json.loads((HERE/'selection.json').read_text())
    with args.data.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==cfg['source_sha256']
    types={'dynamic_conflict':'medium','static_conflict':'hard','conditional_conflict':'hard'}
    people={}
    with args.data.open() as f:
        for line in f:
            person=json.loads(line);uid=person['ID']
            if uid==cfg['schema_preview_persona']:continue
            by_type=defaultdict(list);sessions=sorted(person['Full_Session_Chain'],key=lambda s:s['Session_ID'])
            for slot,s in enumerate(sessions):
                for q in s['Session_Questions']:
                    k=q['conflict_type']
                    if types.get(k)!=q['difficulty']:continue
                    key=hash_text('mem-witness-1/q/'+uid+'/'+str(s['Session_ID'])+'/'+q['question_id'])
                    by_type[k].append((key,slot,q))
            if set(by_type)==set(types):people[uid]=(sessions,by_type)
    ids=sorted(people,key=lambda uid:hash_text('mem-witness-1/person/'+uid))
    assert len(ids)==29
    groups={'dev':ids[:8],'transfer':ids[8:24],'other':ids[24:]};save('split.json',groups)
    samples=[];labels={};histories={}
    for uid in groups['dev']:
        sessions,by_type=people[uid]
        for kind in types:
            key,n,q=min(by_type[kind]);rows=[];visible=sessions[:n+1]
            for s in visible:
                turns=[]
                for dialog in s['Session_Dialogue'].values():
                    for t in dialog:
                        speaker=t.get('role','assistant' if 'assistant' in t else None)
                        assert speaker in ('user','assistant')
                        text=t.get('content')
                        if text is None and isinstance(t.get('assistant'),dict):
                            text=t['assistant'].get('content')
                        if text:
                            assert isinstance(text,str)
                            turns.append(dict(speaker=speaker,text=text))
                rows.append(dict(session=s['Session_ID'],date=s['Date'],turns=turns))
            histories[key]=rows
            samples.append(dict(id=key,persona=uid,question=q['question'],date=sessions[n]['Date']))
            labels[key]=dict(answer=q['answer'],category=kind,persona=uid)
    assert len(samples)==24 and len(labels)==24
    save('questions.json',samples);save('labels.json',labels);save('histories.json',histories)
    save('prep_receipt.json',dict(eligible_personas=len(ids),development=len(samples),reserve_personas=len(groups['transfer']),
          sessions_per_request=[len(histories[v['id']]) for v in samples]))
if __name__=='__main__':main()
