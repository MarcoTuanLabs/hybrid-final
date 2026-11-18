"""Prepare blinded individual responses and link them back only during final scoring."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;SRC=HERE.parent/'hoh_boundary'
def norm(s):return ' '.join(s.casefold().split())
def main():
    labels=json.loads((SRC/'test_labels.json').read_text());items={q['id']:q for q in json.loads((SRC/'test_inputs.json').read_text())}
    entries={};links={}
    for family,base in [('cpu','full'),('qwen','prefix'),('nocache','full')]:
        for r in map(json.loads,(SRC/('test_'+family+'.jsonl')).read_text().splitlines()):
            uid=r['id'];q=items[uid];gold=labels[uid]['current']
            for mode in [base,'guarded_unique']:
                answer=r['policies'][mode]['current']
                k=hashlib.sha256(('hoh-semantic-1118|'+uid+'|'+norm(answer)).encode()).hexdigest()[:12]
                v=dict(question=q['question'],reference=gold,evidence=q['current'],response=answer,
                       title=q['title'],source='https://en.wikipedia.org/?curid='+str(q['page']))
                if k in entries:assert norm(entries[k]['response'])==norm(answer)
                else:entries[k]=v
                links.setdefault(family+'_'+mode,{})[uid]=k
    blind=[dict(key=k,**v) for k,v in sorted(entries.items()) if norm(v['response'])!=norm(v['reference'])]
    (HERE/'blind.json').write_text(json.dumps(blind,ensure_ascii=False)+'\n')
    (HERE/'linkage.json').write_text(json.dumps(links)+'\n')
    (HERE/'automatic.json').write_text(json.dumps([k for k,v in entries.items() if norm(v['response'])==norm(v['reference'])])+'\n')
    print(len(blind),'nonliteral distinct responses;',len(entries),'unique case/response pairs')
if __name__=='__main__':main()
