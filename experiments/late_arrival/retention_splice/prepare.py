"""Load only preregistered IDs from the official data snapshot."""
import hashlib,json,sys
from pathlib import Path
p=Path(__file__).resolve().parent
reg=json.loads((p/'contract.json').read_text())
split=json.loads((p.parent/'suffix_transport/split.json').read_text())
path=Path(reg['dataset'])
assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==split['source_sha256']
source={r['question_id']:r for r in json.loads(path.read_text())}
items=[];labels={}
for uid in split['reserve']:
    row=source[uid]
    arrivals=sorted(row['haystack_session_ids'].index(s) for s in row['answer_session_ids'])
    def turns(indices):
        return [row['haystack_dates'][i]+' User: '+t['content'] for i in indices for t in row['haystack_sessions'][i] if t['role']=='user']
    items.append(dict(id=uid,prior=turns(arrivals[:-1]),insert=turns(arrivals[-1:]),later=turns(range(arrivals[-1]+1,len(row['haystack_sessions']))),question=row['question'],date=row['question_date']))
    labels[uid]=row['answer']
assert len(items)==34 and not set(labels).intersection(split['dev'])
(p/'inputs.json').write_text(json.dumps(items,ensure_ascii=False)+'\n')
(p/'labels.json').write_text(json.dumps(labels,ensure_ascii=False)+'\n')
