"""Extract a bounded retrospective insertion workload from official timestamped user turns."""
import hashlib
import json
from pathlib import Path

p = Path(__file__).resolve().parent
source = Path('interpretability/runs/utility_pilot_v1/sources/longmemeval_s_cleaned.json')
raw = json.loads(source.read_text())
def turns(row, indices):
    return [row['haystack_dates'][i] + ' User: ' + turn['content'] for i in indices
            for turn in row['haystack_sessions'][i] if turn['role'] == 'user']
items, labels = [], {}
for row in raw:
    if row['question_type'] != 'knowledge-update' or row['question_id'].endswith('_abs'):
        continue
    arrivals = sorted(row['haystack_session_ids'].index(s) for s in row['answer_session_ids'])
    if len(arrivals) < 2:
        continue
    old = turns(row,arrivals[:-1])
    update = turns(row,arrivals[-1:])
    later = turns(row,range(arrivals[-1]+1,len(row['haystack_sessions'])))
    if len(' '.join(later).split()) < 450 or not old or not update:
        continue
    uid = row['question_id']
    items.append(dict(id=uid,prior=old,insert=update,later=later,question=row['question'],date=row['question_date']))
    labels[uid]=row['answer']
items.sort(key=lambda x: hashlib.sha256(('hybrid-late-arrival-dev-v1/'+x['id']).encode()).digest())
# Do not materialize a holdout answer file at development time.
pilot=items[:16]
(p/'inputs.json').write_text(json.dumps(pilot,ensure_ascii=False)+'\n')
(p/'labels.json').write_text(json.dumps({x['id']:labels[x['id']] for x in pilot},ensure_ascii=False)+'\n')
(p/'split.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),dev=[x['id'] for x in pilot],reserve=[x['id'] for x in items[16:]],rule='knowledge-update, nonabstention, >=2 time-ordered evidence sessions, >=450 words of later user turns; choose by id hash without inspecting model outputs. Dataset labels only define simulated insertion boundary and evaluation.'),indent=2)+'\n')
print('eligible',len(items),'development',len(pilot))
