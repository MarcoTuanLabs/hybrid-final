"""Build a subscription workload from historical page IDs and old text only."""
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'hoh_updates'
def words(s):
    return set(re.findall(r'[^\W_]+',s.casefold())) - set('a an the is that to what are was how who in of and for does'.split())

def main():
    queries = json.loads((SOURCE/'questions.json').read_text())
    pages = json.loads((SOURCE/'split.json').read_text())['dev']
    matches = {p: [] for p in pages}
    with (SOURCE/'bank.jsonl').open() as f:
        for key, page, title, before, old, after, current in map(json.loads,f):
            if page in matches:
                matches[page].append(dict(key=key,title=title,old=old,current=current,old_date=before,date=after))
    result = []
    for q, page in zip(queries,pages, strict=True):
        def rank(row):
            w = words(row['old']); return (-len(w & words(q['question']))/math.sqrt(max(1,len(w))),row['old_date'],row['key'])
        row = min(matches[page],key=rank)
        result.append(dict(**q,page=page,**row))
    (HERE/'inputs.json').write_text(json.dumps(result,ensure_ascii=False)+'\n')

if __name__ == '__main__':
    main()
