"""Inventory natural, labeled revision contrasts before any model run."""
from collections import Counter,defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PIN="544934677f5d133873e6d38f4557f8966f4efa5d3d70874ffe6913f2091b86b5"
p=Path(sys.argv[1])
with p.open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==PIN
versions=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
types=Counter();n=0
for line in p.open():
    row=json.loads(line);types[row['revision_type']]+=1;n+=1
    if row['revision_type']!='real':continue
    key=(row['page'],str(row['wiki_revision_id']),row['case_id'])
    versions[key][row['evidence']][row['claim']].add(row['label'])
pairs=defaultdict(lambda:defaultdict(set))
for (page,rev,case),by_text in versions.items():
    if len(by_text)!=2:continue
    a,b=sorted(by_text)
    for claim in by_text[a].keys() & by_text[b].keys():
        if len(by_text[a][claim])==len(by_text[b][claim])==1:
            labels=next(iter(by_text[a][claim])),next(iter(by_text[b][claim]))
            pairs[page,rev,a,b][claim].add(labels)
totals=Counter();pages=set();invariant_pages=set();both_ids=[]
for (page,rev,a,b),claims in pairs.items():
    labels=[next(iter(v)) for v in claims.values() if len(v)==1]
    changed=sum(x!=y for x,y in labels);stable=sum(x==y for x,y in labels)
    if changed:totals['any_changed']+=1
    if stable:totals['any_stable']+=1
    if changed and stable:
        totals['changed_and_stable']+=1;invariant_pages.add(page)
    if changed<2:continue
    a_tokens,b_tokens=a.split(),b.split()
    ops=[v for v in SequenceMatcher(None,a_tokens,b_tokens,autojunk=False).get_opcodes() if v[0]!='equal']
    if len(ops)!=1:continue
    _,lo,hi,x,y=ops[0]
    if not (1<=hi-lo<=8 and 1<=y-x<=8):continue
    totals['compact_replacement_two_changed']+=1;pages.add(page)
    if stable:
        totals['compact_with_native_stable']+=1;both_ids.append(hashlib.sha256(json.dumps([page,rev,a,b]).encode()).hexdigest())
s=dict(source="tals/vitaminc@be6febb761b0b2807687e61e0b5282e459df2fa0/dev.jsonl",sha256=PIN,all_rows=n,source_types=dict(types),
       natural_cases=len(versions),natural_page_revision_evidence_pairs=len(pairs),counts=dict(sorted(totals.items())),compact_pages=len(pages),
       same_pair_changed_and_stable_pages=len(invariant_pages),compact_with_stable_ids=sorted(both_ids),
       scope="Groups require exact evidence strings and the same page and revision. No chronological direction is inferred from row order. A native stable claim requires release labels on both texts. No model outputs or test-split rows used.")
(HERE/"summary.json").write_text(json.dumps(s,indent=2)+'\n')
print({k:v for k,v in s.items() if k not in ('scope','compact_with_stable_ids')})
