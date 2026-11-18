"""Compile a policy-blind sensitivity review on the 32 exposed development questions."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'hoh_updates'
def read(p):return json.loads(p.read_text())
def compact(text):return ' '.join(text.lower().split())
def main():
    cfg=read(HERE/'semantic_scope.json')
    questions={r['id']:r['question'] for r in read(SOURCE/'questions.json')}
    labels=read(SOURCE/'labels.json');spans=read(HERE/'spans.json');source=read(HERE/'pool.json')
    out={};links={};literal=set()
    for row in map(json.loads,(HERE/'answers.jsonl').read_text().splitlines()):
        uid=row['id'];label=labels[uid];passage=next(r['current'] for r in source[uid] if r['key']==label['source_key'])
        answers=dict(row['answers']);answers['upr_top1_extractive']=spans[uid]['answers'][row['ranked']['upr'][0]]['answer']
        links[uid]={}
        for m in cfg['methods']:
            reply=answers[m];blind=hashlib.sha256((uid+'|'+compact(reply)).encode()).hexdigest()[:12]
            links[uid][m]=blind
            if compact(reply)==compact(label['current']):literal.add(blind)
            else:out[blind]=dict(question=questions[uid],reference=label['current'],answer=reply,source=passage)
    assert len(links)==32
    for name,val in [('blind_cold_rows',dict(sorted(out.items()))),('blind_cold_links',links),('blind_cold_literals',sorted(literal))]:
        (HERE/(name+'.json')).write_text(json.dumps(val,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
