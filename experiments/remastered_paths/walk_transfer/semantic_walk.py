"""Evaluate the few-shot comparator with semantic relation matching and open-relation fallback."""
import copy
import json
import sys
import time
from pathlib import Path

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'bridge_reads'))
from engine import Engine
from bridge import Bank


def read(f):
    return json.loads(f.read_text())


def main(tag,model):
    reg=read(P/'semantic_contract.json')
    import hashlib
    for rel,expected in reg['pins'].items():
        assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==expected,rel
    engine=Engine(model)
    ids=engine.encode((P/'relation2subq_prompt2.txt').read_text())
    common=engine.prefill(ids)
    banks=read(P.parent/'remastered/banks.json')
    schema=read(P.parent/'typed_reads/schema.json')
    items={r['id']:r for r in read(P.parent/'state_parser/items.json') if r['split']=='exposed'}
    pplans=read(P/(tag+'_semantic_paths.json'))
    rows=list(map(json.loads,(P/(tag+'_rows.jsonl')).read_text().splitlines()))
    dest=P/(tag+'_semantic_rows.jsonl')
    done={r['id'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    begin=time.monotonic()
    with dest.open('a') as out:
        for row in rows:
            uid=row['id']
            if uid in done:
                continue
            bank=Bank(banks[row['bank']]['documents'])
            answer=row['reference'];hops=[]
            if row['route']=='context' and pplans[uid]:
                entity=items[uid]['root']
                for step in pplans[uid]:
                    rel=step['rel']
                    idx=bank.link(entity,rel) if rel else None
                    if idx is not None:
                        entity=bank.documents[idx]['obj']
                        hops.append(dict(doc=idx,answer=entity))
                        continue
                    if rel:
                        question=schema[rel].format(entity)
                    else:
                        prompt='Given this relation: "'+step['label']+'" and this subject: "'+entity+'",\nThe corresponding question is "'
                        raw,_=engine.answer(engine.encode(prompt),copy.deepcopy(common),stop=32)
                        question=raw.strip().strip('"')
                    entity=row['primitive'].get(question)
                    if entity is None:
                        entity=engine.factual(question)
                    hops.append(dict(question=question,answer=entity))
                answer=entity
            out.write(json.dumps(dict(id=uid,answer=answer,hops=hops),ensure_ascii=False)+'\n');out.flush()
    (P/(tag+'_semantic_receipt.json')).write_text(json.dumps(dict(wall_s=time.monotonic()-begin,additional_positions=engine.positions,prior=len(done)))+'\n')
if __name__=='__main__':main(*sys.argv[1:])
