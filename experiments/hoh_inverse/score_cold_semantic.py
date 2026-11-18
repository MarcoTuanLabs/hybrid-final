"""Rescore archived editorial decisions without access to models or passages."""
from collections import Counter
import json
from pathlib import Path
P=Path(__file__).resolve().parent
def read(name):return json.loads((P/(name+'.json')).read_text())
def main():
    adjud=read('blind_cold_assessment')
    status={k:v['decision'] for k,v in adjud.items()}
    literal=read('blind_cold_literals')
    assert not set(status)&set(literal)
    status.update(dict.fromkeys(literal,'accept'))
    links=read('blind_cold_links')
    assert set(status)=={s for row in links.values() for s in row.values()} and len(links)==32
    summary={}
    for m in read('semantic_scope')['methods']:
        counts=Counter(status[row[m]] for row in links.values())
        summary[m]=[counts['accept'],counts['accept']+counts['uncertain']]
    obj=dict(n=len(links),literal_pairs=len(literal),reviewed_pairs=len(adjud),decisions=dict(Counter(v['decision'] for v in adjud.values())),range=summary)
    (P/'cold_semantic_summary.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj))
if __name__=='__main__':main()
