"""Score the frozen blinded judgments and bound paired improvements under all uncertain labels."""
import itertools,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def main():
    links=read('linkage.json');manual=read('adjudication.json');auto=set(read('automatic.json'))
    assert set(manual)=={r['key'] for r in read('blind.json')} and not auto&set(manual)
    uncertain=[k for k,v in manual.items() if v['judgment']=='uncertain']
    fixed={k:True for k in auto}|{k:v['judgment']=='accept' for k,v in manual.items() if k not in uncertain}
    values={m:[] for m in links};paired={m:[] for m in ['qwen','nocache','cpu']}
    for world in itertools.product([False,True],repeat=len(uncertain)):
        verdict=fixed|dict(zip(uncertain,world,strict=True))
        for m,by_id in links.items():
            assert len(by_id)==96
            values[m].append(sum(verdict[k] for k in by_id.values()))
        for family in paired:
            base=links[family+('_prefix' if family=='qwen' else '_full')]
            new=links[family+'_guarded_unique']
            paired[family].append(sum(verdict[new[i]]-verdict[base[i]] for i in base))
    bounds=lambda vals:{k:[min(v),max(v)] for k,v in vals.items()}
    res=dict(n=96,automatic_literals=len(auto),manual_responses=len(manual),uncertain=len(uncertain),scores=bounds(values),paired_change=bounds(paired))
    (HERE/'summary.json').write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(res))
if __name__=='__main__':main()
