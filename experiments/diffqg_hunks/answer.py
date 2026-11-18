"""Generate free factual answers with matched controls and candidate evidence proposals."""
import argparse,json,math,sys,time
from pathlib import Path
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine,generate

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);a=parser.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    read=lambda name:json.loads((HERE/name).read_text())
    qs=read('questions.json');ranks=read('ranks.json');bank={r['key']:r for r in read('bank.json')}
    spans={r['id']:r for r in map(json.loads,(HERE/'span_rows.jsonl').read_text().splitlines())}
    output=HERE/'open_rows.jsonl';seen={r['id'] for r in map(json.loads,output.read_text().splitlines())} if output.exists() else set()
    start=time.monotonic()
    with output.open('a') as f:
        for j,q in enumerate(qs):
            uid=q['id']
            if uid in seen:continue
            t=time.monotonic();n=engine.positions;actual={};answers={}
            def answer(docs):
                prompt=('Answer the question from the updated Wikipedia evidence. Give only the shortest complete factual answer. '
                        'If a proposed phrase addresses a different question, answer from the evidence.\n'
                        +'\n\n'.join(docs)+'\nRequested snapshot year: '+q['as_of']+'\nQuestion: '+q['question'])
                if prompt not in actual:actual[prompt]=generate(engine,prompt,40)
                return actual[prompt]
            keys=ranks[uid]['main'];data=spans[uid]['docs']
            text=lambda key:bank[key]['title']+' | '+bank[key]['date']+'\n'+bank[key]['current']
            for k in [1,3,6]:
                answers['current_'+str(k)]=answer([text(key) for key in keys[:k]])
            answers['paired_3']=answer([text(key)+'\nEarlier edition ('+bank[key]['old_date']+'): '+bank[key]['old'] for key in keys[:3]])
            for mode in ['current','changed','whole','rollback','symmetric']:
                ordered=sorted(enumerate(keys),key=lambda v:-(data[v[1]]['policy'][mode][0]-math.log1p(v[0])))
                docs=[]
                for _,key in ordered[:3]:
                    phrase=data[key]['policy'][mode][1]
                    docs.append(text(key)+'\nProposed current answer: '+phrase)
                answers[mode+'_hint']=answer(docs)
            f.write(json.dumps(dict(id=uid,answers=answers,actual_calls=len(actual),positions=engine.positions-n,seconds=time.monotonic()-t),ensure_ascii=False)+'\n');f.flush()
            print(j+1,round(time.monotonic()-start,1),flush=True);mx.clear_cache()
            if time.monotonic()-start>1800:raise TimeoutError('30-minute free generation comparison')
if __name__=='__main__':main()
