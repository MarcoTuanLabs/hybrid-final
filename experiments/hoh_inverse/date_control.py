"""Answer directly from six fresh passages, with and without actual snapshot dates."""
import argparse,json,time
from pathlib import Path
import mlx.core as mx
from query import generate,Engine,SOURCE
from answer import prompt
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    qs=json.loads((SOURCE/'questions.json').read_text());pool=json.loads((HERE/'pool.json').read_text())
    dest=HERE/'date_answers.jsonl';known={r['id'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as f:
        for q in qs:
            i=q['id']
            if i in known:continue
            t=time.monotonic();n=engine.positions;ans={}
            docs=pool[i];ans['all_six']=generate(engine,prompt(q['question'],docs,{},'expanded'),40)
            evidence='\n\n'.join(r['title']+' | Wikipedia snapshot '+r['date']+'\n'+r['current'] for r in docs)
            text=('Find the passage answering this exact question. Use the most recent applicable fact if sources disagree. '
                  'Answer with only the complete short factual answer.\n'+evidence+'\nUser question: '+q['question'])
            ans['all_six_dated']=generate(engine,text,40)
            f.write(json.dumps(dict(id=i,answers=ans,positions=engine.positions-n,seconds=time.monotonic()-t),ensure_ascii=False)+'\n');f.flush()
            mx.clear_cache()
            if time.monotonic()-start>1800:raise TimeoutError('date control phase')
if __name__=='__main__':main()
