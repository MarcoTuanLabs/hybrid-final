"""Create a query-side hypothetical passage without consulting the retrieval corpus."""
import argparse,json,sys,time
from pathlib import Path
import mlx.core as mx

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'hoh_updates'
sys.path.insert(0,str(SOURCE))
from generate import Engine,history

def generate(engine,prompt,limit):
    cache,tail=history(engine,prompt)
    return engine.answer(tail,cache,stop=limit)[0].strip()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    qs=json.loads((SOURCE/'questions.json').read_text());out={};start=time.monotonic()
    for q in qs:
        t=time.monotonic();n=engine.positions
        reply=generate(engine,'Write one plausible Wikipedia evidence sentence that would answer the following question. Include distinctive entity and relation words. Output only the sentence.\nQuestion: '+q['question'],52)
        out[q['id']]=dict(hypothetical=reply,positions=engine.positions-n,seconds=time.monotonic()-t)
        mx.clear_cache()
        if time.monotonic()-start>1800:raise TimeoutError('query phase')
    (HERE/'rewrites.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
