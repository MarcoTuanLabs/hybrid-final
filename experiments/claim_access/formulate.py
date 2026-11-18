"""Formulate the query without access to a conversation or an answer key."""
import argparse
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine, generate
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('data',type=Path);a=ap.parse_args()
    qs=json.loads((a.data/'questions.json').read_text());engine=Engine(a.model);mx.set_cache_limit(64*1024**2)
    output={};start=time.monotonic()
    for q in qs:
        question=q['question'];t=time.monotonic();n=engine.positions
        vocab=generate(engine,'Give twelve short words or phrases that could occur in a first-person statement answering this question. Include common synonyms, inflections and different possible values. Separate phrases by commas on ONE line, without numbering or explanation. The question is: '+question,70)
        perspective=generate(engine,'Reword this question so it addresses the person directly as "you". Keep the meaning, reference and time frame identical. Write only the question in one line. Original question: '+question,40)
        output[q['id']]=dict(terms=vocab,question=perspective,seconds=time.monotonic()-t,positions=engine.positions-n)
        if time.monotonic()-start>300:raise TimeoutError('Query-side budget')
    (HERE/'formulations.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
