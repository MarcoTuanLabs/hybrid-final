"""Compute how much work exact common-prefix reuse could avoid in the fixed prompts."""
import argparse
import json
from pathlib import Path
import sys
import statistics
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('public',type=Path);a=ap.parse_args()
    sys.path.insert(0,str(a.public.resolve()/'experiments/quantized_write'))
    from generator import make_example
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);marker,=tok.encode('<think>',add_special_tokens=False)
    out=[]
    for seed in range(27112000,27112008):
        for surface in ['table','prose']:
            for nuisance in [False,True]:
                v=make_example(seed,surface,nuisance);pair=[]
                for side in ['a','b']:
                    text=tok.apply_chat_template([dict(role='user',content=v['prefix_'+side]+v['suffix'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    ids=tok.encode(text,add_special_tokens=False);pair.append(ids[:ids.index(marker)])
                left,right=pair;assert len(left)==len(right);cut=next(i for i,(l,r) in enumerate(zip(left,right)) if l!=r)
                out.append(dict(seed=seed,surface=surface,nuisance=nuisance,per_branch_tokens=len(left),common=cut))
    original=sum(v['per_branch_tokens']*2+4 for v in out);common=sum(v['common'] for v in out)
    obj=dict(scope='Position count for two full-model branch prefills and four suffix tokens. Excludes query design, precision effects, kernel scheduling and branching overhead. This is not a measured FLOP or wall-time reduction.',distinct_prompt_pairs=len(out),original_positions=original,positions_avoided=common,avoided_fraction=common/original,rows=out)
    (HERE/'prefix_budget.json').write_text(json.dumps(obj,indent=2)+'\n');print(original,common,common/original)
if __name__=='__main__':main()
