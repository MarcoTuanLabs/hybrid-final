"""Check positive-phase inversion on single positions and long windows with the production RoPE."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from cache_rotary import inverse_rotary


def check(rope):
    mx.random.seed(208485)
    records=[]
    for start,n in ((0,1),(24,2),(239,32),(3075,32)):
        values=mx.random.normal((1,4,n,256))
        rotated=rope(values,offset=start)
        recovered=inverse_rotary(rope,rotated,start)
        diff=float(mx.max(mx.abs(recovered-values)).item())
        assert diff<3e-5,(start,n,diff)
        records.append(dict(start=start,length=n,max_float_error=diff))
    return records


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("model",type=Path)
    m,_=load(str(ap.parse_args().model))
    print(json.dumps(check(m.model.layers[3].self_attn.rope),indent=2))
