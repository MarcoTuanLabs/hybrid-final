"""Check causal indexing and receiver independence against small FP32 native Qwen blocks."""
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_5 import TextModelArgs,Qwen3_5TextModel
from packed import Fork,lower

def main():
    mx.random.seed(161634)
    args=TextModelArgs(hidden_size=96,intermediate_size=192,num_hidden_layers=4,num_attention_heads=3,num_key_value_heads=1,head_dim=32,linear_num_value_heads=4,linear_num_key_heads=2,linear_key_head_dim=32,linear_value_head_dim=32,vocab_size=128)
    net=Qwen3_5TextModel(args);net.eval()
    rows=[]
    for n,c in ((9,0),(13,6),(18,17)):
        x=mx.random.normal((2,n,96))
        both=mx.concatenate([x[:1],mx.concatenate([x[:1,:c],x[1:,c:]],1)])
        fork=Fork(n,c);assert mx.array_equal(fork.expand(fork.pack(both)),both).item()
        altered=mx.concatenate([both[:1],mx.concatenate([both[1:2,:c],both[1:2,c:]*-2],1)])
        for m in ('mlp','packed'):
            h=fork.pack(both);changed=fork.pack(altered);ref=both;err=[]
            for layer in net.layers:
                ref=layer(ref,mask=None if layer.is_linear else create_attention_mask(ref,None))
                h=lower(layer,h,fork,m);changed=lower(layer,changed,fork,m)
                err.append(float(mx.max(mx.abs(fork.expand(h)-ref)).item()))
                assert mx.array_equal(h[:,:n],changed[:,:n]).item()
            assert max(err)<1e-4,(m,n,c,err)
            rows.append(dict(length=n,common=c,mode=m,layer_max_abs=err,recipient_unchanged=True))
    print(json.dumps(rows))
if __name__=='__main__':main()
