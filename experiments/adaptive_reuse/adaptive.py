"""Refresh a fixed number of donor MLP contributions using normalized activation drift."""
import copy
from pathlib import Path
import sys
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mlp_reuse'))
from residual import positions,UpperLayers


def forward(trial,a,b,tail,active,mode,deps):
    _,_,_,set_write,_,stream=deps
    aa=a+tail;bb=b+tail
    n=len(aa);p=len(a);fixed=positions(a,b,'residual');k=len(fixed)
    edit={j+d for j,(v,w) in enumerate(zip(a,b)) if v!=w for d in range(4) if j+d<p}
    anchor=mx.array([j in edit or j==p-1 for j in range(p)])
    fixed=mx.array(fixed);offset=mx.arange(k)+n
    net=trial.model.model;x=net.embed_tokens(mx.array([aa,bb]));mask=create_attention_mask(x,None)
    for i,layer in enumerate(net.layers[:28]):
        op=layer.linear_attn if layer.is_linear else layer.self_attn
        mixed=x+op(layer.input_layernorm(x),mask=None if layer.is_linear else mask)
        norm=layer.post_attention_layernorm(mixed)
        if mode=='upper' and i>=24:
            x=mixed+layer.mlp(norm)
            continue
        if mode=='upper':
            idx=fixed
        else:
            diff=norm[0,:p].astype(mx.float32)-norm[1,:p].astype(mx.float32)
            scores=mx.sum(diff*diff,axis=-1)
            if mode=='anchored':scores=mx.where(anchor,mx.array(float('inf')),scores)
            else:assert mode=='drift'
            idx=mx.argsort(-scores)[:k]
        values=layer.mlp(mx.concatenate([norm[:1],mx.take(norm[1:],idx,axis=1)],axis=1))
        take=mx.arange(n);take[idx]=offset
        x=mixed+mx.concatenate([values[:,:n],mx.take(values,take,axis=1)],axis=0)
    recipient=x[:1];donor=x[1:,p-1:p]
    cut=net.layers[28];slot=cut.linear_attn.in_proj_qkv(cut.input_layernorm(donor))[:,:,trial.lo:trial.lo+256]
    base=make_prompt_cache(trial.model);prefix=cut(recipient[:,:p],cache=base[28])
    edit_cache=set_write(trial,base,slot);adapter=copy.copy(trial);adapter.ordinal=0
    adapter.model=UpperLayers(trial.model,prefix,recipient[:,p:],(mask,None),False)
    y,q=stream(adapter,base,edit_cache,tail,active,'batch_observed')
    return y,q,slot,base[28][0][:,-1:,trial.lo:trial.lo+256],k
