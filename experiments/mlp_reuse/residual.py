"""Approximate the donor's MLP contribution while retaining its own sequence-mixer residual."""
import copy
from pathlib import Path
import sys
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fork_prefill'))
from packed import UpperLayers


def positions(a,b,mode):
    diff=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    assert len(diff)==2 and len(a)==len(b)
    ids={i+s for i in diff for s in range(4) if i+s<len(a)}
    ids.update(range(max(0,len(a)-12),len(a)))
    if mode=='residual':return sorted(ids)
    if mode=='tail':return list(range(len(a)-len(ids),len(a)))
    if mode=='reuse':return []
    raise ValueError(mode)


def forward(trial,a,b,tail,active,mode,deps):
    _,_,_,set_write,_,stream=deps
    aa=a+tail;bb=b+tail;chosen=positions(a,b,mode);n=len(aa)
    if chosen:
        take=mx.array(chosen)
        mapping=list(range(n))
        for j,i in enumerate(chosen):mapping[i]=n+j
        mapping=mx.array(mapping)
    net=trial.model.model
    x=net.embed_tokens(mx.array([aa,bb]));mask=create_attention_mask(x,None)
    for layer in net.layers[:28]:
        norm=layer.input_layernorm(x)
        op=layer.linear_attn if layer.is_linear else layer.self_attn
        mixed=x+op(norm,mask=None if layer.is_linear else mask)
        # Both branches keep their full sequence-mixer outputs, including at reused sites.
        src=mixed[:1] if not chosen else mx.concatenate([mixed[:1],mx.take(mixed[1:],take,axis=1)],axis=1)
        ff=layer.mlp(layer.post_attention_layernorm(src))
        update=ff if not chosen else mx.concatenate([ff[:,:n],mx.take(ff,mapping,axis=1)],axis=0)
        x=mixed+update
    recipient=x[:1];donor=x[1:,len(b)-1:len(b)]
    cut=net.layers[28];slot=cut.linear_attn.in_proj_qkv(cut.input_layernorm(donor))[:,:,trial.lo:trial.lo+256]
    base=make_prompt_cache(trial.model)
    prefix=cut(recipient[:,:len(a)],cache=base[28]);edit=set_write(trial,base,slot)
    adapter=copy.copy(trial);adapter.ordinal=0
    adapter.model=UpperLayers(trial.model,prefix,recipient[:,len(a):],(mask,None),False)
    out,qs=stream(adapter,base,edit,tail,active,'batch_observed')
    return out,qs,slot,base[28][0][:,-1:,trial.lo:trial.lo+256],len(chosen)
