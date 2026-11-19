"""Execute a single-layer native write and its known read suffix in one causal pass."""
import copy
import hashlib
from pathlib import Path
import sys
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask,create_ssm_mask,scaled_dot_product_attention
from mlx_lm.models.cache import make_prompt_cache
import numpy as np


def dependencies(repo):
    paths=["quantized_write","jacobian_service","shared_prefill"]
    sys.path[:0]=[str(repo/"experiments"/p) for p in paths]
    from run import Trial,make_example,host
    from compare import set_write
    from compare_prefix import branch
    from runtime import stream
    return Trial,make_example,host,set_write,branch,stream


def last_attention(layer,x):
    """Run a full-attention block for the final position, retaining every preceding key and value."""
    att=layer.self_attn;xn=layer.input_layernorm(x);last=xn[:,-1:]
    b,n,_=x.shape
    q,gate=mx.split(att.q_proj(last).reshape(b,1,att.num_attention_heads,-1),2,axis=-1)
    q=att.rope(att.q_norm(q).transpose(0,2,1,3),offset=n-1)
    k=att.k_proj(xn).reshape(b,n,att.num_key_value_heads,-1)
    k=att.rope(att.k_norm(k).transpose(0,2,1,3))
    v=att.v_proj(xn).reshape(b,n,att.num_key_value_heads,-1).transpose(0,2,1,3)
    o=scaled_dot_product_attention(q,k,v,cache=None,scale=att.scale,mask=None)
    h=x[:,-1:]+att.o_proj(o.transpose(0,2,1,3).reshape(b,1,-1)*mx.sigmoid(gate.reshape(b,1,-1)))
    return h+layer.mlp(layer.post_attention_layernorm(h))


class UpperLayers:
    def __init__(self,model,prefix,suffix,mask,prune):
        self.text=model
        self.model=model.model
        self.prefix=prefix
        self.suffix=suffix
        self.mask=mask
        self.prune=prune

    def __call__(self,ids,cache):
        assert ids.shape[1]==self.suffix.shape[1]
        net=self.model
        suffix=net.layers[28](self.suffix,cache=cache[28])
        merged=mx.concatenate((self.prefix,suffix),axis=1)
        for i in (29,30,31):
            l=net.layers[i]
            if self.prune and i==31:
                merged=last_attention(l,merged)
            else:
                merged=l(merged,mask=self.mask[int(l.is_linear)])
        h=net.norm(merged[:,-1:]);lm=getattr(self.text,"language_model",self.text)
        return net.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)


def write_window(trial,a,b,tail,active,mode,deps):
    _,_,_,set_write,_,stream=deps
    net=trial.model.model
    full=mx.array([a+tail,b+tail])
    x=net.embed_tokens(full);mask=(create_attention_mask(x,None),create_ssm_mask(x,None))
    prune=mode=="window_last"
    for i in range(27 if prune else 28):
        layer=net.layers[i]
        x=layer(x,mask=mask[int(layer.is_linear)])
    if prune:
        recipient=net.layers[27](x[:1],mask=mask[0])
        donor=last_attention(net.layers[27],x[1:,:len(b)])
    else:
        recipient=x[:1];donor=x[1:,len(b)-1:len(b)]
    cut=net.layers[28]
    donor_slot=cut.linear_attn.in_proj_qkv(cut.input_layernorm(donor))[:,:,trial.lo:trial.lo+256]
    base=make_prompt_cache(trial.model)
    prefix=cut(recipient[:,:len(a)],cache=base[28])
    edited=set_write(trial,base,donor_slot)
    adapter=copy.copy(trial)
    adapter.ordinal=0
    adapter.model=UpperLayers(trial.model,prefix,recipient[:,len(a):],mask,prune)
    output,chosen=stream(adapter,base,edited,tail,active,"batch_observed")
    return output,chosen,donor_slot,base[28][0][:,-1:,trial.lo:trial.lo+256]


def execute(trial,example,mode,active,deps):
    _,_,_,set_write,branch,stream=deps
    mx.synchronize();start=time.perf_counter()
    a,tail=trial.encode(example,"a");b,other=trial.encode(example,"b")
    assert len(a)==len(b) and tail==other and len(tail)==4
    if mode in ("cut","joint"):
        base,slot=branch(trial,a,b,mode)
        edited=set_write(trial,base,slot)
        logits,qs=stream(trial,base,edited,tail,active,"batch_observed")
        original=base[28][0][:,-1:,trial.lo:trial.lo+256]
    else:
        assert mode in ("window","window_last")
        logits,qs,slot,original=write_window(trial,a,b,tail,active,mode,deps)
    top=int(np.argmax(logits));mx.synchronize()
    wall=time.perf_counter()-start
    assert len(qs)==4 and np.isfinite(logits).all()
    a,b=[np.asarray(v.astype(mx.float32)).copy() for v in (slot,original)]
    row=dict(top=top,seconds=wall,query_digest=hashlib.sha256(np.asarray(qs).tobytes()).hexdigest(),
             write_digest=hashlib.sha256(a.tobytes()).hexdigest(),original_digest=hashlib.sha256(b.tobytes()).hexdigest())
    return row,logits,a


def validate(model,repo):
    import json
    for f,h in json.loads((repo/'experiments/read_projection/release.json').read_text())['model_files'].items():
        with (model/f).open('rb') as fp:assert hashlib.file_digest(fp,'sha256').hexdigest()==h
    deps=dependencies(repo);Trial=deps[0]
    t=Trial(model,json.loads((repo/'evidence/quantized_9b/contract.json').read_text()))
    assert len(t.model.model.layers)==32
    return t,deps
