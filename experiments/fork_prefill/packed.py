"""Project shared tokens once and expand into two physical rows only for sequence mixing."""
import copy
import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models.qwen3_5 as qwen
from mlx_lm.models.base import create_attention_mask,scaled_dot_product_attention
from mlx_lm.models.cache import make_prompt_cache

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'write_window'))
from service import UpperLayers


class Fork:
    def __init__(self,n,common):
        self.n=n
        self.common=common

    def expand(self,x):
        assert x.shape[0]==1 and x.shape[1]==2*self.n-self.common
        second=mx.concatenate([x[:,:self.common],x[:,self.n:]],axis=1)
        return mx.concatenate([x[:,:self.n],second],axis=0)

    def pack(self,x):
        assert x.shape[:2]==(2,self.n)
        return mx.concatenate([x[:1],x[1:,self.common:]],axis=1)


def attention(att,x,fork):
    b,p,_=x.shape
    q,gate=mx.split(att.q_proj(x).reshape(b,p,att.num_attention_heads,-1),2,axis=-1)
    k=att.k_proj(x).reshape(b,p,att.num_key_value_heads,-1)
    v=att.v_proj(x).reshape(b,p,att.num_key_value_heads,-1)
    q=att.rope(fork.expand(att.q_norm(q)).transpose(0,2,1,3))
    k=att.rope(fork.expand(att.k_norm(k)).transpose(0,2,1,3))
    v=fork.expand(v).transpose(0,2,1,3)
    y=scaled_dot_product_attention(q,k,v,cache=None,scale=att.scale,mask='causal')
    y=fork.pack(y.transpose(0,2,1,3).reshape(2,fork.n,-1))
    return att.o_proj(y*mx.sigmoid(gate.reshape(b,p,-1)))


def recurrent(att,x,fork):
    b,p,_=x.shape
    qkv=fork.expand(att.in_proj_qkv(x))
    zeros=mx.zeros((2,att.conv_kernel_size-1,att.conv_dim),dtype=x.dtype)
    raw=nn.silu(att.conv1d(mx.concatenate([zeros,qkv],axis=1)))
    q,k,v=[t.reshape(2,fork.n,h,d) for t,h,d in zip(mx.split(raw,[att.key_dim,2*att.key_dim],axis=-1),
                                                      [att.num_k_heads,att.num_k_heads,att.num_v_heads],
                                                      [att.head_k_dim,att.head_k_dim,att.head_v_dim])]
    q,k=qwen.normalize_qk(q,k,inv_scale=att.head_k_dim**-.5,eps=1e-6)
    y,_=qwen.gated_delta_update(q,k,v,fork.expand(att.in_proj_a(x)),fork.expand(att.in_proj_b(x)),
                                att.A_log,att.dt_bias,None,None,use_kernel=not att.training)
    z=att.in_proj_z(x).reshape(b,p,att.num_v_heads,att.head_v_dim)
    return att.out_proj(att.norm(fork.pack(y),z).reshape(b,p,-1))


def lower(layer,x,fork,mode):
    z=layer.input_layernorm(x)
    if mode=='mlp':
        twin=fork.expand(z)
        op=layer.linear_attn if layer.is_linear else layer.self_attn
        y=fork.pack(op(twin,mask=None if layer.is_linear else create_attention_mask(twin,None)))
    else:
        y=recurrent(layer.linear_attn,z,fork) if layer.is_linear else attention(layer.self_attn,z,fork)
    h=x+y
    return h+layer.mlp(layer.post_attention_layernorm(h))


def forward(trial,a,b,tail,active,mode,deps):
    _,_,_,set_write,_,stream=deps
    aa=a+tail;bb=b+tail
    assert len(aa)==len(bb)
    first=next(i for i,(v,w) in enumerate(zip(aa,bb)) if v!=w)
    fork=Fork(len(aa),first);net=trial.model.model
    x=net.embed_tokens(mx.array([aa+bb[first:]]))
    for layer in net.layers[:28]:
        x=lower(layer,x,fork,mode)
    paired=fork.expand(x);recipient=paired[:1];donor=paired[1:,len(b)-1:len(b)]
    cut=net.layers[28]
    slot=cut.linear_attn.in_proj_qkv(cut.input_layernorm(donor))[:,:,trial.lo:trial.lo+256]
    base=make_prompt_cache(trial.model)
    prefix=cut(recipient[:,:len(a)],cache=base[28])
    edit=set_write(trial,base,slot)
    adapter=copy.copy(trial);adapter.ordinal=0
    adapter.model=UpperLayers(trial.model,prefix,recipient[:,len(a):],(create_attention_mask(recipient,None),None),False)
    out,qs=stream(adapter,base,edit,tail,active,'batch_observed')
    return out,qs,slot,base[28][0][:,-1:,trial.lo:trial.lo+256],first
