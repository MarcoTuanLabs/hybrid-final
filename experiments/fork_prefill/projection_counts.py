"""Estimate logical MACs in the shared model's dense decoder projections."""
import json
from pathlib import Path
import sys
HERE=Path(sys.argv[1]);rows=[json.loads(l) for l in (HERE/'rows.jsonl').read_text().splitlines()]
cfg=json.loads(Path(sys.argv[2]).read_text())['text_config']
d=cfg['hidden_size'];f=3*d*cfg['intermediate_size']
kh=cfg['linear_num_key_heads']*cfg['linear_key_head_dim'];vh=cfg['linear_num_value_heads']*cfg['linear_value_head_dim']
ssm=d*(2*kh+vh+vh+2*cfg['linear_num_value_heads']+vh)
att=d*cfg['head_dim']*(3*cfg['num_attention_heads']+2*cfg['num_key_value_heads'])
kv=2*d*cfg['head_dim']*cfg['num_key_value_heads']
mat=[f+(ssm if x=='linear_attention' else att) for x in cfg['layer_types']]
sums={m:0 for m in rows[0]['arms']}
for r in rows:
    n,t,c=r['prefix'],r['suffix'],r['common'];l=n+t
    base=2*l*sum(mat[:28])+l*sum(mat[28:])+d*(2*kh+vh)
    logical=dict(joint=2*n*(sum(mat[:-1])+kv)+t*sum(mat),window=base,
                 mlp=base-28*c*f,packed=base-c*sum(mat[:28]))
    for m in ('reuse','tail','residual'):
        if m in sums:logical[m]=base-28*(l-(r['refresh'] if m!='reuse' else 0))*f
    for m in sums:sums[m]+=logical[m]
summary=dict(requests=len(rows),logical_macs=sums,per_joint={k:v/sums['joint'] for k,v in sums.items()},
             scope="Logical weight-matrix multiply-accumulates in Q/K/V, gates, output and feed-forward projections. Includes the donor partial projection and final attention K/V in prefix caching. Excludes state updates, attention score products, convolution, output vocabulary, solver, memory traffic and backend tiling. Measures a subset of arithmetic, not total hardware FLOPs.")
(HERE/'projection_cost.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['per_joint'])
