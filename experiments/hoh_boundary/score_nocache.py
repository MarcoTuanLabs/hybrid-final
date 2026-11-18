"""Recompute the added 9B no-checkpoint sensitivity from its raw measurements."""
import json
from score_boundary import HERE,summarize

def main():
    labels=json.loads((HERE/'test_labels.json').read_text())
    rows=summarize(HERE/'test_nocache.jsonl',labels,list(labels),'full')
    s=json.loads((HERE/'test_summary.json').read_text())
    for name,other in [('full','prefix'),('guarded_unique','guarded_unique')]:
        rows[name]['em_difference_vs_checkpointed']=rows[name]['em']-s['qwen'][other]['em']
    (HERE/'test_nocache_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
